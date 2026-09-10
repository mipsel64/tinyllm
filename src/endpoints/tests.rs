use crate::{
    config::{Config, ProviderConfig, Server},
    providers::openrouter,
    server,
};
use axum::{Json, Router, extract::Request, routing::post};
use serde_json::{Value, json};

async fn serve(router: Router) -> (String, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let task = tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    (url, task)
}

#[tokio::test]
async fn incomplete_bodies_time_out_and_release_admission() {
    use crate::{models::ApiFormat, providers::registry::Registry, server::AppState};
    use axum::{
        body::{Body, Bytes},
        http::StatusCode,
    };
    use std::{
        convert::Infallible,
        sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        },
        time::Duration,
    };
    use tokio::sync::{Notify, Semaphore};

    let calls = Arc::new(AtomicUsize::new(0));
    let observed = calls.clone();
    let (upstream, task) = serve(Router::new().route("/{*path}", post(move || {
        observed.fetch_add(1, Ordering::SeqCst);
        async { Json(json!({"id":"fixture","status":"completed","output":[],"choices":[],"content":[]})) }
    }))).await;
    let config: Config = serde_json::from_value(json!({
        "server":{"request_body_timeout_seconds":1,"max_concurrent_requests":1},
        "providers":{"router":{"type":"openrouter","api_key":"fixture-key","base_url":upstream}}
    }))
    .unwrap();
    config.validate().unwrap();
    let app = Arc::new(AppState {
        providers: Registry::new(&config).await.unwrap(),
        config,
        permits: Some(Arc::new(Semaphore::new(1))),
    });
    for format in [
        ApiFormat::Anthropic,
        ApiFormat::ChatCompletions,
        ApiFormat::Responses,
    ] {
        let reading = Arc::new(Notify::new());
        let started = reading.clone();
        let body = Body::from_stream(async_stream::stream! {
            started.notify_one();
            yield Ok::<_, Infallible>(Bytes::from_static(b"{"));
            futures::future::pending::<()>().await;
        });
        let request = |body| {
            Request::builder()
                .method("POST")
                .header("content-type", "application/json")
                .body(body)
                .unwrap()
        };
        let pending = tokio::spawn(super::endpoint::execute(app.clone(), request(body), format));
        tokio::time::timeout(Duration::from_secs(2), reading.notified())
            .await
            .unwrap();
        let before = calls.load(Ordering::SeqCst);
        let valid = || {
            request(Body::from(
                r#"{"model":"router/fixture","messages":[],"input":"hello"}"#,
            ))
        };
        let busy = super::endpoint::execute(app.clone(), valid(), format).await;
        assert_eq!(busy.status(), StatusCode::SERVICE_UNAVAILABLE);
        let response = tokio::time::timeout(Duration::from_secs(3), pending)
            .await
            .expect("inbound body deadline must fire")
            .unwrap();
        assert_eq!(response.status(), StatusCode::REQUEST_TIMEOUT);
        let bytes = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .unwrap();
        let error: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(error["error"]["message"], "request body timed out");
        assert_eq!(error.get("type").is_some(), format == ApiFormat::Anthropic);
        assert_eq!(calls.load(Ordering::SeqCst), before);
        assert_eq!(app.permits.as_ref().unwrap().available_permits(), 1);
        assert_eq!(
            super::endpoint::execute(app.clone(), valid(), format)
                .await
                .status(),
            StatusCode::OK
        );
        assert_eq!(calls.load(Ordering::SeqCst), before + 1);
    }
    task.abort();
}

#[tokio::test]
async fn endpoint_formats_share_native_provider_routing_and_local_auth() {
    let (upstream,up_task)=serve(Router::new().route("/{*path}",post(|request:Request|async move {
        assert_eq!(request.headers()["authorization"],"Bearer upstream-key");
        assert!(!request.headers().contains_key("x-api-key"));
        let body=axum::body::to_bytes(request.into_body(),10000).await.unwrap();
        let request:Value=serde_json::from_slice(&body).unwrap();
        assert_eq!(request["model"],"deepseek/deepseek-4-pro");
        Json(json!({"id":"fixture","model":request["model"],"status":"completed","output":[],"choices":[],"content":[]}))
    }))).await;
    let config = Config {
        server: Server {
            auth_token: Some("local-key".into()),
            ..Default::default()
        },
        logging: Default::default(),
        providers: [(
            "openrouter".into(),
            ProviderConfig::OpenRouter(openrouter::models::Config {
                api_key: "upstream-key".into(),
                base_url: Some(upstream),
                models: [(
                    "deepseek/deepseek-4-pro".into(),
                    openrouter::models::Model::default(),
                )]
                .into(),
            }),
        )]
        .into(),
    };
    let (gateway, task) = serve(server::router(config).await.unwrap()).await;
    let client = reqwest::Client::new();
    for path in [
        "/anthropic/v1/messages?beta=true",
        "/v1/chat/completions",
        "/v1/responses",
    ] {
        let body = json!({"model":"openrouter/deepseek/deepseek-4-pro","messages":[{"role":"user","content":"hello"}],"input":"hello","max_tokens":20});
        let response = client
            .post(format!("{gateway}{path}"))
            .bearer_auth("local-key")
            .json(&body)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        assert_eq!(
            response.json::<Value>().await.unwrap()["model"],
            body["model"]
        );
        let unauthorized = client
            .post(format!("{gateway}{path}"))
            .json(&body)
            .send()
            .await
            .unwrap();
        assert_eq!(unauthorized.status(), 401);
        let error = unauthorized.json::<Value>().await.unwrap();
        assert_eq!(error["error"]["type"], "authentication_error");
        assert_eq!(error.get("type").is_some(), path.starts_with("/anthropic"));
        for model in [
            "deepseek/deepseek-4-pro",
            "openrouter/",
            "openrouter/deepseek//model",
            "unconfigured/model",
        ] {
            let response = client
                .post(format!("{gateway}{path}"))
                .bearer_auth("local-key")
                .json(&json!({"model":model}))
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), 400, "{model}");
        }
    }
    for path in ["/anthropic/v1/models", "/v1/models"] {
        let response = client
            .get(format!("{gateway}{path}"))
            .bearer_auth("local-key")
            .send()
            .await
            .unwrap()
            .json::<Value>()
            .await
            .unwrap();
        assert_eq!(
            response["data"][0]["id"],
            "openrouter/deepseek/deepseek-4-pro"
        );
        if path == "/v1/models" {
            assert_eq!(response["object"], "list");
        }
    }
    task.abort();
    up_task.abort();
}
