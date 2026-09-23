use super::{Provider, anthropic, http, openrouter, zai};
use crate::{
    config::Server,
    models::{ApiEvent, ApiFormat, ApiRequest, RequestContext, ResponseBody},
};
use axum::{
    Json, Router,
    body::{Body, Bytes},
    http::{HeaderMap, StatusCode, Uri},
    routing::post,
};
use futures::{StreamExt, stream};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};

fn context(model: &str) -> RequestContext {
    let mut headers = HeaderMap::new();
    headers.insert("authorization", "Bearer local-secret".parse().unwrap());
    headers.insert("x-api-key", "local-secret".parse().unwrap());
    headers.insert("anthropic-version", "2023-06-01".parse().unwrap());
    headers.insert("anthropic-beta", "test-beta".parse().unwrap());
    headers.insert("x-private", "private".parse().unwrap());
    RequestContext {
        request_id: "req-local".into(),
        provider: "router".into(),
        model: model.into(),
        public_model: format!("router/{model}"),
        headers,
        query: Some("beta=true&value=a%2Fb".into()),
    }
}

#[test]
fn native_model_options_validate_effort() {
    for provider in ["openai", "openrouter", "zai"] {
        for effort in [
            json!(null),
            json!("none"),
            json!("minimal"),
            json!("low"),
            json!("medium"),
            json!("high"),
            json!("xhigh"),
            json!("max"),
            json!("invalid"),
            json!("High"),
            json!(""),
            json!(true),
            json!(42),
        ] {
            let valid = effort.is_null()
                || matches!(
                    effort.as_str(),
                    Some("none" | "minimal" | "low" | "medium" | "high" | "xhigh" | "max")
                );
            let mut provider_config = if provider == "openai" {
                json!({"type":provider,"auth":{"type":"ApiKey","options":"fixture-key"}})
            } else {
                json!({"type":provider,"api_key":"fixture-key"})
            };
            provider_config["models"] = json!({"native/model":{"reasoning_effort":effort}});
            let config = serde_json::from_value::<crate::config::Config>(json!({
                "providers":{"fixture":provider_config}
            }));
            assert_eq!(
                config.is_ok(),
                valid,
                "deserialize {provider} effort {effort}"
            );
            if let Ok(config) = &config {
                use crate::config::ProviderConfig;
                let configured = match &config.providers["fixture"] {
                    ProviderConfig::Anthropic(_) => {
                        unreachable!("anthropic models take no options")
                    }
                    ProviderConfig::OpenAi(c) => c.models["native/model"].reasoning_effort,
                    ProviderConfig::OpenRouter(c) => c.models["native/model"].reasoning_effort,
                    ProviderConfig::Zai(c) => c.models["native/model"].reasoning_effort,
                };
                assert_eq!(json!(configured), effort);
                assert_eq!(configured.map(|e| e.as_str()), effort.as_str());
            }
            assert_eq!(config.is_ok_and(|config| config.validate().is_ok()), valid);
        }
    }
}

#[tokio::test]
async fn provider_trait_maps_effort_for_the_selected_model() {
    use crate::providers::registry::Registry;
    let directory = std::env::temp_dir().join(format!("tinyllm-effort-{}", uuid::Uuid::new_v4()));
    let config: crate::config::Config = serde_json::from_value(json!({
        "server":{"state_dir":directory},
        "providers":{
            "openai":{"type":"openai","auth":{"type":"ApiKey","options":"fixture-key"},"models":{"gpt-5.6-sol":{"reasoning_effort":"max"}}},
            "router":{"type":"openrouter","api_key":"fixture-key"},
            "zai":{"type":"zai","api_key":"fixture-key"}
        }
    })).unwrap();
    config.validate().unwrap();
    let registry = Registry::new(&config).await.unwrap();
    for (model, effort, expected) in [
        ("openai/gpt-5.6-sol", "max", Some("max")),
        ("openai/gpt-6-astra", "max", Some("max")),
        ("openai/gpt-5.4", "max", Some("xhigh")),
        ("openai/gpt-5.4-2026-03-05", "max", Some("xhigh")),
        ("openai/gpt-5.40", "max", Some("max")),
        ("openai/gpt-5.4", "low", Some("low")),
        ("openai/gpt-6-astra", "none", Option::None),
        ("router/deepseek/deepseek-4-pro", "max", Some("max")),
        ("router/anthropic/claude-opus-4.6", "medium", Some("medium")),
        ("zai/glm-5.3", "medium", Some("high")),
        ("zai/glm-5.3-flash", "xhigh", Some("max")),
        ("zai/glm-5.3", "low", Some("low")),
        ("zai/glm-5.3", "none", Option::None),
        ("zai/glm-5.3", "minimal", Option::None),
        ("zai/glm-5.2", "low", Some("high")),
        ("zai/glm-5.2", "minimal", Some("minimal")),
        ("zai/glm-5.1", "high", Option::None),
        ("zai/glm-future", "max", Some("max")),
    ] {
        let (provider, native) = registry.resolve(model).unwrap();
        match provider.convert_reasoning_effort(&native, effort) {
            Ok(actual) => assert_eq!(Some(actual.as_str()), expected, "{model} {effort:?}"),
            Err(error) => {
                assert!(expected.is_none(), "{model} {effort:?}");
                assert_eq!(error.status, StatusCode::BAD_REQUEST);
            }
        }
    }
    drop(registry);
    let _ = std::fs::remove_dir_all(directory);
}

fn request(format: ApiFormat, model: &str, stream: bool) -> ApiRequest {
    ApiRequest::parse(
        format,
        json!({
            "model": format!("router/{model}"), "stream": stream,
            "messages": [{"role":"user","content":"tinyllm:v1: is a marker to discuss"}],
            "max_tokens": 128, "provider_extension": {"keep":true},
        }),
    )
    .unwrap()
}

#[tokio::test]
async fn native_routes_preserve_models_extensions_and_allowed_headers() {
    let (sender, mut receiver) = tokio::sync::mpsc::channel(6);
    let app = Router::new().fallback(post(move |uri: Uri, headers: HeaderMap, Json(body): Json<Value>| {
        let sender = sender.clone();
        async move {
            let model = body["model"].as_str().unwrap();
            let streaming = body["stream"] == true;
            let response = if streaming {
                let values = if uri.path().ends_with("/messages") {
                    vec![json!({"type":"message_start","message":{"model":model}}),json!({"type":"message_stop","provider_extension":{"keep":true}})]
                } else if uri.path().ends_with("/responses") {
                    vec![json!({"type":"response.completed","response":{"model":model,"status":"completed"},"provider_extension":{"keep":true}})]
                } else {
                    vec![json!({"model":model,"choices":[{"index":0,"delta":{"content":"hello"},"finish_reason":"stop"}],"provider_extension":{"keep":true}})]
                };
                let mut wire = values.iter().map(|value|format!("data: {value}\n\n")).collect::<String>();
                if uri.path().ends_with("/chat/completions") { wire.push_str("data: [DONE]\n\n"); }
                let chunks = wire.as_bytes().chunks(3).map(|bytes|Ok::<_,std::io::Error>(Bytes::copy_from_slice(bytes))).collect::<Vec<_>>();
                Body::from_stream(stream::iter(chunks))
            } else {
                Body::from(json!({"model":model,"status":"completed","provider_extension":{"keep":true},"content":[]}).to_string())
            };
            sender.send((uri, headers, body)).await.unwrap();
            ([ ("x-request-id", "req-upstream"), ("retry-after", "3"), ("x-private", "hidden"), ("content-type", if streaming {"text/event-stream; charset=utf-8"} else {"application/json"}) ], response)
        }
    }));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let server = Server::default();
    let client = http::client(&server).unwrap();
    let router = openrouter::OpenRouterProvider::new(
        openrouter::Config {
            api_key: "upstream-secret".into(),
            user_agent: Some("openrouter-client/test".into()),
            base_url: Some(format!("{base}/api/v1/")),
            models: BTreeMap::from([(
                "deepseek/deepseek-4-pro".into(),
                serde_json::from_value(json!({"reasoning_effort":"high"})).unwrap(),
            )]),
        },
        client.clone(),
        server.clone(),
    )
    .unwrap();
    let zai = zai::ZaiProvider::new(
        zai::Config {
            api_key: "upstream-secret".into(),
            user_agent: Some("zai-client/test".into()),
            base_url: Some(format!("{base}/api/")),
            models: BTreeMap::from([(
                "glm-5.3".into(),
                serde_json::from_value(json!({"reasoning_effort":"medium"})).unwrap(),
            )]),
        },
        client,
        server,
    )
    .unwrap();
    assert_eq!(router.models()[0].id, "deepseek/deepseek-4-pro");
    for (provider, model, format, path) in [
        (
            &router as &dyn Provider,
            "deepseek/deepseek-4-pro",
            ApiFormat::Anthropic,
            "/api/v1/messages",
        ),
        (
            &router,
            "deepseek/deepseek-4-pro",
            ApiFormat::ChatCompletions,
            "/api/v1/chat/completions",
        ),
        (
            &router,
            "deepseek/deepseek-4-pro",
            ApiFormat::Responses,
            "/api/v1/responses",
        ),
        (
            &zai,
            "glm-5.3",
            ApiFormat::Anthropic,
            "/api/anthropic/v1/messages",
        ),
        (
            &zai,
            "glm-5.3",
            ApiFormat::ChatCompletions,
            "/api/coding/paas/v4/chat/completions",
        ),
        (&zai, "glm-5.3", ApiFormat::Responses, "/api/v1/responses"),
    ] {
        let output = provider
            .execute(request(format, model, false), context(model))
            .await
            .unwrap();
        assert_eq!(output.headers["x-request-id"], "req-upstream");
        assert_eq!(output.headers["retry-after"], "3");
        assert!(!output.headers.contains_key("x-private"));
        let ResponseBody::Json(body) = output.body else {
            panic!("expected JSON")
        };
        assert_eq!(body["model"], format!("router/{model}"));
        assert_eq!(body["provider_extension"], json!({"keep":true}));
        let (uri, headers, body) = receiver.recv().await.unwrap();
        assert_eq!(uri.path(), path);
        assert_eq!(uri.query(), Some("beta=true&value=a%2Fb"));
        assert_eq!(headers["authorization"], "Bearer upstream-secret");
        assert_eq!(
            headers["user-agent"],
            if model == "glm-5.3" {
                "zai-client/test"
            } else {
                "openrouter-client/test"
            }
        );
        assert!(!headers.contains_key("x-api-key"));
        assert!(!headers.contains_key("x-private"));
        assert_eq!(
            headers.contains_key("anthropic-beta"),
            format == ApiFormat::Anthropic
        );
        if format == ApiFormat::Anthropic {
            assert_eq!(headers["anthropic-version"], "2023-06-01");
            assert_eq!(headers["anthropic-beta"], "test-beta");
        }
        assert_eq!(body["model"], model);
        assert_eq!(body["provider_extension"], json!({"keep":true}));
        let effort_pointer = match format {
            ApiFormat::Anthropic => "/output_config/effort",
            ApiFormat::Responses => "/reasoning/effort",
            ApiFormat::ChatCompletions if model == "glm-5.3" => "/reasoning_effort",
            ApiFormat::ChatCompletions => "/reasoning/effort",
        };
        assert_eq!(body.pointer(effort_pointer), Some(&json!("high")));
        let output = provider
            .execute(request(format, model, true), context(model))
            .await
            .unwrap();
        let ResponseBody::Stream(mut events) = output.body else {
            panic!("expected SSE")
        };
        let mut values = Vec::new();
        let mut done = false;
        while let Some(event) = events.next().await {
            match event.unwrap() {
                ApiEvent::Anthropic(v) | ApiEvent::ChatCompletions(v) | ApiEvent::Responses(v) => {
                    values.push(v)
                }
                ApiEvent::Done => done = true,
            }
        }
        let returned_model = match format {
            ApiFormat::Anthropic => &values[0]["message"]["model"],
            ApiFormat::ChatCompletions => &values[0]["model"],
            ApiFormat::Responses => &values[0]["response"]["model"],
        };
        assert_eq!(returned_model, &json!(format!("router/{model}")));
        assert_eq!(
            values.last().unwrap()["provider_extension"],
            json!({"keep":true})
        );
        assert_eq!(done, format == ApiFormat::ChatCompletions);
        let (_, _, body) = receiver.recv().await.unwrap();
        assert_eq!(body["stream"], true);
        assert_eq!(body.pointer(effort_pointer), Some(&json!("high")));

        let mut explicit = request(format, model, false).into_value();
        let (container, key) = effort_pointer[1..]
            .split_once('/')
            .unwrap_or(("", "reasoning_effort"));
        for (client_effort, provider_effort) in [
            ("medium", if model == "glm-5.3" { "high" } else { "medium" }),
            ("xhigh", if model == "glm-5.3" { "max" } else { "xhigh" }),
            ("max", "max"),
        ] {
            let mut body = request(format, model, false).into_value();
            if container.is_empty() {
                body[key] = json!(client_effort);
            } else {
                body[container] = json!({key: client_effort, "keep": true});
            }
            provider
                .execute(
                    ApiRequest::parse(format, body.clone()).unwrap(),
                    context(model),
                )
                .await
                .unwrap();
            body["model"] = json!(model);
            *body.pointer_mut(effort_pointer).unwrap() = json!(provider_effort);
            assert_eq!(receiver.recv().await.unwrap().2, body);
        }
        for invalid in [json!("unsupported"), json!(42)] {
            let mut body = request(format, model, false).into_value();
            if container.is_empty() {
                body[key] = invalid;
            } else {
                body[container] = json!({key: invalid});
            }
            let error = provider
                .execute(ApiRequest::parse(format, body).unwrap(), context(model))
                .await
                .err()
                .unwrap();
            assert_eq!(error.status, StatusCode::BAD_REQUEST);
            assert!(receiver.try_recv().is_err());
        }
        if container.is_empty() {
            explicit[key] = json!("low");
        } else {
            explicit[container] = json!({key: "low", "keep": true});
        }
        for overrides in [
            explicit,
            json!({"reasoning_effort":null}),
            json!({"reasoning":{"effort":null}}),
            json!({"reasoning":{"enabled":false}}),
            json!({"reasoning":{"max_tokens":2048}}),
            json!({"reasoning":null}),
            json!({"thinking":{"type":"disabled"}}),
            json!({"thinking":{"type":"enabled","budget_tokens":2048}}),
            json!({"thinking":null}),
            json!({"output_config":{"effort":null}}),
            json!({"output_config":null}),
        ] {
            let mut expected = request(format, model, false).into_value();
            expected
                .as_object_mut()
                .unwrap()
                .extend(overrides.as_object().unwrap().clone());
            provider
                .execute(
                    ApiRequest::parse(format, expected.clone()).unwrap(),
                    context(model),
                )
                .await
                .unwrap();
            expected["model"] = json!(model);
            assert_eq!(receiver.recv().await.unwrap().2, expected);
        }

        let mut expected = request(format, model, false).into_value();
        expected["thinking"] = json!({"type":"adaptive"});
        expected["reasoning"] = json!({"summary":"auto"});
        expected["output_config"] = json!({"format":{"type":"json_schema"}});
        provider
            .execute(
                ApiRequest::parse(format, expected.clone()).unwrap(),
                context(model),
            )
            .await
            .unwrap();
        expected["model"] = json!(model);
        if container.is_empty() {
            expected[key] = json!("high");
        } else {
            expected[container][key] = json!("high");
        }
        assert_eq!(receiver.recv().await.unwrap().2, expected);

        provider
            .execute(
                request(format, "unlisted/model", false),
                context("unlisted/model"),
            )
            .await
            .unwrap();
        assert!(
            receiver
                .recv()
                .await
                .unwrap()
                .2
                .pointer(effort_pointer)
                .is_none()
        );
    }
    task.abort();
}

fn decode(
    data: &str,
    format: ApiFormat,
    limit: usize,
) -> futures::stream::BoxStream<'static, crate::Result<ApiEvent>> {
    let chunks = data
        .as_bytes()
        .chunks(3)
        .map(|bytes| Ok::<_, std::io::Error>(Bytes::copy_from_slice(bytes)))
        .collect::<Vec<_>>();
    http::decode_native(
        stream::iter(chunks),
        format,
        "router/native/model".into(),
        "secret".into(),
        limit,
    )
}

#[tokio::test]
async fn native_fragmented_tool_events_stay_incremental_and_preserve_fields() {
    let events = [
        json!({"type":"message_start","message":{"model":"native/model","usage":{"input_tokens":3,"cache_read_input_tokens":2}}}),
        json!({"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"tool_1","name":"lookup","input":{}}}),
        json!({"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"city\":\"Hà"}}),
        json!({"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":" Nội\"}"}}),
        json!({"type":"content_block_stop","index":0}),
        json!({"type":"message_stop","provider_extension":{"keep":true}}),
    ];
    let wire = events
        .iter()
        .map(|v| {
            format!(
                "event: {}\r\ndata: {v}\r\n\r\n",
                v["type"].as_str().unwrap()
            )
        })
        .collect::<String>();
    let actual = decode(&wire, ApiFormat::Anthropic, wire.len())
        .collect::<Vec<_>>()
        .await;
    assert_eq!(actual.len(), events.len());
    for (i, event) in actual.into_iter().enumerate() {
        let ApiEvent::Anthropic(value) = event.unwrap() else {
            panic!("expected Anthropic")
        };
        let mut expected = events[i].clone();
        if i == 0 {
            expected["message"]["model"] = json!("router/native/model");
        }
        assert_eq!(value, expected);
    }
    let mut chat = decode(
        "data: {\"model\":\"native/model\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\"}}]}}]}\n\ndata: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\ndata: [DONE]\n\n",
        ApiFormat::ChatCompletions,
        4096,
    );
    let ApiEvent::ChatCompletions(value) = chat.next().await.unwrap().unwrap() else {
        panic!("expected Chat")
    };
    assert_eq!(value["model"], "router/native/model");
    assert_eq!(
        value["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"],
        "{"
    );
    assert!(matches!(
        chat.next().await.unwrap().unwrap(),
        ApiEvent::ChatCompletions(_)
    ));
    assert!(matches!(
        chat.next().await.unwrap().unwrap(),
        ApiEvent::Done
    ));
    assert!(chat.next().await.is_none());
}

#[tokio::test]
async fn native_streams_reject_failures_truncation_and_wire_limit() {
    for (format, wire, limit) in [
        (ApiFormat::ChatCompletions, "data: [DONE]\n\n", 4096),
        (
            ApiFormat::ChatCompletions,
            "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":null}]}\n\ndata: [DONE]\n\n",
            4096,
        ),
        (
            ApiFormat::ChatCompletions,
            "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"},{\"index\":1,\"delta\":{}}]}\n\ndata: [DONE]\n\n",
            4096,
        ),
        (
            ApiFormat::Responses,
            "data: {\"type\":\"response.completed\",\"response\":{\"status\":\"failed\",\"error\":{\"message\":\"secret failed\"}}}\n\n",
            4096,
        ),
        (
            ApiFormat::Responses,
            "data: {\"type\":\"response.completed\",\"response\":{\"status\":\"in_progress\"}}}\n\n",
            4096,
        ),
        (
            ApiFormat::Anthropic,
            "data: {\"type\":\"message_start\"}\n\n",
            4096,
        ),
        (
            ApiFormat::ChatCompletions,
            "data: {\"choices\":[]}\n\n",
            4096,
        ),
        (
            ApiFormat::Responses,
            "data: {\"type\":\"response.created\"}\n\n",
            4096,
        ),
        (ApiFormat::Responses, "data: [DONE]\n\n", 4096),
        (
            ApiFormat::Responses,
            "data: {\"type\":\"response.failed\",\"response\":{\"error\":{\"message\":\"secret failed\"}}}\n\n",
            4096,
        ),
        (
            ApiFormat::Anthropic,
            "event: error\ndata: {\"type\":\"error\",\"error\":{\"message\":\"secret failed\"}}\n\n",
            4096,
        ),
        (
            ApiFormat::ChatCompletions,
            "data: {\"error\":{\"message\":\"secret failed\"}}\n\n",
            4096,
        ),
        (
            ApiFormat::Anthropic,
            "event: message_stop\ndata: {\"type\":\"message_start\"}\n\n",
            4096,
        ),
        (
            ApiFormat::ChatCompletions,
            ": long keepalive comment\n\ndata: [DONE]\n\n",
            15,
        ),
        (ApiFormat::ChatCompletions, "data: {broken json}\n\n", 4096),
    ] {
        let events = decode(wire, format, limit).collect::<Vec<_>>().await;
        let error = events
            .last()
            .unwrap()
            .as_ref()
            .err()
            .expect("expected terminal error");
        assert!(!error.message.contains("secret"));
        assert_eq!(events.iter().filter(|event| event.is_err()).count(), 1);
    }
    for status in ["completed", "incomplete"] {
        let wire = format!(
            "data: {{\"type\":\"response.{status}\",\"response\":{{\"model\":\"native/model\",\"status\":\"{status}\"}}}}\n\n"
        );
        let events = decode(&wire, ApiFormat::Responses, 4096)
            .collect::<Vec<_>>()
            .await;
        assert_eq!(events.len(), 1);
        let ApiEvent::Responses(value) = events.into_iter().next().unwrap().unwrap() else {
            panic!("expected Responses")
        };
        assert_eq!(value["response"]["model"], "router/native/model");
    }
}

#[tokio::test]
async fn dropping_native_stream_cancels_source_without_waiting_for_completion() {
    struct Dropped(Arc<AtomicBool>);
    impl Drop for Dropped {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }
    let dropped = Arc::new(AtomicBool::new(false));
    let guard = Dropped(dropped.clone());
    let source = async_stream::stream! {
        let _guard = guard;
        yield Ok::<_, std::io::Error>(Bytes::from_static(b"data: {\"choices\":[]}\n\n"));
        futures::future::pending::<()>().await;
    };
    let mut output = http::decode_native(
        source,
        ApiFormat::ChatCompletions,
        "router/model".into(),
        "secret".into(),
        4096,
    );
    tokio::time::timeout(std::time::Duration::from_secs(1), output.next())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    drop(output);
    assert!(dropped.load(Ordering::SeqCst));
}

#[tokio::test]
async fn native_strips_foreign_reasoning_carriers_before_network() {
    let server = Server::default();
    let client = http::client(&server).unwrap();
    let provider = zai::ZaiProvider::new(
        zai::Config {
            api_key: "secret".into(),
            user_agent: None,
            base_url: Some("http://127.0.0.1:1".into()),
            models: BTreeMap::new(),
        },
        client,
        server,
    )
    .unwrap();
    // A model switch carries OpenAI reasoning into a provider that cannot read
    // it. The carrier is dropped so the rest of the turn still forwards.
    let request = ApiRequest::parse(
        ApiFormat::Anthropic,
        json!({"model":"router/glm","messages":[{"role":"assistant","content":[
            {"type":"redacted_thinking","data":"tinyllm:v1:Zm9yZWlnbg:opaque"},
            {"type":"text","text":"visible"}
        ]}]}),
    )
    .unwrap();
    let result = provider.execute(request, context("glm")).await;
    // The upstream address is unroutable, so reaching the network is the proof
    // that translation accepted the carrier instead of rejecting it.
    assert_eq!(result.err().unwrap().status, StatusCode::BAD_GATEWAY);
    for (format, body, status) in [
        (
            ApiFormat::Responses,
            json!({"model":"router/glm","background":true,"input":"hello"}),
            StatusCode::BAD_REQUEST,
        ),
        (
            ApiFormat::Responses,
            json!({"model":"router/glm","messages":[{"role":"assistant","reasoning_details":[{"type":"tinyllm_continuation","data":"tinyllm:v1:Zm9yZWlnbg:opaque"}]}]}),
            StatusCode::BAD_GATEWAY,
        ),
        (
            ApiFormat::ChatCompletions,
            json!({"model":"router/glm","messages":[{"role":"tool","tool_call_id":"call_any","content":"ok"}]}),
            StatusCode::BAD_GATEWAY,
        ),
    ] {
        let request = ApiRequest::parse(format, body).unwrap();
        let result = provider.execute(request, context("glm")).await;
        assert_eq!(result.err().unwrap().status, status);
    }
}

#[tokio::test]
async fn native_http_errors_preserve_status_headers_and_redact_credentials() {
    let app = Router::new().fallback(post(|uri: Uri| async move {
        let (status, body) = match uri.path() {
            "/rate/chat/completions" => (
                StatusCode::TOO_MANY_REQUESTS,
                r#"{"error":{"message":"upstream-secret is rate limited"}}"#.to_owned(),
            ),
            "/failed/chat/completions" => (
                StatusCode::OK,
                r#"{"status":"failed","error":{"message":"upstream-secret failed"}}"#.to_owned(),
            ),
            "/limit/chat/completions" => (StatusCode::OK, " ".repeat(4097)),
            "/queued/responses" => (
                StatusCode::OK,
                r#"{"status":"queued","model":"native/model"}"#.into(),
            ),
            "/completed/responses" => (
                StatusCode::OK,
                r#"{"status":"completed","model":"native/model"}"#.into(),
            ),
            _ => (StatusCode::OK, "invalid JSON".into()),
        };
        (
            status,
            [("retry-after", "7"), ("x-request-id", "req-error")],
            body,
        )
    }));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let server = Server {
        max_response_bytes: 4096,
        ..Server::default()
    };
    let client = http::client(&server).unwrap();
    for (path, status) in [
        ("rate", StatusCode::TOO_MANY_REQUESTS),
        ("failed", StatusCode::BAD_GATEWAY),
        ("limit", StatusCode::BAD_GATEWAY),
        ("invalid", StatusCode::BAD_GATEWAY),
        ("queued", StatusCode::BAD_GATEWAY),
        ("completed", StatusCode::BAD_GATEWAY),
    ] {
        let provider = openrouter::OpenRouterProvider::new(
            openrouter::Config {
                api_key: "upstream-secret".into(),
                user_agent: None,
                base_url: Some(format!("{base}/{path}")),
                models: BTreeMap::new(),
            },
            client.clone(),
            server.clone(),
        )
        .unwrap();
        let format = if matches!(path, "queued" | "completed") {
            ApiFormat::Responses
        } else {
            ApiFormat::ChatCompletions
        };
        let error = provider
            .execute(
                request(format, "native/model", path == "completed"),
                context("native/model"),
            )
            .await
            .err()
            .unwrap();
        assert_eq!(error.status, status);
        assert_eq!(error.headers["retry-after"], "7");
        assert_eq!(error.headers["x-request-id"], "req-error");
        assert!(!error.message.contains("upstream-secret"));
        if path == "rate" {
            assert_eq!(error.kind, "rate_limit_error");
        }
    }
    task.abort();
}

#[test]
fn native_error_mapping_preserves_compaction_and_server_credential_semantics() {
    let error = http::upstream_error(
        StatusCode::BAD_REQUEST,
        &json!({"error":{"code":"context_length_exceeded","message":"secret too long"}}),
        "secret",
    );
    assert!(
        error
            .message
            .starts_with("capability_rejected: prompt_too_long; ")
    );
    assert!(!error.message.contains("secret"));
    for status in [
        StatusCode::UNAUTHORIZED,
        StatusCode::FORBIDDEN,
        StatusCode::SERVICE_UNAVAILABLE,
        StatusCode::from_u16(529).unwrap(),
    ] {
        let error = http::upstream_error(status, &json!({"detail":"secret unavailable"}), "secret");
        assert_eq!(error.status, StatusCode::BAD_GATEWAY);
        assert_eq!(error.message, "[redacted] unavailable");
        if status.as_u16() >= 500 {
            assert_eq!(error.kind, "overloaded_error");
        }
    }
}

fn anthropic_provider(base: &str, key: &str) -> anthropic::AnthropicProvider {
    let server = Server::default();
    anthropic::AnthropicProvider::new(
        serde_json::from_value(json!({
            "auth": {"type": "ApiKey", "options": key},
            "user_agent": "anthropic-client/test",
            "base_url": base,
            "models": {"claude-sonnet-4-6": {}},
        }))
        .unwrap(),
        http::client(&server).unwrap(),
        server,
    )
    .unwrap()
}

#[tokio::test]
async fn anthropic_native_sends_api_key_and_round_trips_thinking_and_unknown_fields() {
    let native_events = vec![
        json!({
            "type":"message_start",
            "message":{"model":"claude-sonnet-4-6","content":"hi"}
        }),
        json!({
            "type":"content_block_start", "index":0,
            "content_block":{"type":"thinking","thinking":""}
        }),
        json!({
            "type":"content_block_delta", "index":0,
            "delta":{"type":"thinking_delta","thinking":"step"},
            "provider_extension":{"keep":true}
        }),
        json!({
            "type":"content_block_delta", "index":0,
            "delta":{"type":"signature_delta","signature":"ErUBCkYIBRgCKkB0aGlzLWlz"}
        }),
        json!({"type":"content_block_stop","index":0}),
        json!({
            "type":"content_block_start", "index":1,
            "content_block":{"type":"tool_use","id":"toolu_01","name":"lookup","input":{}}
        }),
        json!({
            "type":"content_block_delta", "index":1,
            "delta":{"type":"input_json_delta","partial_json":"{\"city\":\"Hà Nội\"}"}
        }),
        json!({"type":"content_block_stop","index":1}),
        json!({
            "type":"message_delta",
            "delta":{"stop_reason":"tool_use","stop_sequence":null},
            "usage":{"output_tokens":12}
        }),
        json!({"type":"message_stop"}),
    ];
    let fixture_events = native_events.clone();
    let (sender, mut receiver) = tokio::sync::mpsc::channel(4);
    let app = Router::new().fallback(post(
        move |uri: Uri, headers: HeaderMap, Json(body): Json<Value>| {
            let sender = sender.clone();
            let fixture_events = fixture_events.clone();
            async move {
                let (status, content_type, payload) = if uri.path().starts_with("/rate") {
                    (
                        StatusCode::TOO_MANY_REQUESTS,
                        "application/json",
                        json!({"error":{"type":"rate_limit_error","message":"upstream-secret is rate limited"}})
                            .to_string(),
                    )
                } else if body["stream"] == true {
                    (
                        StatusCode::OK,
                        "text/event-stream",
                        fixture_events
                            .iter()
                            .map(|event| {
                                format!(
                                    "event: {}\ndata: {event}\n\n",
                                    event["type"].as_str().unwrap()
                                )
                            })
                            .collect(),
                    )
                } else {
                    // Echoing the request proves both directions keep the blocks
                    // and fields tinyllm does not model.
                    let mut echo = body.clone();
                    echo["type"] = json!("message");
                    (StatusCode::OK, "application/json", echo.to_string())
                };
                sender.send((uri, headers, body)).await.unwrap();
                (status, [("content-type", content_type)], payload)
            }
        },
    ));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let provider = anthropic_provider(&format!("{base}/v1"), "upstream-secret");
    assert_eq!(provider.models()[0].id, "claude-sonnet-4-6");

    // A true signature, a provider-owned redacted_thinking block that is not a
    // tinyllm carrier, and a field tinyllm does not know.
    let blocks = json!([
        {"type":"thinking","thinking":"step","signature":"ErUBCkYIBRgCKkB0aGlzLWlz"},
        {"type":"redacted_thinking","data":"EroBCkYIBRgCKkBuYXRpdmUtb3BhcXVl"},
        {"type":"text","text":"answer","cache_control":{"type":"ephemeral"}},
        {"type":"tool_use","id":"toolu_01","name":"lookup","input":{"city":"Hà Nội"}}
    ]);
    let sent = json!({
        "model": "router/claude-sonnet-4-6", "max_tokens": 128,
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": blocks},
            {"role": "user", "content": [{"type":"tool_result","tool_use_id":"toolu_01","content":"ok"}]},
        ],
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "high"},
        "provider_extension": {"keep": true},
    });
    let output = provider
        .execute(
            ApiRequest::parse(ApiFormat::Anthropic, sent.clone()).unwrap(),
            context("claude-sonnet-4-6"),
        )
        .await
        .unwrap();
    let ResponseBody::Json(returned) = output.body else {
        panic!("expected JSON")
    };
    let (uri, headers, upstream) = receiver.recv().await.unwrap();
    assert_eq!(uri.path(), "/v1/messages");
    assert_eq!(uri.query(), Some("beta=true&value=a%2Fb"));
    assert_eq!(headers["x-api-key"], "upstream-secret");
    assert!(!headers.contains_key("authorization"));
    assert_eq!(headers["anthropic-version"], "2023-06-01");
    assert_eq!(headers["anthropic-beta"], "test-beta");
    assert_eq!(headers["user-agent"], "anthropic-client/test");
    assert!(!headers.contains_key("x-private"));
    let mut expected = sent.clone();
    expected["model"] = json!("claude-sonnet-4-6");
    expected["stream"] = json!(false);
    assert_eq!(upstream, expected);
    expected["model"] = json!("router/claude-sonnet-4-6");
    expected["type"] = json!("message");
    assert_eq!(returned, expected);

    // Without a client anthropic-version the shared default is sent.
    let mut plain = context("claude-sonnet-4-6");
    plain.headers.remove("anthropic-version");
    plain.headers.remove("anthropic-beta");
    let mut streamed = sent.clone();
    streamed["stream"] = json!(true);
    let output = provider
        .execute(
            ApiRequest::parse(ApiFormat::Anthropic, streamed).unwrap(),
            plain,
        )
        .await
        .unwrap();
    let ResponseBody::Stream(events) = output.body else {
        panic!("expected SSE")
    };
    let returned_events = events
        .map(|event| match event.unwrap() {
            ApiEvent::Anthropic(value) => value,
            _ => panic!("expected Anthropic"),
        })
        .collect::<Vec<_>>()
        .await;
    let mut expected_events = native_events;
    expected_events[0]["message"]["model"] = json!("router/claude-sonnet-4-6");
    assert_eq!(returned_events, expected_events);
    let (_, headers, _) = receiver.recv().await.unwrap();
    assert_eq!(headers["anthropic-version"], "2023-06-01");
    assert!(!headers.contains_key("anthropic-beta"));

    // OpenAI formats route through the same registry, so they must be refused
    // before a request leaves the process.
    for format in [ApiFormat::ChatCompletions, ApiFormat::Responses] {
        let error = provider
            .execute(
                request(format, "claude-sonnet-4-6", false),
                context("claude-sonnet-4-6"),
            )
            .await
            .err()
            .unwrap();
        assert_eq!(error.status, StatusCode::BAD_REQUEST);
        assert!(receiver.try_recv().is_err());
    }
    assert_eq!(
        provider
            .convert_reasoning_effort("claude-sonnet-4-6", "high")
            .unwrap_err()
            .status,
        StatusCode::BAD_REQUEST
    );

    let error = anthropic_provider(&format!("{base}/rate/v1"), "upstream-secret")
        .execute(
            ApiRequest::parse(ApiFormat::Anthropic, sent).unwrap(),
            context("claude-sonnet-4-6"),
        )
        .await
        .err()
        .unwrap();
    assert_eq!(error.status, StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(error.kind, "rate_limit_error");
    assert!(!error.message.contains("upstream-secret"));
    assert!(error.message.contains("[redacted]"));
    task.abort();
}

#[tokio::test]
async fn anthropic_subscription_sends_bearer_beta_and_retries_only_one_401() {
    let messages = Arc::new(AtomicUsize::new(0));
    let refreshes = Arc::new(AtomicUsize::new(0));
    let (sender, mut receiver) = tokio::sync::mpsc::channel(4);
    let app = Router::new()
        .route(
            "/v1/oauth/token",
            post({
                let refreshes = refreshes.clone();
                move |Json(body): Json<Value>| {
                    let refreshes = refreshes.clone();
                    async move {
                        refreshes.fetch_add(1, Ordering::SeqCst);
                        assert_eq!(body["grant_type"], "refresh_token");
                        assert_eq!(body["refresh_token"], "refresh-old");
                        assert_eq!(body["client_id"], "9d1c250a-e61b-44d9-88ed-5944d1962f5e");
                        Json(json!({
                            "access_token":"access-new",
                            "refresh_token":"refresh-new",
                            "expires_in":3600
                        }))
                    }
                }
            }),
        )
        .route(
            "/v1/messages",
            post({
                let messages = messages.clone();
                move |headers: HeaderMap, Json(body): Json<Value>| {
                    let messages = messages.clone();
                    let sender = sender.clone();
                    async move {
                        let call = messages.fetch_add(1, Ordering::SeqCst);
                        sender.send(headers.clone()).await.unwrap();
                        if call == 0 {
                            assert_eq!(headers["authorization"], "Bearer access-old");
                            return (
                                StatusCode::UNAUTHORIZED,
                                Json(json!({"error":{"message":"expired"}})),
                            );
                        }
                        assert_eq!(headers["authorization"], "Bearer access-new");
                        if call == 2 {
                            return (
                                StatusCode::FORBIDDEN,
                                Json(json!({"error":{"message":"denied"}})),
                            );
                        }
                        let mut body = body;
                        body["type"] = json!("message");
                        (StatusCode::OK, Json(body))
                    }
                }
            }),
        );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let directory =
        std::env::temp_dir().join(format!("tinyllm-anthropic-native-{}", uuid::Uuid::new_v4()));
    anthropic::auth::save_fixture(
        &directory,
        "access-old",
        "refresh-old",
        anthropic::auth::fixture_expiry(),
    );
    let server = Server::default();
    let mut provider = anthropic::AnthropicProvider::new(
        serde_json::from_value(json!({
            "auth": {"type":"Subscription", "options":{"credentials_dir":directory.clone()}},
            "base_url": format!("{base}/v1"),
            "models": {"claude-sonnet-4-6": {}},
        }))
        .unwrap(),
        http::client(&server).unwrap(),
        server,
    )
    .unwrap();
    provider.set_token_url(format!("{base}/v1/oauth/token"));
    let request = json!({
        "model":"router/claude-sonnet-4-6",
        "max_tokens":32,
        "messages":[{"role":"user","content":"hello"}]
    });
    let mut subscriber_context = context("claude-sonnet-4-6");
    subscriber_context.headers.insert(
        "user-agent",
        "claude-cli/2.1.300 (external, cli)".parse().unwrap(),
    );
    let returned = provider
        .execute(
            ApiRequest::parse(ApiFormat::Anthropic, request.clone()).unwrap(),
            subscriber_context,
        )
        .await
        .unwrap();
    // The client's Claude Code identity reaches Anthropic: its user-agent is
    // forwarded and the billing header keys on that version.
    let headers = receiver.recv().await.unwrap();
    assert_eq!(headers["user-agent"], "claude-cli/2.1.300 (external, cli)");
    let ResponseBody::Json(returned) = returned.body else {
        panic!("expected JSON")
    };
    assert_eq!(
        returned["system"],
        json!([{
            "type": "text",
            "text": "x-anthropic-billing-header: cc_version=2.1.300.2d3; cc_entrypoint=sdk-cli; cch=2cf24;"
        }])
    );
    {
        let headers = receiver.recv().await.unwrap();
        assert!(!headers.contains_key("x-api-key"));
        assert_eq!(headers["anthropic-version"], "2023-06-01");
        let betas = headers
            .get_all("anthropic-beta")
            .iter()
            .flat_map(|value| value.to_str().unwrap().split(','))
            .map(str::trim)
            .collect::<Vec<_>>();
        assert!(betas.contains(&"test-beta"));
        assert_eq!(
            betas
                .iter()
                .filter(|beta| **beta == "oauth-2025-04-20")
                .count(),
            1
        );
        assert!(!headers.contains_key("x-app"));
        assert!(!headers.contains_key("x-stainless-helper-method"));
    }
    let error = match provider
        .execute(
            ApiRequest::parse(ApiFormat::Anthropic, request).unwrap(),
            context("claude-sonnet-4-6"),
        )
        .await
    {
        Ok(_) => panic!("403 unexpectedly succeeded"),
        Err(error) => error,
    };
    assert_eq!(error.status, StatusCode::BAD_GATEWAY);
    assert_eq!(messages.load(Ordering::SeqCst), 3);
    assert_eq!(refreshes.load(Ordering::SeqCst), 1);
    assert!(receiver.recv().await.is_some());
    task.abort();
    drop(provider);
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn anthropic_config_requires_a_tagged_api_key_and_option_free_models() {
    for (provider, valid) in [
        (
            json!({"type":"anthropic","auth":{"type":"ApiKey","options":"fixture-key"},
                "models":{"claude-sonnet-4-6":{}}}),
            true,
        ),
        (
            json!({"type":"anthropic","auth":{"type":"ApiKey","options":"fixture-key"},
                "base_url":"http://127.0.0.1:9/v1"}),
            true,
        ),
        (
            json!({"type":"anthropic","auth":{"type":"Subscription","options":{}}}),
            true,
        ),
        (
            json!({"type":"anthropic","auth":{"type":"Subscription"},
                "base_url":"https://example.com/v1"}),
            false,
        ),
        (json!({"type":"anthropic","api_key":"fixture-key"}), false),
        (
            json!({"type":"anthropic","auth":{"type":"ApiKey","options":""}}),
            false,
        ),
        (
            json!({"type":"anthropic","auth":{"type":"ApiKey","options":"has space"}}),
            false,
        ),
        (
            json!({"type":"anthropic","auth":{"type":"ApiKey","options":"fixture-key"},
                "base_url":"http://api.anthropic.com/v1"}),
            false,
        ),
        (
            json!({"type":"anthropic","auth":{"type":"ApiKey","options":"fixture-key"},
                "base_url":"https://api.anthropic.com/v1?beta=true"}),
            false,
        ),
        (
            json!({"type":"anthropic","auth":{"type":"ApiKey","options":"fixture-key"},
                "models":{"claude-sonnet-4-6":{"reasoning_effort":"high"}}}),
            false,
        ),
        (
            json!({"type":"anthropic","auth":{"type":"ApiKey","options":"fixture-key"},
                "user_agent":" trailing "}),
            false,
        ),
    ] {
        let config = serde_json::from_value::<crate::config::Config>(json!({
            "providers": {"fixture": provider.clone()}
        }));
        assert_eq!(
            config.is_ok_and(|config| config.validate().is_ok()),
            valid,
            "{provider}"
        );
    }
}

#[test]
fn anthropic_provider_loads_from_toml_and_yaml() {
    let directory =
        std::env::temp_dir().join(format!("tinyllm-anthropic-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&directory).unwrap();
    for (name, text) in [
        (
            "config.toml",
            "[server]\nstate_dir = \"state\"\n[providers.claude]\ntype = \"anthropic\"\n\
             [providers.claude.auth]\ntype = \"ApiKey\"\noptions = \"fixture-key\"\n\
             [providers.claude.models.\"claude-sonnet-4-6\"]\n",
        ),
        (
            "config.yaml",
            "server:\n  state_dir: state\nproviders:\n  claude:\n    type: anthropic\n\
             \n    auth:\n      type: ApiKey\n      options: fixture-key\n\
             \n    models:\n      claude-sonnet-4-6: {}\n",
        ),
    ] {
        let path = directory.join(name);
        std::fs::write(&path, text).unwrap();
        let config = crate::config::Config::load(&path).unwrap();
        let crate::config::ProviderConfig::Anthropic(provider) = &config.providers["claude"] else {
            panic!("wrong provider variant")
        };
        assert_eq!(provider.auth.api_key().unwrap(), "fixture-key");
        assert!(provider.models.contains_key("claude-sonnet-4-6"));
        assert!(provider.base_url.is_none());
    }

    let path = directory.join("subscription.toml");
    std::fs::write(
        &path,
        "[server]\nstate_dir = 'state'\n[providers.anthropic]\ntype = 'anthropic'\n\
         [providers.anthropic.auth]\ntype = 'Subscription'\n",
    )
    .unwrap();
    let config = crate::config::Config::load(&path).unwrap();
    let crate::config::ProviderConfig::Anthropic(provider) = &config.providers["anthropic"] else {
        panic!("wrong provider variant")
    };
    let anthropic::models::AnthropicAuth::Subscription(options) = &provider.auth else {
        panic!("wrong auth variant")
    };
    assert_eq!(
        options.credentials_dir,
        directory.join("state/auth/anthropic")
    );
    assert_ne!(options.credentials_dir, directory.join("state/auth"));

    std::fs::write(
        &path,
        "[server]\nstate_dir = 'state'\n[providers.claude]\ntype = 'anthropic'\n\
         [providers.claude.auth]\ntype = 'Subscription'\n",
    )
    .unwrap();
    let config = crate::config::Config::load(&path).unwrap();
    let crate::config::ProviderConfig::Anthropic(provider) = &config.providers["claude"] else {
        panic!("wrong provider variant")
    };
    let anthropic::models::AnthropicAuth::Subscription(options) = &provider.auth else {
        panic!("wrong auth variant")
    };
    assert_eq!(options.credentials_dir, directory.join("state/auth/claude"));

    std::fs::write(
        &path,
        "[server]\nstate_dir = 'state'\n[providers.claude]\ntype = 'anthropic'\n\
         [providers.claude.auth]\ntype = 'Subscription'\noptions = { credentials_dir = 'relative-auth' }\n",
    )
    .unwrap();
    let config = crate::config::Config::load(&path).unwrap();
    let crate::config::ProviderConfig::Anthropic(provider) = &config.providers["claude"] else {
        panic!("wrong provider variant")
    };
    let anthropic::models::AnthropicAuth::Subscription(options) = &provider.auth else {
        panic!("wrong auth variant")
    };
    assert_eq!(options.credentials_dir, directory.join("relative-auth"));

    std::fs::write(
        &path,
        "[server]\nstate_dir = 'state'\n[providers.claude]\ntype = 'anthropic'\n\
         [providers.claude.auth]\ntype = 'Subscription'\noptions = { credentials_dir = 'state/auth' }\n",
    )
    .unwrap();
    assert!(crate::config::Config::load(&path).is_err());

    std::fs::write(
        &path,
        "[server]\nstate_dir = 'state'\n[providers.claude]\ntype = 'anthropic'\n\
         [providers.claude.auth]\ntype = 'Subscription'\noptions = { credentials_dir = 'state/tmp/../auth' }\n",
    )
    .unwrap();
    let error = match crate::config::Config::load(&path) {
        Ok(_) => panic!("parent-directory alias unexpectedly loaded"),
        Err(error) => error.to_string(),
    };
    assert!(error.contains("must not contain '..'"));
    let _ = std::fs::remove_dir_all(directory);
}

#[test]
fn anthropic_credentials_reject_absolute_alias_of_relative_openai_root() {
    let relative_root =
        std::path::PathBuf::from(format!(".tinyllm-anthropic-path-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir(&relative_root).unwrap();
    let absolute_auth = std::env::current_dir()
        .unwrap()
        .join(&relative_root)
        .join("state/auth");
    let path = relative_root.join("config.toml");
    std::fs::write(
        &path,
        format!(
            "[server]\nstate_dir = 'state'\n[providers.claude]\ntype = 'anthropic'\n\
             [providers.claude.auth]\ntype = 'Subscription'\noptions = {{ credentials_dir = {} }}\n",
            serde_json::to_string(&absolute_auth.to_string_lossy()).unwrap()
        ),
    )
    .unwrap();

    let error = match crate::config::Config::load(&path) {
        Ok(_) => panic!("absolute alias of relative OpenAI root unexpectedly loaded"),
        Err(error) => error.to_string(),
    };
    assert!(error.contains("must not use OpenAI's default auth directory"));
    assert!(!absolute_auth.exists());
    std::fs::remove_dir_all(relative_root).unwrap();
}

#[cfg(unix)]
#[test]
fn anthropic_credentials_reject_existing_symlink_ancestor_alias() {
    use std::os::unix::fs::symlink;

    let root =
        std::env::temp_dir().join(format!("tinyllm-anthropic-alias-{}", uuid::Uuid::new_v4()));
    let real = root.join("real");
    let alias = root.join("alias");
    std::fs::create_dir_all(&real).unwrap();
    symlink(&real, &alias).unwrap();
    let path = alias.join("config.toml");
    let real_auth = real.join("state/auth");
    std::fs::write(
        &path,
        format!(
            "[server]\nstate_dir = 'state'\n[providers.claude]\ntype = 'anthropic'\n\
             [providers.claude.auth]\ntype = 'Subscription'\noptions = {{ credentials_dir = {} }}\n",
            serde_json::to_string(&real_auth.to_string_lossy()).unwrap()
        ),
    )
    .unwrap();

    let error = match crate::config::Config::load(&path) {
        Ok(_) => panic!("symlink ancestor alias of OpenAI root unexpectedly loaded"),
        Err(error) => error.to_string(),
    };
    assert!(error.contains("must not use OpenAI's default auth directory"));
    assert!(!real_auth.exists());
    std::fs::remove_dir_all(root).unwrap();
}
