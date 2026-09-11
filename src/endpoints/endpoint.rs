use crate::{
    error::Error,
    models::{ApiEvent, ApiFormat, ApiRequest, RequestContext, ResponseBody},
    server::AppState,
};
use axum::{
    Json, Router,
    body::Body,
    extract::{FromRequest, Request},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use bytes::Bytes;
use futures::StreamExt;
use serde_json::{Value, json};
use std::{convert::Infallible, sync::Arc, time::Duration};
use tracing::Instrument;

pub trait Endpoint: Send + Sync {
    fn router(&self) -> Router<Arc<AppState>>;
}

pub fn error_response(error: Error, format: ApiFormat) -> Response {
    tracing::warn!(
        error=?error,
        "request failed"
    );
    let value = error_value(&error, format);
    (error.status, *error.headers, Json(value)).into_response()
}
fn error_value(error: &Error, format: ApiFormat) -> Value {
    if format == ApiFormat::Anthropic {
        error.json()
    } else {
        json!({"error":{"message":error.message,"type":error.kind,"code":error.kind,"param":null}})
    }
}

pub async fn execute(app: Arc<AppState>, request: Request, format: ApiFormat) -> Response {
    match run(app, request, format).await {
        Ok(response) => response,
        Err(error) => error_response(error, format),
    }
}

async fn run(app: Arc<AppState>, request: Request, format: ApiFormat) -> crate::Result<Response> {
    let permit = app
        .permits
        .as_ref()
        .map(|permits| permits.clone().try_acquire_owned())
        .transpose()
        .map_err(|_| Error {
            status: StatusCode::SERVICE_UNAVAILABLE,
            kind: "overloaded_error",
            message: "max_concurrent_requests reached".into(),
            headers: Box::default(),
        })?;
    let query = request.uri().query().map(str::to_owned);
    let request_id = request
        .extensions()
        .get::<uuid::Uuid>()
        .copied()
        .unwrap_or_else(uuid::Uuid::new_v4)
        .to_string();
    let mut headers = request.headers().clone();
    headers.remove("authorization");
    headers.remove("x-api-key");
    let Json(value) = tokio::time::timeout(
        Duration::from_secs(app.config.server.request_body_timeout_seconds),
        Json::<Value>::from_request(request, &app),
    )
    .await
    .map_err(|_| Error {
        status: StatusCode::REQUEST_TIMEOUT,
        kind: "timeout_error",
        message: "request body timed out".into(),
        headers: Box::default(),
    })?
    .map_err(|e| Error {
        status: e.status(),
        kind: if e.status() == StatusCode::PAYLOAD_TOO_LARGE {
            "request_too_large"
        } else {
            "invalid_request_error"
        },
        message: "invalid JSON request or max_request_bytes exceeded".into(),
        headers: Box::default(),
    })?;
    let mut request = ApiRequest::parse(format, value)?;
    // Clients that hardcode an Anthropic model ID reach a configured target
    // instead of an unknown-model error.
    if let Some(target) = app.config.server.model_aliases.get(&request.body.model) {
        tracing::debug!(alias = %request.body.model, target, "resolved model alias");
        request.body.model = target.clone();
    }
    if format == ApiFormat::Anthropic
        && let Some(reviewer) = app.config.server.auto_review_model.as_deref()
        && request.body.is_auto_review()
    {
        tracing::debug!(
            requested = %request.body.model,
            reviewer,
            "routing permission classifier to the auto-review model"
        );
        request.body.model = reviewer.to_owned();
    }
    let (provider, model) = app.providers.resolve(&request.body.model)?;
    let context = RequestContext {
        request_id,
        provider: request
            .body
            .model
            .split_once('/')
            .expect("validated model")
            .0
            .into(),
        model,
        public_model: request.body.model.clone(),
        headers,
        query,
    };
    tracing::debug!(request_id=%context.request_id,provider=%context.provider,model=%context.model,"dispatch request");
    let result = provider.execute(request, context).await?;
    let response = match result.body {
        ResponseBody::Json(value) => (result.headers, Json(value)).into_response(),
        ResponseBody::Stream(mut upstream) => {
            let span = tracing::Span::current();
            let duration = Duration::from_secs(app.config.server.keep_alive_seconds);
            let body = async_stream::stream! {
                let _permit=permit;
                let mut keep_alive=tokio::time::interval(duration);
                keep_alive.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
                keep_alive.tick().await;
                let mut sequence=0u64;
                loop {
                    let item=tokio::select! {
                        item=upstream.next().instrument(span.clone())=>item,
                        _=keep_alive.tick()=> {
                            yield Ok::<_,Infallible>(Bytes::from_static(if format==ApiFormat::Anthropic {b"event: ping\ndata: {\"type\":\"ping\"}\n\n"} else {b": keep-alive\n\n"}));
                            continue;
                        }
                    };
                    match item {
                        Some(Ok(event))=> {
                            if let ApiEvent::Responses(value)=&event { sequence=value["sequence_number"].as_u64().unwrap_or(sequence).saturating_add(1); }
                            yield Ok(serialize(event));
                        }
                        Some(Err(error))=> {
                            span.in_scope(|| tracing::warn!(error_type=error.kind,"upstream stream failed"));
                            let value=error_value(&error,format);
                            let event=match format {
                                ApiFormat::Anthropic=>ApiEvent::Anthropic(value),
                                ApiFormat::ChatCompletions=>ApiEvent::ChatCompletions(value),
                                ApiFormat::Responses=>ApiEvent::Responses(json!({"type":"error","code":error.kind,"message":error.message,"param":null,"sequence_number":sequence})),
                            };
                            yield Ok(serialize(event));break;
                        }
                        None=>break,
                    }
                }
            };
            (
                result.headers,
                [
                    ("content-type", "text/event-stream"),
                    ("cache-control", "no-cache"),
                    ("x-accel-buffering", "no"),
                ],
                Body::from_stream(body),
            )
                .into_response()
        }
    };
    Ok(response)
}

fn serialize(event: ApiEvent) -> Bytes {
    match event {
        ApiEvent::Anthropic(value) | ApiEvent::Responses(value) => Bytes::from(format!(
            "event: {}\ndata: {value}\n\n",
            value["type"].as_str().unwrap_or("error")
        )),
        ApiEvent::ChatCompletions(value) => Bytes::from(format!("data: {value}\n\n")),
        ApiEvent::Done => Bytes::from_static(b"data: [DONE]\n\n"),
    }
}
