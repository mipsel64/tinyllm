use crate::{
    Result,
    config::Server,
    error::Error,
    models::{ApiEvent, ApiFormat, ApiRequest, ProviderOutput, RequestContext, ResponseBody},
};
use axum::http::{HeaderMap, StatusCode};
use bytes::Bytes;
use eventsource_stream::Eventsource;
use futures::{Stream, StreamExt, stream::BoxStream};
use reqwest::{Client, Response, Url};
use serde_json::Value;
use std::{collections::BTreeMap, time::Duration};

pub fn client(server: &Server) -> eyre::Result<Client> {
    Ok(Client::builder()
        .user_agent(concat!("tinyllm/", env!("CARGO_PKG_VERSION")))
        .redirect(reqwest::redirect::Policy::none())
        .connect_timeout(Duration::from_secs(10))
        .timeout(Duration::from_secs(server.timeout_seconds))
        .build()?)
}

pub async fn read_bounded(response: Response, limit: usize) -> Result<Vec<u8>> {
    let mut body = Vec::new();
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|_| Error::upstream("cannot read upstream response"))?;
        if chunk.len() > limit.saturating_sub(body.len()) {
            return Err(Error::upstream(
                "upstream response exceeds max_response_bytes",
            ));
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

pub fn copy_headers(source: &HeaderMap, target: &mut HeaderMap) {
    for name in ["retry-after", "x-request-id", "request-id"] {
        for value in source.get_all(name) {
            target.append(name, value.clone());
        }
    }
}

pub fn upstream_error(status: StatusCode, value: &Value, key: &str) -> Error {
    let detail = value
        .get("error")
        .filter(|v| v.is_object())
        .unwrap_or(value);
    let message = detail["message"]
        .as_str()
        .or_else(|| value["detail"].as_str())
        .unwrap_or("upstream request failed");
    let message = if key.is_empty() {
        message.to_owned()
    } else {
        message.replace(key, "[redacted]")
    };
    let mut error = Error::upstream(message.chars().take(8192).collect::<String>());
    if detail["code"] == "context_length_exceeded" {
        error.message = format!("capability_rejected: prompt_too_long; {}", error.message);
    }
    error.status = if status.is_client_error() && !matches!(status.as_u16(), 401 | 403) {
        status
    } else {
        StatusCode::BAD_GATEWAY
    };
    error.kind = match status.as_u16() {
        400 | 404 | 422 => "invalid_request_error",
        429 => "rate_limit_error",
        503 | 529 => "overloaded_error",
        _ => "api_error",
    };
    if matches!(detail["type"].as_str(), Some("overloaded_error")) {
        error.kind = "overloaded_error";
    }
    if matches!(detail["type"].as_str(), Some("rate_limit_error"))
        || detail["code"] == "rate_limit_exceeded"
    {
        error.kind = "rate_limit_error";
    }
    error
}

pub(crate) fn endpoint(base: &str, path: &str) -> eyre::Result<Url> {
    Url::parse(&format!("{}{path}", base.trim_end_matches('/')))
        .map_err(|_| eyre::eyre!("cannot parse provider endpoint"))
}

pub(crate) async fn forward(
    client: &Client,
    mut url: Url,
    key: &str,
    request: ApiRequest,
    context: RequestContext,
    limit: usize,
) -> Result<ProviderOutput> {
    let format = request.format;
    let streaming = request.body.stream;
    let mut value = request.into_value();
    if contains_reference(&value) {
        return Err(Error::invalid(
            "tinyllm reasoning references cannot be sent to another provider",
        ));
    }
    if format == ApiFormat::Responses && value["background"] == true {
        return Err(Error::invalid(
            "background Responses require lifecycle endpoints that tinyllm does not expose",
        ));
    }
    value["model"] = Value::String(context.model);
    url.set_query(context.query.as_deref());
    let mut builder = client.post(url).bearer_auth(key).json(&value);
    if format == ApiFormat::Anthropic {
        for name in ["anthropic-version", "anthropic-beta"] {
            for value in context.headers.get_all(name) {
                builder = builder.header(name, value);
            }
        }
        if !context.headers.contains_key("anthropic-version") {
            builder = builder.header("anthropic-version", "2023-06-01");
        }
    }
    let response = builder.send().await.map_err(|error| {
        Error::upstream(if error.is_timeout() {
            "upstream request timed out"
        } else {
            "cannot connect to upstream"
        })
    })?;
    let status = response.status();
    let mut headers = HeaderMap::new();
    copy_headers(response.headers(), &mut headers);
    if !status.is_success() {
        let bytes = read_bounded(response, limit).await.map_err(|mut error| {
            error.headers = Box::new(headers.clone());
            error
        })?;
        let value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
        let mut error = upstream_error(status, &value, key);
        error.headers = Box::new(headers);
        return Err(error);
    }
    let body = if streaming {
        if !response
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .is_some_and(|v| {
                v.split(';')
                    .next()
                    .is_some_and(|v| v.trim().eq_ignore_ascii_case("text/event-stream"))
            })
        {
            let mut error = Error::upstream("upstream did not return an SSE content type");
            error.headers = Box::new(headers);
            return Err(error);
        }
        ResponseBody::Stream(decode_native(
            response.bytes_stream(),
            format,
            context.public_model,
            key.to_owned(),
            limit,
        ))
    } else {
        let with_headers = |mut error: Error| {
            error.headers = Box::new(headers.clone());
            error
        };
        let bytes = read_bounded(response, limit).await.map_err(with_headers)?;
        let mut value: Value = serde_json::from_slice(&bytes)
            .map_err(|_| with_headers(Error::upstream("invalid upstream JSON")))?;
        if !value.is_object() {
            return Err(with_headers(Error::upstream(
                "upstream response must be a JSON object",
            )));
        }
        if value.get("error").is_some_and(|error| !error.is_null()) || value["status"] == "failed" {
            let mut error = upstream_error(StatusCode::BAD_GATEWAY, &value, key);
            error.headers = Box::new(headers);
            return Err(error);
        }
        if format == ApiFormat::Responses
            && !matches!(value["status"].as_str(), Some("completed" | "incomplete"))
        {
            return Err(with_headers(Error::upstream(
                "upstream response did not complete",
            )));
        }
        public_model(&mut value, &context.public_model);
        ResponseBody::Json(value)
    };
    Ok(ProviderOutput { headers, body })
}

fn contains_reference(value: &Value) -> bool {
    match value {
        Value::Array(values) => values.iter().any(contains_reference),
        Value::Object(values) => {
            (value["type"] == "redacted_thinking"
                && value["data"]
                    .as_str()
                    .is_some_and(|s| s.starts_with("tinyllm:v1:")))
                || value["type"] == "tinyllm_continuation"
                || values.values().any(contains_reference)
        }
        _ => false,
    }
}

fn public_model(value: &mut Value, model: &str) {
    if let Some(field) = value.get_mut("model") {
        *field = Value::String(model.into());
    }
    if value["type"] == "message_start"
        && let Some(field) = value.get_mut("message").and_then(|v| v.get_mut("model"))
    {
        *field = Value::String(model.into());
    }
    if let Some(field) = value.get_mut("response").and_then(|v| v.get_mut("model")) {
        *field = Value::String(model.into());
    }
}

pub(crate) fn decode_native<S, E>(
    source: S,
    format: ApiFormat,
    model: String,
    key: String,
    limit: usize,
) -> BoxStream<'static, Result<ApiEvent>>
where
    S: Stream<Item = std::result::Result<Bytes, E>> + Send + 'static,
    E: Send + 'static,
{
    let bounded = async_stream::try_stream! {
        futures::pin_mut!(source);
        let mut remaining = limit;
        while let Some(chunk) = source.next().await {
            let chunk = chunk.map_err(|_| std::io::Error::other("upstream connection failed"))?;
            remaining = remaining.checked_sub(chunk.len())
                .ok_or_else(|| std::io::Error::other("max_response_bytes exceeded"))?;
            yield chunk;
        }
    };
    let bounded: BoxStream<'static, std::io::Result<Bytes>> = Box::pin(bounded);
    Box::pin(async_stream::try_stream! {
        let events = bounded.eventsource();
        futures::pin_mut!(events);
        let mut complete = false;
        let mut choices = BTreeMap::<u64,bool>::new();
        while let Some(event) = events.next().await {
            let event = event.map_err(|_| Error::upstream("upstream SSE failed, was malformed, or exceeded max_response_bytes"))?;
            if event.data.is_empty() { continue; }
            if event.data == "[DONE]" {
                if format != ApiFormat::ChatCompletions || choices.is_empty() || choices.values().any(|done| !done) {
                    Err(Error::upstream("upstream stream ended before protocol completion"))?;
                }
                complete = true;
                yield ApiEvent::Done;
                break;
            }
            let mut value: Value = serde_json::from_str(&event.data)
                .map_err(|_| Error::upstream("invalid upstream SSE JSON"))?;
            if !value.is_object() { Err(Error::upstream("upstream SSE event must be a JSON object"))?; }
            if !event.event.is_empty() && event.event != "message" && value["type"] != event.event {
                Err(Error::upstream("upstream SSE event name does not match its type"))?;
            }
            if value["type"] == "error" || value.get("error").is_some_and(|v| !v.is_null()) {
                Err(upstream_error(StatusCode::BAD_GATEWAY, &value, &key))?;
            }
            if value["type"] == "response.failed" || value["response"]["status"] == "failed" || value["response"].get("error").is_some_and(|v| !v.is_null()) {
                Err(upstream_error(StatusCode::BAD_GATEWAY, &value["response"], &key))?;
            }
            complete = match format {
                ApiFormat::Anthropic => value["type"] == "message_stop",
                ApiFormat::ChatCompletions => false,
                ApiFormat::Responses => matches!(value["type"].as_str(), Some("response.completed" | "response.incomplete")),
            };
            if format == ApiFormat::ChatCompletions && let Some(items) = value["choices"].as_array() {
                for choice in items {
                    let index = choice["index"].as_u64().ok_or_else(|| Error::upstream("upstream chat choice has no valid index"))?;
                    if choice["finish_reason"] == "error" { Err(Error::upstream("upstream chat generation failed"))?; }
                    let done = choices.entry(index).or_default();
                    *done |= choice["finish_reason"].as_str().is_some_and(|reason| !reason.is_empty());
                }
            }
            if format == ApiFormat::Responses && complete
                && !matches!(value["response"]["status"].as_str(),Some("completed"|"incomplete")) {
                Err(Error::upstream("upstream response did not complete"))?;
            }
            public_model(&mut value, &model);
            yield match format {
                ApiFormat::Anthropic => ApiEvent::Anthropic(value),
                ApiFormat::ChatCompletions => ApiEvent::ChatCompletions(value),
                ApiFormat::Responses => ApiEvent::Responses(value),
            };
            if complete { break; }
        }
        if !complete { Err(Error::upstream("upstream stream ended before protocol completion"))?; }
    })
}
