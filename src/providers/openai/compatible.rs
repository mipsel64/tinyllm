use super::{OpenAiProvider, reasoning};
use crate::{
    Result,
    error::Error,
    models::{ApiEvent, ApiFormat, ApiRequest, ProviderOutput, RequestContext, ResponseBody},
    providers::{Provider, http},
};
use axum::http::{HeaderMap, HeaderValue};
use futures::StreamExt;
use serde_json::{Value, json};
use std::collections::HashSet;

#[path = "compatible_request.rs"]
mod request;
#[path = "compatible_stream.rs"]
mod stream;
#[cfg(test)]
#[path = "compatible_tests.rs"]
mod tests;

pub async fn execute(
    provider: &OpenAiProvider,
    api: ApiRequest,
    context: RequestContext,
) -> Result<ProviderOutput> {
    let format = api.format;
    if format == ApiFormat::Anthropic {
        return Err(Error::invalid("expected an OpenAI request"));
    }
    if context
        .query
        .as_deref()
        .is_some_and(|query| !query.is_empty())
    {
        return Err(Error::invalid(
            "query parameters are unsupported by the OpenAI Responses backend",
        ));
    }
    let streaming = api.body.stream;
    let source = api.into_value();
    let model = provider.model(&context.model);
    let subscription = provider.config.auth.is_subscription();
    let is_chat = format == ApiFormat::ChatCompletions;
    let include_usage = source["stream_options"]["include_usage"] == true;
    let continuation = if is_chat {
        carrier_status(&source)
    } else {
        "fresh"
    };
    let mut body = if is_chat {
        request::chat(&source, &model, subscription)?
    } else {
        request::native(source, &model, subscription)?
    };
    if let Some(effort) = body
        .get("reasoning")
        .and_then(|v| v.get("effort"))
        .filter(|v| !v.is_null())
    {
        let effort = effort
            .as_str()
            .ok_or_else(|| Error::invalid("reasoning effort must be a string"))?;
        body["reasoning"]["effort"] =
            serde_json::json!(provider.convert_reasoning_effort(&model.id, effort)?);
    }
    let (upstream, key) = provider.send(&mut body).await?;
    let mut headers = HeaderMap::new();
    http::copy_headers(upstream.headers(), &mut headers);
    if is_chat {
        headers.insert(
            "x-tinyllm-continuation",
            HeaderValue::from_static(continuation),
        );
        headers.insert(
            "x-tinyllm-compatibility",
            HeaderValue::from_static(if subscription {
                "chat-reasoning-details-carrier-v1; subscription-max-tokens-unenforced"
            } else {
                "chat-reasoning-details-carrier-v1"
            }),
        );
    }
    let upstream_streams = upstream
        .headers()
        .get("content-type")
        .map(|v| v.to_str().is_ok_and(|v| v.starts_with("text/event-stream")))
        .unwrap_or(subscription);
    if body["stream"] == true && !upstream_streams {
        return Err(Error::upstream("OpenAI did not return text/event-stream"));
    }
    let limit = provider.server.max_response_bytes;
    let public_model = context.public_model;
    let mut tracker = stream::Native::new(subscription);
    let body = if streaming {
        let decoded = super::stream::decode(upstream.bytes_stream(), limit);
        let mut chat = stream::Chat::new(public_model.clone());
        let events = async_stream::try_stream! {
            futures::pin_mut!(decoded);
            while let Some(event) = decoded.next().await {
                let mut event = tracker.accept(event?)?;
                if is_chat {
                    for chunk in chat.accept(&event)? { yield ApiEvent::ChatCompletions(chunk); }
                } else {
                    if let Some(model) = event.get_mut("response").and_then(|v| v.get_mut("model")) { *model = json!(public_model); }
                    yield ApiEvent::Responses(event);
                }
                if tracker.is_complete() { break; }
            }
            let native = tracker.finish()?;
            if is_chat {
                let response = chat_response(&native, &public_model)?;
                for chunk in chat.finish(&response, include_usage)? { yield ApiEvent::ChatCompletions(chunk); }
                yield ApiEvent::Done;
            }
        };
        let events =
            events.map(move |event: Result<ApiEvent>| event.map_err(|error| redact(error, &key)));
        ResponseBody::Stream(Box::pin(events))
    } else {
        let native = if upstream_streams {
            let decoded = super::stream::decode(upstream.bytes_stream(), limit);
            futures::pin_mut!(decoded);
            while let Some(event) = decoded.next().await {
                tracker
                    .accept(event?)
                    .map_err(|error| redact(error, &key))?;
                if tracker.is_complete() {
                    break;
                }
            }
            tracker.finish()?
        } else {
            let bytes = http::read_bounded(upstream, limit).await?;
            let value = serde_json::from_slice(&bytes)
                .map_err(|_| Error::upstream("OpenAI returned invalid JSON"))?;
            validate_response(&value).map_err(|error| redact(error, &key))?;
            value
        };
        if is_chat {
            let response =
                chat_response(&native, &public_model).map_err(|error| redact(error, &key))?;
            ResponseBody::Json(response)
        } else {
            let mut native = native;
            native["model"] = json!(public_model);
            ResponseBody::Json(native)
        }
    };
    Ok(ProviderOutput { headers, body })
}

/// Reports whether the replayed Chat history carried resumable OpenAI reasoning.
fn carrier_status(source: &Value) -> &'static str {
    let carried = source["messages"]
        .as_array()
        .into_iter()
        .flatten()
        .flat_map(|message| {
            message["reasoning_details"]
                .as_array()
                .into_iter()
                .flatten()
        })
        .any(|detail| {
            detail["data"]
                .as_str()
                .is_some_and(|data| reasoning::decode(data).is_some())
        });
    if carried { "restored" } else { "fresh" }
}

fn redact(mut error: Error, key: &str) -> Error {
    if !key.is_empty() {
        error.message = error.message.replace(key, "[redacted]");
    }
    error
}

fn created_at(response: &Value) -> Result<Value> {
    match response.get("created_at").filter(|value| !value.is_null()) {
        Some(value) if value.is_u64() => Ok(value.clone()),
        Some(_) => Err(Error::upstream("invalid upstream creation timestamp")),
        None => Ok(json!(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(|_| Error::upstream("invalid gateway clock"))?
                .as_secs()
        )),
    }
}

fn validate_response(response: &Value) -> Result<()> {
    if response["status"] == "failed" || response.get("error").is_some_and(|error| !error.is_null())
    {
        return Err(Error::openai(response));
    }
    if !matches!(
        response["status"].as_str(),
        Some("completed" | "incomplete")
    ) {
        return Err(Error::upstream("upstream response did not complete"));
    }
    if response["id"].as_str().is_none_or(str::is_empty) || !response["output"].is_array() {
        return Err(Error::upstream("upstream response ID or output is missing"));
    }
    if response["status"] == "incomplete"
        && response["incomplete_details"]["reason"].as_str().is_none()
    {
        return Err(Error::upstream(
            "upstream incomplete response omitted its reason",
        ));
    }
    Ok(())
}

fn chat_response(response: &Value, model: &str) -> Result<Value> {
    validate_response(response)?;
    let mut carriers = Vec::new();
    let mut content = String::new();
    let mut refusal = String::new();
    let mut tools = Vec::new();
    let mut calls = HashSet::new();
    let mut annotations = Vec::new();
    for item in response["output"].as_array().unwrap() {
        match item["type"].as_str() {
            Some("reasoning") => {
                if let Some(data) = reasoning::capture(item)
                    .as_ref()
                    .and_then(reasoning::encode)
                {
                    carriers.push(json!({"type":"tinyllm_continuation","data":data}));
                }
            }
            Some("message") => {
                if item["role"] != "assistant" {
                    return Err(Error::upstream("upstream message must have assistant role"));
                }
                for part in item["content"]
                    .as_array()
                    .ok_or_else(|| Error::upstream("upstream message content is missing"))?
                {
                    match part["type"].as_str() {
                        Some("output_text") => {
                            let offset = content.chars().count();
                            let text = part["text"].as_str().ok_or_else(|| {
                                Error::upstream("upstream output text is missing")
                            })?;
                            content.push_str(text);
                            if let Some(items) = part.get("annotations").filter(|v| !v.is_null()) {
                                for annotation in items.as_array().ok_or_else(|| {
                                    Error::upstream("invalid upstream annotations")
                                })? {
                                    if annotation["type"] != "url_citation" {
                                        return Err(Error::upstream(
                                            "unsupported Chat annotation; use native Responses",
                                        ));
                                    }
                                    let mut citation = annotation.clone();
                                    citation.as_object_mut().unwrap().remove("type");
                                    for key in ["start_index", "end_index"] {
                                        citation[key] = json!(
                                            citation[key]
                                                .as_u64()
                                                .and_then(|n| n.checked_add(offset as u64))
                                                .ok_or_else(|| Error::upstream(
                                                    "invalid citation index"
                                                ))?
                                        );
                                    }
                                    annotations.push(
                                        json!({"type":"url_citation","url_citation":citation}),
                                    );
                                }
                            }
                        }
                        Some("refusal") => refusal.push_str(
                            part["refusal"]
                                .as_str()
                                .ok_or_else(|| Error::upstream("upstream refusal is missing"))?,
                        ),
                        _ => {
                            return Err(Error::upstream(
                                "unsupported Chat output content; use native Responses",
                            ));
                        }
                    }
                }
            }
            Some("function_call") => {
                let id = item["call_id"]
                    .as_str()
                    .filter(|id| !id.is_empty())
                    .ok_or_else(|| Error::upstream("tool call ID is missing"))?;
                if !calls.insert(id) {
                    return Err(Error::upstream("duplicate upstream tool call ID"));
                }
                let name = item["name"]
                    .as_str()
                    .ok_or_else(|| Error::upstream("tool name is missing"))?;
                let arguments = item["arguments"]
                    .as_str()
                    .ok_or_else(|| Error::upstream("tool arguments are missing"))?;
                if !serde_json::from_str::<Value>(arguments).is_ok_and(|v| v.is_object()) {
                    return Err(Error::upstream(
                        "upstream returned incomplete or invalid tool JSON; tool was not executed",
                    ));
                }
                tools.push(json!({"id":id,"type":"function","function":{"name":name,"arguments":arguments}}));
            }
            _ => {
                return Err(Error::upstream(
                    "unsupported Chat output item; use native Responses for hosted tools",
                ));
            }
        }
    }
    let reason = if response["status"] == "incomplete" {
        match response["incomplete_details"]["reason"].as_str() {
            Some("max_output_tokens") => "length",
            Some("content_filter") => "content_filter",
            _ => return Err(Error::upstream("unsupported incomplete response reason")),
        }
    } else if tools.is_empty() {
        "stop"
    } else {
        "tool_calls"
    };
    let mut message = json!({"role":"assistant","content":if content.is_empty() { Value::Null } else {json!(content)},"refusal":if refusal.is_empty() {Value::Null} else {json!(refusal)},"reasoning_details":carriers});
    if !tools.is_empty() {
        message["tool_calls"] = json!(tools);
    }
    if !annotations.is_empty() {
        message["annotations"] = json!(annotations);
    }
    let native_usage = &response["usage"];
    let input = native_usage["input_tokens"]
        .as_u64()
        .ok_or_else(|| Error::upstream("upstream omitted input usage"))?;
    let output = native_usage["output_tokens"]
        .as_u64()
        .ok_or_else(|| Error::upstream("upstream omitted output usage"))?;
    for (details, field, maximum) in [
        ("input_tokens_details", "cached_tokens", input),
        ("output_tokens_details", "reasoning_tokens", output),
    ] {
        if let Some(value) = native_usage[details]
            .get(field)
            .filter(|value| !value.is_null())
            && value.as_u64().is_none_or(|value| value > maximum)
        {
            return Err(Error::upstream("invalid upstream usage details"));
        }
    }
    let total = input
        .checked_add(output)
        .ok_or_else(|| Error::upstream("invalid upstream usage"))?;
    if native_usage["total_tokens"]
        .as_u64()
        .is_some_and(|value| value != total)
    {
        return Err(Error::upstream(
            "upstream total usage does not match input and output",
        ));
    }
    let mut usage = json!({"prompt_tokens":input,"completion_tokens":output,"total_tokens":total});
    if let Some(details) = native_usage
        .get("input_tokens_details")
        .filter(|v| !v.is_null())
    {
        usage["prompt_tokens_details"] = details.clone();
    }
    if let Some(details) = native_usage
        .get("output_tokens_details")
        .filter(|v| !v.is_null())
    {
        usage["completion_tokens_details"] = details.clone();
    }
    let mut result = json!({"id":response["id"],"object":"chat.completion","created":created_at(response)?,"model":model,"choices":[{"index":0,"message":message,"finish_reason":reason,"logprobs":null}],"usage":usage});
    for key in ["service_tier", "system_fingerprint"] {
        if let Some(value) = response.get(key) {
            result[key] = value.clone();
        }
    }
    Ok(result)
}
