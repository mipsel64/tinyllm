pub mod auth;
mod compatible;
pub mod models;
pub mod protocol;
pub mod state;
mod stop;
pub mod stream;

use self::{
    auth::Auth,
    models::{Config, Model},
    state::Store,
};
use crate::{
    Result,
    config::Server,
    error::Error,
    models::{
        ApiEvent, ApiFormat, ApiRequest, ModelInfo, ProviderOutput, RequestContext, ResponseBody,
        anthropic::StreamEvent,
    },
    providers::{Provider, http, validate_effort},
};
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use futures::StreamExt;
use serde_json::Value;
use std::sync::Arc;

pub struct OpenAiProvider {
    config: Config,
    server: Server,
    store: Arc<Store>,
    client: reqwest::Client,
    auth: Auth,
}

impl OpenAiProvider {
    pub fn new(
        config: Config,
        client: reqwest::Client,
        server: Server,
        store: Arc<Store>,
    ) -> eyre::Result<Self> {
        let auth = Auth::open(&config.auth)?;
        Ok(Self {
            config,
            server,
            store,
            client,
            auth,
        })
    }

    fn model(&self, native: &str) -> Model {
        Model {
            id: native.into(),
            reasoning_effort: self
                .config
                .models
                .get(native)
                .and_then(|m| m.reasoning_effort),
        }
    }

    async fn send(&self, body: &mut Value) -> Result<(reqwest::Response, String)> {
        if body.get("service_tier").is_none()
            && let Some(tier) = body["model"]
                .as_str()
                .and_then(|model| self.config.models.get(model))
                .and_then(|options| options.service_tier)
        {
            body["service_tier"] = serde_json::json!(tier);
        }
        if body["service_tier"] == "fast" {
            body["service_tier"] = serde_json::json!("priority");
        }
        let (mut key, mut account) = self
            .auth
            .access(None)
            .await
            .map_err(|e| Error::upstream(e.to_string()))?;
        let mut response = self
            .send_authenticated(body, &key, account.as_deref())
            .await?;
        if self.config.auth.is_subscription() && response.status() == StatusCode::UNAUTHORIZED {
            drop(response);
            (key, account) = self
                .auth
                .access(Some(&key))
                .await
                .map_err(|e| Error::upstream(e.to_string()))?;
            response = self
                .send_authenticated(body, &key, account.as_deref())
                .await?;
        }
        if !response.status().is_success() {
            let status = response.status();
            let headers = response.headers().clone();
            let bytes = http::read_bounded(response, self.server.max_response_bytes).await?;
            let body = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
            let mut error = http::upstream_error(status, &body, &key);
            http::copy_headers(&headers, &mut error.headers);
            return Err(error);
        }
        Ok((response, key))
    }

    async fn send_authenticated(
        &self,
        body: &Value,
        key: &str,
        account: Option<&str>,
    ) -> Result<reqwest::Response> {
        let mut authorization = HeaderValue::from_str(&format!("Bearer {key}"))
            .map_err(|_| Error::upstream("invalid upstream access token"))?;
        authorization.set_sensitive(true);
        let mut request = self
            .client
            .post(format!(
                "{}/responses",
                self.config.base_url().trim_end_matches('/')
            ))
            .header("authorization", authorization)
            .json(body);
        if let Some(org) = &self.config.organization {
            request = request.header("openai-organization", org);
        }
        if let Some(project) = &self.config.project {
            request = request.header("openai-project", project);
        }
        if let Some(account) = account {
            let mut account = HeaderValue::from_str(account)
                .map_err(|_| Error::upstream("invalid ChatGPT account ID"))?;
            account.set_sensitive(true);
            request = request
                .header("chatgpt-account-id", account)
                .header("originator", "tinyllm")
                .header("accept", "text/event-stream");
        }
        request.send().await.map_err(|e| {
            Error::upstream(if e.is_timeout() {
                "OpenAI request timed out"
            } else {
                "cannot connect to OpenAI"
            })
        })
    }

    async fn anthropic(&self, req: Value, context: RequestContext) -> Result<ProviderOutput> {
        let model = self.model(&context.model);
        let mut state_pins = Vec::new();
        let restored = self
            .store
            .restore_scoped_pinned(&req, &context.provider, &model.id, &mut state_pins)
            .await?;
        let continuation = if restored.is_empty() {
            "fresh"
        } else {
            "restored"
        };
        let subscription = self.config.auth.is_subscription();
        let streaming = req["stream"] == true;
        let mut stops = stop::StopFilter::new(&req["stop_sequences"])?;
        let defaults = state::tool_defaults(&req["tools"]);
        let mut upstream_request = protocol::request(&req, &model, &restored)?;
        let search = upstream_request["tools"]
            .as_array()
            .is_some_and(|tools| tools.iter().any(|tool| tool["type"] == "web_search"));
        let search_limit = upstream_request.get("max_tool_calls").is_some();
        let constrained_output =
            stops.enabled() || upstream_request["text"]["format"]["type"] == "json_schema";
        if subscription {
            upstream_request = protocol::subscription_request(&req, upstream_request)?;
        }
        if let Some(effort) = upstream_request["reasoning"]["effort"].as_str() {
            upstream_request["reasoning"]["effort"] =
                serde_json::json!(self.convert_reasoning_effort(&model.id, effort)?);
        }
        let (upstream, key) = self.send(&mut upstream_request).await?;
        let mut headers = HeaderMap::new();
        http::copy_headers(upstream.headers(), &mut headers);
        headers.insert(
            "x-tinyllm-continuation",
            HeaderValue::from_static(continuation),
        );
        if search {
            headers.insert(
                "x-tinyllm-web-search",
                HeaderValue::from_static(match (subscription, search_limit) {
                    (true, true) => "native; citations=markdown; max-uses=best-effort",
                    (false, true) => "native; citations=markdown; max-uses=upstream",
                    (_, false) => "native; citations=markdown; max-uses=unset",
                }),
            );
        }
        if stops.enabled() {
            headers.insert(
                "x-tinyllm-stop-sequences",
                HeaderValue::from_static("local; usage-includes-discarded-output"),
            );
        }
        headers.insert("x-tinyllm-compatibility",HeaderValue::from_static(if subscription {
            "anthropic-cache-hints-ignored; openai-automatic-caching; reasoning-reference-v1; model-switch-carriers-v1; subscription-max-tokens-unenforced"
        } else { "anthropic-cache-hints-ignored; openai-automatic-caching; reasoning-reference-v1; model-switch-carriers-v1" }));
        if upstream_request["stream"] == true
            && !upstream
                .headers()
                .get("content-type")
                .map(|h| h.to_str().is_ok_and(|h| h.starts_with("text/event-stream")))
                .unwrap_or(subscription)
        {
            return Err(Error::upstream("OpenAI did not return text/event-stream"));
        }
        let reference = Store::reference();
        state_pins.extend(self.store.pin(&reference).await?);
        let alias = context.public_model;
        let mut translator = stream::Translator::new(alias.clone(), reference.clone());
        translator.sparse_completion = subscription;
        let body = if streaming {
            let decoded = stream::decode(upstream.bytes_stream(), self.server.max_response_bytes);
            let store = self.store.clone();
            let events = async_stream::try_stream! {
                futures::pin_mut!(decoded);
                loop {
                    let event = decoded.next().await.ok_or_else(|| Error::upstream("upstream stream ended before completion"))??;
                    let events = translator.accept(&event).map_err(|mut e| {e.message=e.message.replace(&key,"[redacted]");e})?;
                    if constrained_output && let Some(native) = &translator.completed {
                        reject_citation_controls(native)?;
                    }
                    for event in events.into_iter().flat_map(|event| stops.push(event)) {
                        yield anthropic_event(event, &reference)?;
                    }
                    if let Some(mut native) = translator.completed.take() {
                        let mut response = protocol::response(&native,&alias,&reference).map_err(|mut e| {e.message=e.message.replace(&key,"[redacted]");e})?;
                        for event in stops.finish() {
                            yield anthropic_event(event, &reference)?;
                        }
                        stops.apply(&mut native, &mut response)?;
                        store.save_scoped(&reference,&context.provider,&model.id,&native,serde_json::to_value(&response.content).map_err(|_| Error::upstream("cannot encode response content"))?,&defaults).await?;
                        tracing::info!(output_tokens=response.usage.output_tokens,"stream completed");
                        for event in stream::finish(&response) { yield anthropic_event(event, &reference)?; }
                        break;
                    }
                }
            };
            ResponseBody::Stream(Box::pin(events))
        } else {
            let mut native: Value = if subscription {
                let decoded =
                    stream::decode(upstream.bytes_stream(), self.server.max_response_bytes);
                futures::pin_mut!(decoded);
                while let Some(event) = decoded.next().await {
                    event.and_then(|e| translator.accept(&e)).map_err(|mut e| {
                        e.message = e.message.replace(&key, "[redacted]");
                        e
                    })?;
                    if translator.completed.is_some() {
                        break;
                    }
                }
                translator
                    .completed
                    .ok_or_else(|| Error::upstream("upstream stream ended before completion"))?
            } else {
                let data = http::read_bounded(upstream, self.server.max_response_bytes).await?;
                serde_json::from_slice(&data)
                    .map_err(|_| Error::upstream("OpenAI returned invalid JSON"))?
            };
            let mut response =
                protocol::response(&native, &alias, &reference).map_err(|mut e| {
                    e.message = e.message.replace(&key, "[redacted]");
                    e
                })?;
            if constrained_output {
                reject_citation_controls(&native)?;
            }
            stops.json(&response);
            stops.apply(&mut native, &mut response)?;
            self.store
                .save_scoped(
                    &reference,
                    &context.provider,
                    &model.id,
                    &native,
                    serde_json::to_value(&response.content)
                        .map_err(|_| Error::upstream("cannot encode response content"))?,
                    &defaults,
                )
                .await?;
            let mut response = serde_json::to_value(response)
                .map_err(|_| Error::upstream("cannot encode response"))?;
            for block in response["content"].as_array_mut().unwrap() {
                state::mark_block(block, &reference)?;
            }
            ResponseBody::Json(response)
        };
        Ok(ProviderOutput {
            headers,
            body,
            state_pins,
        })
    }
}

fn anthropic_event(event: StreamEvent, reference: &str) -> Result<ApiEvent> {
    let mut event =
        serde_json::to_value(event).map_err(|_| Error::upstream("cannot encode stream event"))?;
    if let Some(block) = event.get_mut("content_block") {
        state::mark_block(block, reference)?;
    }
    Ok(ApiEvent::Anthropic(event))
}

fn reject_citation_controls(native: &Value) -> Result<()> {
    if !protocol::citation_sources(native)?.is_empty() {
        return Err(Error::upstream(
            "cited output with stop_sequences or structured output is unsupported",
        ));
    }
    Ok(())
}

#[async_trait::async_trait]
impl Provider for OpenAiProvider {
    fn convert_reasoning_effort(&self, model: &str, effort: &str) -> Result<String> {
        validate_effort(effort)?;
        let family = model
            .strip_prefix("gpt-")
            .and_then(|name| name.split('-').next());
        match (family, effort) {
            (Some("5.2" | "5.3" | "5.4" | "5.5"), "max") => Ok("xhigh".into()),
            (Some("6"), "none") if model == "gpt-6-astra" || model.starts_with("gpt-6-astra-") => {
                Err(Error::invalid(
                    "GPT-6 Astra requires reasoning; use low or higher",
                ))
            }
            _ => Ok(effort.into()),
        }
    }

    fn models(&self) -> Vec<ModelInfo> {
        self.config
            .models
            .keys()
            .map(|id| ModelInfo {
                id: id.clone(),
                display_name: id.clone(),
            })
            .collect()
    }

    async fn execute(
        &self,
        request: ApiRequest,
        context: RequestContext,
    ) -> Result<ProviderOutput> {
        if request.format == ApiFormat::Anthropic {
            self.anthropic(request.into_value(), context).await
        } else {
            compatible::execute(self, request, context).await
        }
    }
}
