pub mod models;

pub use models::Config;

use crate::{
    Result,
    config::Server,
    error::Error,
    models::{ApiFormat, ApiRequest, ModelInfo, ProviderOutput, RequestContext},
    providers::{Provider, http, validate_effort},
};
use reqwest::{Client, Url};

pub struct OpenRouterProvider {
    config: Config,
    client: Client,
    messages: Url,
    chat: Url,
    responses: Url,
    limit: usize,
}

impl OpenRouterProvider {
    pub fn new(config: Config, client: Client, server: Server) -> eyre::Result<Self> {
        let base = config
            .base_url
            .as_deref()
            .unwrap_or("https://openrouter.ai/api/v1");
        Ok(Self {
            messages: http::endpoint(base, "/messages")?,
            chat: http::endpoint(base, "/chat/completions")?,
            responses: http::endpoint(base, "/responses")?,
            config,
            client,
            limit: server.max_response_bytes,
        })
    }
}

#[async_trait::async_trait]
impl Provider for OpenRouterProvider {
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

    fn convert_reasoning_effort(&self, _model: &str, effort: &str) -> Result<String> {
        validate_effort(effort)?;
        Ok(effort.into())
    }

    async fn execute(
        &self,
        mut request: ApiRequest,
        context: RequestContext,
    ) -> Result<ProviderOutput> {
        let body = &mut request.body;
        if let Some(effort) = self
            .config
            .models
            .get(&context.model)
            .and_then(|m| m.reasoning_effort)
        {
            let effort = effort.as_str();
            match request.format {
                ApiFormat::Anthropic => body.default_effort(Some("output_config"), effort),
                ApiFormat::ChatCompletions | ApiFormat::Responses => {
                    body.default_effort(Some("reasoning"), effort)
                }
            }
        }
        let effort = match request.format {
            ApiFormat::Anthropic => body
                .fields
                .get_mut("output_config")
                .and_then(|v| v.get_mut("effort")),
            ApiFormat::ChatCompletions if body.fields.contains_key("reasoning_effort") => {
                body.fields.get_mut("reasoning_effort")
            }
            ApiFormat::ChatCompletions | ApiFormat::Responses => body
                .fields
                .get_mut("reasoning")
                .and_then(|v| v.get_mut("effort")),
        };
        if let Some(effort) = effort.filter(|v| !v.is_null()) {
            let value = effort
                .as_str()
                .ok_or_else(|| Error::invalid("reasoning effort must be a string"))?;
            *effort = serde_json::json!(self.convert_reasoning_effort(&context.model, value)?);
        }
        let url = match request.format {
            ApiFormat::Anthropic => &self.messages,
            ApiFormat::ChatCompletions => &self.chat,
            ApiFormat::Responses => &self.responses,
        };
        http::forward(
            &self.client,
            url.clone(),
            &self.config.api_key,
            request,
            context,
            self.limit,
        )
        .await
    }
}
