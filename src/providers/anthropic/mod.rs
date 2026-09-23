pub mod auth;
pub mod billing;
pub mod models;

pub use models::Config;

use self::auth::Auth;
use crate::{
    Result,
    config::Server,
    error::Error,
    models::{ApiFormat, ApiRequest, ModelInfo, ProviderOutput, RequestContext},
    providers::{Provider, http, http::Auth as HttpAuth},
};
use axum::http::StatusCode;
use reqwest::{Client, Url};

pub struct AnthropicProvider {
    config: Config,
    auth: Auth,
    client: Client,
    messages: Url,
    limit: usize,
}

impl AnthropicProvider {
    pub fn new(config: Config, client: Client, server: Server) -> eyre::Result<Self> {
        let base = config
            .base_url
            .as_deref()
            .unwrap_or("https://api.anthropic.com/v1");
        let auth = Auth::open(&config.auth)?;
        Ok(Self {
            messages: http::endpoint(base, "/messages")?,
            auth,
            config,
            client,
            limit: server.max_response_bytes,
        })
    }

    #[cfg(test)]
    pub(crate) fn set_token_url(&mut self, token_url: String) {
        self.auth.set_token_url(token_url);
    }
}

#[async_trait::async_trait]
impl Provider for AnthropicProvider {
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

    fn convert_reasoning_effort(&self, _model: &str, _effort: &str) -> Result<String> {
        Err(Error::invalid(
            "anthropic models take native thinking controls; reasoning_effort is not converted",
        ))
    }

    async fn execute(
        &self,
        request: ApiRequest,
        context: RequestContext,
    ) -> Result<ProviderOutput> {
        if request.format != ApiFormat::Anthropic {
            return Err(Error::invalid(
                "anthropic providers serve the Anthropic Messages API only; use /anthropic/v1/messages",
            ));
        }
        let subscription = self.auth.is_subscription();
        // Subscription requests carry the client's user-agent and billing
        // header so Anthropic bills them as the client's Claude Code instead
        // of rejecting them as third-party usage.
        let client_agent = context
            .headers
            .get("user-agent")
            .and_then(|value| value.to_str().ok())
            .filter(|_| subscription)
            .map(str::to_owned);
        let user_agent = self
            .config
            .user_agent
            .as_deref()
            .or(client_agent.as_deref());
        let mut prepared = http::prepare(self.messages.clone(), request, context, user_agent)?;
        if subscription {
            let version = self
                .config
                .claude_code_version
                .as_deref()
                .or_else(|| billing::claude_cli_version(user_agent))
                .unwrap_or(billing::FALLBACK_CLAUDE_CODE_VERSION);
            billing::prepend(prepared.payload_mut(), version);
        }
        let mut key = self
            .auth
            .access(None)
            .await
            .map_err(|error| Error::upstream(error.to_string()))?;
        let http_auth = if subscription {
            HttpAuth::Bearer(&key)
        } else {
            HttpAuth::ApiKey(&key)
        };
        let mut response = http::send(
            &self.client,
            &prepared,
            http_auth,
            subscription.then_some("oauth-2025-04-20"),
        )
        .await?;
        if subscription && response.status() == StatusCode::UNAUTHORIZED {
            drop(response);
            key = self
                .auth
                .access(Some(&key))
                .await
                .map_err(|error| Error::upstream(error.to_string()))?;
            response = http::send(
                &self.client,
                &prepared,
                HttpAuth::Bearer(&key),
                Some("oauth-2025-04-20"),
            )
            .await?;
        }
        http::receive(response, prepared, &key, self.limit).await
    }
}
