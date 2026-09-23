use super::Endpoint;
use crate::{config::ProviderConfig, server::AppState};
use axum::{Json, Router, extract::State, routing::get};
use serde_json::{Value, json};
use std::sync::Arc;

pub struct ApiEndpoint;
impl Endpoint for ApiEndpoint {
    fn router(&self) -> Router<Arc<AppState>> {
        Router::new().route("/api/v1/models", get(models))
    }
}

async fn models(State(app): State<Arc<AppState>>) -> Json<Value> {
    Json(json!({
        "object": "list",
        "configured_models": app.providers.models().iter().map(|model| json!({
            "id": model.id,
            "object": "model",
            "created": 0,
            "owned_by": model.id.split('/').next().unwrap_or(""),
        })).collect::<Vec<_>>(),
        "providers": app.config.providers.iter().map(|(id, provider)| provider_metadata(id, provider)).collect::<Vec<_>>(),
    }))
}

fn provider_metadata(id: &str, provider: &ProviderConfig) -> Value {
    let (provider_type, auth) = match provider {
        ProviderConfig::Anthropic(config) => (
            "anthropic",
            if config.auth.is_subscription() {
                "subscription"
            } else {
                "api_key"
            },
        ),
        ProviderConfig::OpenAi(config) => (
            "openai",
            if config.auth.is_subscription() {
                "subscription"
            } else {
                "api_key"
            },
        ),
        ProviderConfig::OpenRouter(_) => ("openrouter", "api_key"),
        ProviderConfig::Zai(_) => ("zai", "api_key"),
    };
    json!({"id": id, "type": provider_type, "auth": auth})
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_metadata_distinguishes_openai_auth_families() {
        let api_key: ProviderConfig = serde_json::from_value(json!({
            "type": "openai",
            "auth": {"type": "ApiKey", "options": "fixture-key"}
        }))
        .unwrap();
        let subscription: ProviderConfig = serde_json::from_value(json!({
            "type": "openai",
            "auth": {"type": "Subscription", "options": {}}
        }))
        .unwrap();

        assert_eq!(
            provider_metadata("openai-api", &api_key),
            json!({"id": "openai-api", "type": "openai", "auth": "api_key"})
        );
        assert_eq!(
            provider_metadata("codex", &subscription),
            json!({"id": "codex", "type": "openai", "auth": "subscription"})
        );
    }
}
