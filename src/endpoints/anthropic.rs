use super::{Endpoint, endpoint};
use crate::{
    error::Error,
    models::{
        ApiFormat,
        anthropic::{ModelInfo, ModelsListResponse},
    },
    server::AppState,
};
use axum::{
    Json, Router,
    extract::{Request, State, rejection::JsonRejection},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde_json::Value;
use std::sync::Arc;

/// Claude Code warms its connection pool against this before the first real
/// call, without credentials, so it is served unauthenticated.
pub const HELLO_PATH: &str = "/anthropic/api/hello";

pub struct AnthropicEndpoint;
impl Endpoint for AnthropicEndpoint {
    fn router(&self) -> Router<Arc<AppState>> {
        Router::new()
            .route(HELLO_PATH, get(hello))
            .route("/anthropic/v1/messages", post(messages))
            .route("/anthropic/v1/messages/count_tokens", post(count_tokens))
            .route("/anthropic/v1/models", get(models))
    }
}
async fn hello() -> Json<serde_json::Value> {
    Json(serde_json::json!({"ok": true}))
}
async fn messages(State(app): State<Arc<AppState>>, request: Request) -> Response {
    endpoint::execute(app, request, ApiFormat::Anthropic).await
}
/// Answered locally: Claude Code decides when to compact from this, and a
/// round trip per estimate would cost more than the estimate is worth.
async fn count_tokens(body: Result<Json<Value>, JsonRejection>) -> Response {
    let rejected = |message: &'static str| {
        let error = Error::invalid(message);
        tracing::warn!(error = ?error, "request failed");
        (error.status, Json(error.json())).into_response()
    };
    let Ok(Json(value)) = body else {
        return rejected("count_tokens requires a JSON request body");
    };
    let Ok(body) = serde_json::from_value::<crate::models::request::RequestBody>(value) else {
        return rejected("count_tokens requires a string model and message list");
    };
    let input_tokens = super::count_tokens::count(&body);
    tracing::debug!(input_tokens, model = %body.model, "estimated input tokens");
    Json(serde_json::json!({ "input_tokens": input_tokens })).into_response()
}
async fn models(State(app): State<Arc<AppState>>) -> Json<ModelsListResponse> {
    let data: Vec<_> = app
        .providers
        .models()
        .into_iter()
        .map(|m| ModelInfo {
            id: m.id,
            display_name: m.display_name,
            model_type: "model".into(),
            created_at: "1970-01-01T00:00:00Z".into(),
        })
        .collect();
    Json(ModelsListResponse {
        first_id: data.first().map(|m| m.id.clone()),
        last_id: data.last().map(|m| m.id.clone()),
        has_more: false,
        data,
    })
}
