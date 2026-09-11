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
    extract::{Request, State},
    response::{IntoResponse, Response},
    routing::{get, post},
};
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
async fn count_tokens(State(app): State<Arc<AppState>>, request: Request) -> Response {
    let fail = |error: Error| {
        tracing::warn!(error = ?error, "request failed");
        (error.status, Json(error.json())).into_response()
    };
    // Shares the deadline and size limit with inference: this route accepts a
    // whole conversation, so it must not be the one that stalls.
    let value = match endpoint::read_body(&app, request).await {
        Ok(value) => value,
        Err(error) => return fail(error),
    };
    let Ok(body) = serde_json::from_value::<crate::models::request::RequestBody>(value) else {
        return fail(Error::invalid(
            "count_tokens requires a string model and message list",
        ));
    };
    // A count derived from input we could not read would be trusted as a context
    // size, so refuse rather than answer with a number that means nothing.
    if body
        .fields
        .get("messages")
        .is_some_and(|messages| !messages.is_array())
    {
        return fail(Error::invalid("count_tokens messages must be an array"));
    }
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
