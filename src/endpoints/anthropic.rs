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
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use std::sync::Arc;

pub struct AnthropicEndpoint;
impl Endpoint for AnthropicEndpoint {
    fn router(&self) -> Router<Arc<AppState>> {
        Router::new()
            .route("/anthropic/v1/messages", post(messages))
            .route(
                "/anthropic/v1/messages/count_tokens",
                post(count_tokens_unavailable),
            )
            .route("/anthropic/v1/models", get(models))
    }
}
async fn messages(State(app): State<Arc<AppState>>, request: Request) -> Response {
    endpoint::execute(app, request, ApiFormat::Anthropic).await
}
async fn count_tokens_unavailable() -> Response {
    let error = Error {
        status: StatusCode::NOT_FOUND,
        kind: "not_found_error",
        message: "token counting is not supported; use client-side context estimation".into(),
        headers: Box::default(),
    };
    tracing::debug!(error = ?error, "optional token counting is unavailable");
    (error.status, Json(error.json())).into_response()
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
