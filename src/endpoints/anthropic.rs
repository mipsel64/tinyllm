use super::{Endpoint, endpoint};
use crate::{
    models::{
        ApiFormat,
        anthropic::{ModelInfo, ModelsListResponse},
    },
    server::AppState,
};
use axum::{
    Json, Router,
    extract::{Request, State},
    response::Response,
    routing::{get, post},
};
use std::sync::Arc;

pub struct AnthropicEndpoint;
impl Endpoint for AnthropicEndpoint {
    fn router(&self) -> Router<Arc<AppState>> {
        Router::new()
            .route("/anthropic/v1/messages", post(messages))
            .route("/anthropic/v1/models", get(models))
    }
}
async fn messages(State(app): State<Arc<AppState>>, request: Request) -> Response {
    endpoint::execute(app, request, ApiFormat::Anthropic).await
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
