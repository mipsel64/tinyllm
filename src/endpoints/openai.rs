use super::{Endpoint, endpoint};
use crate::{models::ApiFormat, server::AppState};
use axum::{
    Json, Router,
    extract::{Request, State},
    response::Response,
    routing::{get, post},
};
use serde_json::{Value, json};
use std::sync::Arc;

pub struct OpenAiEndpoint;
impl Endpoint for OpenAiEndpoint {
    fn router(&self) -> Router<Arc<AppState>> {
        Router::new()
            .route("/v1/chat/completions", post(chat))
            .route("/v1/responses", post(responses))
            .route("/v1/models", get(models))
    }
}
async fn chat(State(app): State<Arc<AppState>>, request: Request) -> Response {
    endpoint::execute(app, request, ApiFormat::ChatCompletions).await
}
async fn responses(State(app): State<Arc<AppState>>, request: Request) -> Response {
    endpoint::execute(app, request, ApiFormat::Responses).await
}
async fn models(State(app): State<Arc<AppState>>) -> Json<Value> {
    Json(
        json!({"object":"list","data":app.providers.models().iter().map(|m|json!({"id":m.id,"object":"model","created":0,"owned_by":m.id.split('/').next().unwrap_or("")})).collect::<Vec<_>>()}),
    )
}
