use crate::Result;
use axum::http::HeaderMap;
use futures::stream::BoxStream;
use serde_json::Value;

pub struct ModelInfo {
    pub id: String,
    pub display_name: String,
}

pub enum ApiEvent {
    Anthropic(Value),
    ChatCompletions(Value),
    Responses(Value),
    Done,
}

pub enum ResponseBody {
    Json(Value),
    Stream(BoxStream<'static, Result<ApiEvent>>),
}

pub struct ProviderOutput {
    pub headers: HeaderMap,
    pub body: ResponseBody,
}
