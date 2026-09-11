use crate::{Result, error::Error};
use axum::http::HeaderMap;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ApiFormat {
    Anthropic,
    ChatCompletions,
    Responses,
}

#[derive(Clone, Deserialize, Serialize)]
pub struct RequestBody {
    pub model: String,
    #[serde(default)]
    pub stream: bool,
    #[serde(flatten)]
    pub fields: Map<String, Value>,
}

/// Claude Code checks each tool call for risky actions and prompt injection by
/// sending this subrequest to the session's model.
const AUTO_REVIEW_SYSTEM_PREFIX: &str =
    "You are a security monitor for autonomous AI coding agents.";

impl RequestBody {
    /// Recognises that permission-classifier subrequest: non-streaming, no
    /// tools, and the monitor system prompt. It wants a short verdict, so a
    /// small model answers it faster and more predictably than a reasoning one.
    pub fn is_auto_review(&self) -> bool {
        if self.stream {
            return false;
        }
        if self
            .fields
            .get("tools")
            .and_then(Value::as_array)
            .is_some_and(|tools| !tools.is_empty())
        {
            return false;
        }
        match self.fields.get("system") {
            Some(Value::String(text)) => text.starts_with(AUTO_REVIEW_SYSTEM_PREFIX),
            Some(Value::Array(blocks)) => blocks.iter().any(|block| {
                block["text"]
                    .as_str()
                    .is_some_and(|text| text.starts_with(AUTO_REVIEW_SYSTEM_PREFIX))
            }),
            _ => false,
        }
    }

    pub fn default_effort(&mut self, container: Option<&str>, effort: &str) {
        let fields = &mut self.fields;
        if fields.contains_key("reasoning_effort")
            || fields.get("reasoning").is_some_and(|v| {
                !v.is_object()
                    || ["effort", "max_tokens", "enabled"]
                        .iter()
                        .any(|key| v.get(key).is_some())
            })
            || fields.get("thinking").is_some_and(|v| {
                !v.is_object()
                    || v["type"] == "disabled"
                    || v.get("budget_tokens").is_some()
                    || (effort == "none" && v.get("type").is_some())
            })
            || fields
                .get("output_config")
                .is_some_and(|v| !v.is_object() || v.get("effort").is_some())
        {
            return;
        }
        if let Some(container) = container {
            if let Some(options) = fields
                .entry(container)
                .or_insert_with(|| Value::Object(Map::new()))
                .as_object_mut()
            {
                options.insert("effort".into(), serde_json::json!(effort));
            }
        } else {
            fields.insert("reasoning_effort".into(), serde_json::json!(effort));
        }
    }
}

pub struct ApiRequest {
    pub format: ApiFormat,
    pub body: RequestBody,
}

impl ApiRequest {
    pub fn parse(format: ApiFormat, value: Value) -> Result<Self> {
        let body: RequestBody = serde_json::from_value(value).map_err(|_| {
            Error::invalid("request must contain a string model and boolean stream")
        })?;
        Ok(Self { format, body })
    }

    pub fn into_value(self) -> Value {
        let body = self.body;
        let mut fields = body.fields;
        fields.insert("model".into(), Value::String(body.model));
        fields.insert("stream".into(), Value::Bool(body.stream));
        Value::Object(fields)
    }
}

pub struct RequestContext {
    pub request_id: String,
    pub provider: String,
    pub model: String,
    pub public_model: String,
    pub headers: HeaderMap,
    pub query: Option<String>,
}
