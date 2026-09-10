use axum::http::{HeaderMap, StatusCode};
use serde_json::{Value, json};

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub struct Error {
    pub status: StatusCode,
    pub kind: &'static str,
    pub message: String,
    pub headers: Box<HeaderMap>,
}

impl Error {
    pub fn invalid(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            kind: "invalid_request_error",
            message: message.into(),
            headers: Box::default(),
        }
    }

    pub fn upstream(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_GATEWAY,
            kind: "api_error",
            message: message.into(),
            headers: Box::default(),
        }
    }

    pub fn openai(value: &Value) -> Self {
        let error = value
            .get("error")
            .filter(|v| v.is_object())
            .unwrap_or(value);
        let message = error["message"]
            .as_str()
            .unwrap_or("OpenAI response failed");
        let code = error["code"].as_str().unwrap_or("api_error");
        let mut result = Self::upstream(format!(
            "{}: {}",
            code.chars().take(128).collect::<String>(),
            message.chars().take(8192).collect::<String>()
        ));
        if code == "context_length_exceeded" {
            result.message = format!("capability_rejected: prompt_too_long; {}", result.message);
        }
        if code == "rate_limit_exceeded" {
            result.kind = "rate_limit_error";
        }
        result
    }

    pub fn json(&self) -> Value {
        json!({"type":"error","error":{"type":self.kind,"message":self.message}})
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}
impl std::error::Error for Error {}
