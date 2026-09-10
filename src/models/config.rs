use crate::providers::{openai, openrouter, zai};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, net::SocketAddr, path::PathBuf};

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    #[serde(default)]
    pub server: Server,
    #[serde(default)]
    pub logging: crate::logging::Config,
    pub providers: BTreeMap<String, ProviderConfig>,
}

#[derive(Clone, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ProviderConfig {
    OpenAi(openai::models::Config),
    OpenRouter(openrouter::models::Config),
    Zai(zai::models::Config),
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    None,
    Minimal,
    Low,
    Medium,
    High,
    XHigh,
    Max,
}

impl ReasoningEffort {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Minimal => "minimal",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::XHigh => "xhigh",
            Self::Max => "max",
        }
    }
}

#[derive(Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Server {
    pub bind: SocketAddr,
    pub auth_token: Option<String>,
    pub state_dir: PathBuf,
    pub max_request_bytes: usize,
    pub max_response_bytes: usize,
    pub max_state_bytes: u64,
    pub max_concurrent_requests: Option<usize>,
    pub request_body_timeout_seconds: u64,
    pub timeout_seconds: u64,
    pub keep_alive_seconds: u64,
}

impl Default for Server {
    fn default() -> Self {
        Self {
            bind: "127.0.0.1:8080".parse().unwrap(),
            auth_token: None,
            state_dir: default_state_dir(),
            max_request_bytes: 8 * 1024 * 1024,
            max_response_bytes: 32 * 1024 * 1024,
            max_state_bytes: 256 * 1024 * 1024,
            max_concurrent_requests: None,
            request_body_timeout_seconds: 30,
            timeout_seconds: 600,
            keep_alive_seconds: 10,
        }
    }
}

fn default_state_dir() -> PathBuf {
    std::env::var_os("XDG_STATE_HOME")
        .map(PathBuf::from)
        .filter(|path| path.is_absolute())
        .or_else(|| {
            std::env::var_os("HOME")
                .map(PathBuf::from)
                .filter(|path| path.is_absolute())
                .map(|home| home.join(".local/state"))
        })
        .map(|root| root.join("tinyllm"))
        .unwrap_or_default()
}
