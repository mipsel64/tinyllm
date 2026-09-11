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

/// Ordered weakest to strongest so a configured ceiling can be compared.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
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

    pub fn parse(value: &str) -> Option<Self> {
        [
            Self::None,
            Self::Minimal,
            Self::Low,
            Self::Medium,
            Self::High,
            Self::XHigh,
            Self::Max,
        ]
        .into_iter()
        .find(|level| level.as_str() == value)
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
    pub max_concurrent_requests: Option<usize>,
    /// Routes Claude Code's permission-classifier subrequests to this
    /// provider/model. Unset leaves them on the session's model.
    pub auto_review_model: Option<String>,
    /// Maps a client model ID onto a configured provider/model, so a client
    /// that hardcodes an Anthropic name still routes somewhere.
    pub model_aliases: BTreeMap<String, String>,
    pub request_body_timeout_seconds: u64,
    pub timeout_seconds: u64,
    pub keep_alive_seconds: u64,
    /// Obsolete since continuations became client-carried. Accepted so existing
    /// configs still start; validation warns and ignores them.
    #[serde(rename = "max_state_bytes")]
    pub obsolete_max_state_bytes: Option<serde_json::Value>,
    #[serde(rename = "state_cleanup")]
    pub obsolete_state_cleanup: Option<serde_json::Value>,
}

impl Default for Server {
    fn default() -> Self {
        Self {
            bind: "127.0.0.1:8080".parse().unwrap(),
            auth_token: None,
            state_dir: default_state_dir(),
            max_request_bytes: 8 * 1024 * 1024,
            max_response_bytes: 32 * 1024 * 1024,
            max_concurrent_requests: None,
            auto_review_model: None,
            model_aliases: BTreeMap::new(),
            request_body_timeout_seconds: 30,
            timeout_seconds: 600,
            keep_alive_seconds: 10,
            obsolete_max_state_bytes: None,
            obsolete_state_cleanup: None,
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
