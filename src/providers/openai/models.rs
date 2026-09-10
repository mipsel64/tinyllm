use crate::models::ReasoningEffort;
use eyre::{Result, bail};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, path::PathBuf};

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    #[serde(default)]
    pub base_url: String,
    pub auth: OpenAiAuth,
    pub organization: Option<String>,
    pub project: Option<String>,
    #[serde(default)]
    pub models: BTreeMap<String, ModelOptions>,
}

#[derive(Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelOptions {
    pub reasoning_effort: Option<ReasoningEffort>,
    pub service_tier: Option<ServiceTier>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ServiceTier {
    Auto,
    Default,
    Flex,
    Priority,
    Fast,
    UltraFast,
}

impl Config {
    pub fn base_url(&self) -> &str {
        if !self.base_url.is_empty() {
            &self.base_url
        } else if self.auth.is_subscription() {
            "https://chatgpt.com/backend-api/codex"
        } else {
            "https://api.openai.com/v1"
        }
    }
}

#[derive(Clone, Deserialize)]
#[serde(tag = "type", content = "options", deny_unknown_fields)]
pub enum OpenAiAuth {
    ApiKey(String),
    Subscription(SubscriptionOptions),
}

#[derive(Clone, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SubscriptionOptions {
    pub credentials_dir: PathBuf,
}

impl OpenAiAuth {
    pub fn is_subscription(&self) -> bool {
        matches!(self, Self::Subscription(_))
    }

    pub fn api_key(&self) -> Result<&str> {
        match self {
            Self::ApiKey(key) => {
                if key.is_empty() || !key.bytes().all(|byte| byte.is_ascii_graphic()) {
                    bail!(
                        "openai.auth.options must be a nonempty API key without whitespace or control characters"
                    );
                }
                Ok(key)
            }
            Self::Subscription(_) => {
                bail!("subscription authentication requires tinyllm openai login")
            }
        }
    }
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Model {
    #[serde(default)]
    pub id: String,
    pub reasoning_effort: Option<ReasoningEffort>,
}
