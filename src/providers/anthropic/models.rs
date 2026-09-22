use eyre::{Result, bail};
use serde::Deserialize;
use std::{collections::BTreeMap, path::PathBuf};

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub auth: AnthropicAuth,
    pub user_agent: Option<String>,
    #[serde(default)]
    pub base_url: Option<String>,
    #[serde(default)]
    pub models: BTreeMap<String, Model>,
}

#[derive(Clone, Deserialize)]
#[serde(tag = "type", content = "options", deny_unknown_fields)]
pub enum AnthropicAuth {
    ApiKey(String),
    Subscription(SubscriptionOptions),
}

#[derive(Clone, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SubscriptionOptions {
    pub credentials_dir: PathBuf,
}

/// Discovery only: native controls travel with the request, so there is nothing to set.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Model {}

impl AnthropicAuth {
    pub fn is_subscription(&self) -> bool {
        matches!(self, Self::Subscription(_))
    }

    pub fn api_key(&self) -> Result<&str> {
        match self {
            Self::ApiKey(key)
                if !key.is_empty() && key.bytes().all(|byte| byte.is_ascii_graphic()) =>
            {
                Ok(key)
            }
            Self::ApiKey(_) => bail!(
                "anthropic.auth.options must be a nonempty API key without whitespace or control characters"
            ),
            Self::Subscription(_) => {
                bail!("subscription authentication requires tinyllm anthropic login")
            }
        }
    }
}
