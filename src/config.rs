pub use crate::models::config::{Config, ProviderConfig, Server};
use crate::providers::openai::models::OpenAiAuth;
use eyre::{Result, WrapErr, bail};
use std::path::{Path, PathBuf};

impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        let format = match path.extension().and_then(|e| e.to_str()) {
            Some("toml") => ::config::FileFormat::Toml,
            Some("yaml" | "yml") => ::config::FileFormat::Yaml,
            _ => bail!("config file must have a .toml, .yaml or .yml extension"),
        };
        let text = std::fs::read_to_string(path).wrap_err_with(|| "cannot read config file")?;
        let rendered = shellexpand::env(&text).map_err(|_| {
            eyre::eyre!("cannot expand config: referenced environment variable is unset or invalid")
        })?;
        let mut value: serde_json::Value = ::config::Config::builder()
            .add_source(::config::File::from_str(&rendered, format))
            .build()
            .and_then(|c| c.try_deserialize())
            .map_err(|_| eyre::eyre!("invalid config syntax; compare tinyllm.example.toml"))?;
        if let Some(providers) = value["providers"].as_object_mut() {
            for provider in providers.values_mut() {
                if provider["auth"]["type"] == "Subscription"
                    && provider["auth"].get("options").is_none()
                {
                    provider["auth"]["options"] = serde_json::json!({});
                }
            }
        }
        let mut config: Self = serde_json::from_value(value).map_err(|_| eyre::eyre!("invalid config value or unknown field; use providers.<name> configuration in tinyllm.example.toml"))?;
        if config.server.state_dir.as_os_str().is_empty() {
            bail!("set server.state_dir or provide HOME/XDG_STATE_HOME for its default");
        }
        resolve_path(&mut config.server.state_dir, path);
        for (prefix, provider) in &mut config.providers {
            if let ProviderConfig::OpenAi(openai) = provider
                && let OpenAiAuth::Subscription(options) = &mut openai.auth
            {
                if options.credentials_dir.as_os_str().is_empty() {
                    options.credentials_dir = config.server.state_dir.join("auth");
                    if prefix != "openai" {
                        options.credentials_dir.push(prefix);
                    }
                } else {
                    resolve_path(&mut options.credentials_dir, path);
                }
            }
        }
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        if self.providers.is_empty() {
            bail!("configure at least one provider");
        }
        if self
            .server
            .auth_token
            .as_deref()
            .is_some_and(|s| !valid_key(s))
        {
            bail!(
                "server.auth_token must be nonempty without whitespace or control characters; omit it to disable authentication"
            );
        }
        for (prefix, provider) in &self.providers {
            if matches!(prefix.as_str(), "." | "..")
                || prefix.is_empty()
                || prefix.len() > 64
                || !prefix
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'.'))
            {
                bail!(
                    "provider names must contain 1–64 ASCII letters, digits, dots, underscores or hyphens"
                );
            }
            match provider {
                ProviderConfig::OpenAi(c) => {
                    let local = validate_url(c.base_url())?;
                    if c.auth.is_subscription() {
                        if !local
                            && c.base_url().trim_end_matches('/')
                                != "https://chatgpt.com/backend-api/codex"
                        {
                            bail!(
                                "subscription auth requires the ChatGPT Codex backend (loopback allowed for fixtures)"
                            );
                        }
                        if c.organization.is_some() || c.project.is_some() {
                            bail!("organization and project apply to API keys only");
                        }
                    } else {
                        c.auth.api_key()?;
                    }
                    for id in c.models.keys() {
                        validate_model(id)?;
                    }
                }
                ProviderConfig::OpenRouter(c) => {
                    if !valid_key(&c.api_key) {
                        bail!(
                            "provider api_key must be nonempty without whitespace or control characters"
                        );
                    }
                    if let Some(url) = &c.base_url {
                        validate_url(url)?;
                    }
                    for id in c.models.keys() {
                        validate_model(id)?;
                    }
                }
                ProviderConfig::Zai(c) => {
                    if !valid_key(&c.api_key) {
                        bail!(
                            "provider api_key must be nonempty without whitespace or control characters"
                        );
                    }
                    if let Some(url) = &c.base_url {
                        validate_url(url)?;
                    }
                    for id in c.models.keys() {
                        validate_model(id)?;
                    }
                }
            }
        }
        let s = &self.server;
        // Both are resolved per request, so a bad value must fail at startup
        // rather than on every affected request.
        let routable = |setting: &str, target: &str| -> Result<()> {
            let (prefix, model) = target
                .split_once('/')
                .ok_or_else(|| eyre::eyre!("{setting} must be provider/native-model-id"))?;
            validate_model(model)?;
            if !self.providers.contains_key(prefix) {
                bail!("{setting} names provider {prefix:?}, which is not configured");
            }
            Ok(())
        };
        if let Some(reviewer) = &s.auto_review_model {
            routable("auto_review_model", reviewer)?;
        }
        for (alias, target) in &s.model_aliases {
            if alias.is_empty() {
                bail!("model_aliases keys must not be empty");
            }
            routable("model_aliases", target)?;
        }
        for name in [
            s.obsolete_max_state_bytes
                .as_ref()
                .map(|_| "max_state_bytes"),
            s.obsolete_state_cleanup.as_ref().map(|_| "state_cleanup"),
        ]
        .into_iter()
        .flatten()
        {
            tracing::warn!(
                setting = name,
                "obsolete server setting ignored; tinyllm no longer stores continuations, so it can be deleted"
            );
        }
        if s.max_concurrent_requests
            .is_some_and(|limit| !(1..=tokio::sync::Semaphore::MAX_PERMITS).contains(&limit))
        {
            bail!(
                "max_concurrent_requests must be between 1 and {} when configured",
                tokio::sync::Semaphore::MAX_PERMITS
            );
        }
        if s.max_request_bytes == 0
            || s.max_response_bytes == 0
            || s.request_body_timeout_seconds == 0
            || s.timeout_seconds == 0
            || s.keep_alive_seconds == 0
        {
            bail!("limits and timeouts must be positive");
        }
        Ok(())
    }
}

fn resolve_path(value: &mut PathBuf, config: &Path) {
    if value.is_relative() {
        *value = config.parent().unwrap_or(Path::new(".")).join(&value);
    }
}
fn valid_key(value: &str) -> bool {
    !value.is_empty() && value.bytes().all(|b| b.is_ascii_graphic())
}
pub(crate) fn validate_model(model: &str) -> Result<()> {
    if model.is_empty()
        || model.len() > 256
        || model.split('/').any(str::is_empty)
        || !model.bytes().all(|b| b.is_ascii_graphic())
    {
        bail!("invalid native model ID");
    }
    Ok(())
}
fn validate_url(value: &str) -> Result<bool> {
    let url = reqwest::Url::parse(value).map_err(|_| eyre::eyre!("invalid provider base_url"))?;
    let local = url.host_str().is_some_and(|h| {
        h == "localhost"
            || h.trim_matches(['[', ']'])
                .parse::<std::net::IpAddr>()
                .is_ok_and(|ip| ip.is_loopback())
    });
    if !(url.scheme() == "https" || (url.scheme() == "http" && local))
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
    {
        bail!(
            "provider base_url must use HTTPS (HTTP allowed for loopback fixtures), without credentials, query or fragment"
        );
    }
    Ok(local)
}
