use super::{
    Provider, http,
    openai::{OpenAiProvider, state::Store},
    openrouter::OpenRouterProvider,
    zai::ZaiProvider,
};
use crate::{
    Result,
    config::{Config, ProviderConfig, validate_model},
    error::Error,
    models::ModelInfo,
};
use std::{collections::BTreeMap, sync::Arc};

pub struct Registry {
    providers: BTreeMap<String, Arc<dyn Provider>>,
}

impl Registry {
    pub async fn new(config: &Config) -> eyre::Result<Self> {
        let client = http::client(&config.server)?;
        let store = if config
            .providers
            .values()
            .any(|p| matches!(p, ProviderConfig::OpenAi(_)))
        {
            Some(Arc::new(
                Store::open(
                    config.server.state_dir.clone(),
                    config.server.max_state_bytes,
                    config.server.max_response_bytes,
                )
                .await?,
            ))
        } else {
            None
        };
        let mut providers: BTreeMap<String, Arc<dyn Provider>> = BTreeMap::new();
        for (name, provider) in &config.providers {
            let provider: Arc<dyn Provider> = match provider {
                ProviderConfig::OpenAi(c) => Arc::new(OpenAiProvider::new(
                    c.clone(),
                    client.clone(),
                    config.server.clone(),
                    store.as_ref().expect("OpenAI store initialized").clone(),
                )?),
                ProviderConfig::OpenRouter(c) => Arc::new(OpenRouterProvider::new(
                    c.clone(),
                    client.clone(),
                    config.server.clone(),
                )?),
                ProviderConfig::Zai(c) => Arc::new(ZaiProvider::new(
                    c.clone(),
                    client.clone(),
                    config.server.clone(),
                )?),
            };
            providers.insert(name.clone(), provider);
        }
        Ok(Self { providers })
    }

    pub fn resolve(&self, public: &str) -> Result<(Arc<dyn Provider>, String)> {
        let (prefix, model) = public
            .split_once('/')
            .ok_or_else(|| Error::invalid("model must be provider/native-model-id"))?;
        validate_model(model).map_err(|_| Error::invalid("invalid native model ID"))?;
        let provider = self
            .providers
            .get(prefix)
            .ok_or_else(|| Error::invalid("unknown model provider"))?;
        Ok((provider.clone(), model.to_owned()))
    }

    pub fn models(&self) -> Vec<ModelInfo> {
        self.providers
            .iter()
            .flat_map(|(prefix, provider)| {
                provider.models().into_iter().map(move |m| ModelInfo {
                    id: format!("{prefix}/{}", m.id),
                    display_name: m.display_name,
                })
            })
            .collect()
    }
}
