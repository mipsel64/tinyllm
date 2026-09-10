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
    cleanup: Option<tokio::task::JoinHandle<()>>,
}

impl Drop for Registry {
    fn drop(&mut self) {
        if let Some(cleanup) = &self.cleanup {
            cleanup.abort();
        }
    }
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
                Store::open_with_cleanup(
                    config.server.state_dir.clone(),
                    config.server.max_state_bytes,
                    config.server.max_response_bytes,
                    config.server.state_cleanup,
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
        let cleanup = store.and_then(|store| {
            config.server.state_cleanup.map(|policy| {
                let store = Arc::downgrade(&store);
                tracing::info!(idle_days=policy.idle_days, interval_seconds=policy.interval_seconds, "automatic continuation cleanup enabled");
                tokio::spawn(async move {
                    let mut interval = tokio::time::interval(std::time::Duration::from_secs(policy.interval_seconds));
                    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
                    loop {
                        interval.tick().await;
                        let Some(store) = store.upgrade() else { break; };
                        if let Err(error) = store.cleanup().await {
                            tracing::warn!(error = %format_args!("{error:#}"), "continuation cleanup failed; will retry next interval");
                        }
                    }
                })
            })
        });
        Ok(Self { providers, cleanup })
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, SystemTime};

    #[tokio::test]
    async fn cleanup_runs_on_interval_and_stops_with_registry() {
        let directory =
            std::env::temp_dir().join(format!("tinyllm-periodic-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir(&directory).unwrap();
        let old = SystemTime::now() - Duration::from_secs(86400 * 60);
        let marker = directory.join(".cleanup-start");
        std::fs::write(&marker, b"").unwrap();
        std::fs::File::open(marker)
            .unwrap()
            .set_modified(old)
            .unwrap();
        let config: Config = serde_json::from_value(serde_json::json!({
            "server":{"state_dir":directory,"state_cleanup":{"idle_days":1,"interval_seconds":1}},
            "providers":{"openai":{"type":"openai","auth":{"type":"ApiKey","options":"fixture"}}}
        }))
        .unwrap();
        let registry = Registry::new(&config).await.unwrap();
        let connection = rusqlite::Connection::open(directory.join("state.sqlite")).unwrap();
        for _ in 0..2 {
            connection.execute_batch("BEGIN; INSERT INTO records VALUES ('fixture', x'00'); INSERT INTO access VALUES ('fixture', 1, 1); COMMIT;").unwrap();
            tokio::time::timeout(Duration::from_secs(5), async {
                while connection
                    .query_row("SELECT count(*) FROM records", [], |row| {
                        row.get::<_, i64>(0)
                    })
                    .unwrap()
                    != 0
                {
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
            })
            .await
            .expect("periodic cleanup must run again after its first tick");
        }
        drop(connection);
        let worker = registry.cleanup.as_ref().unwrap().abort_handle();
        drop(registry);
        tokio::time::timeout(Duration::from_secs(5), async {
            while !worker.is_finished() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        let store = tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                if let Ok(store) = Store::open(directory.clone(), 10000, 1000).await {
                    break store;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("admitted cleanup must release its directory lock");
        drop(store);
        std::fs::remove_dir_all(directory).unwrap();
    }
}
