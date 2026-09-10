pub mod http;
pub mod openai;
pub mod openrouter;
pub mod registry;
pub mod zai;

#[cfg(test)]
mod native_tests;

use crate::{
    Result,
    error::Error,
    models::{ApiRequest, ModelInfo, ProviderOutput, ReasoningEffort, RequestContext},
};
use serde::{Deserialize, de::IntoDeserializer};

#[async_trait::async_trait]
pub trait Provider: Send + Sync {
    fn models(&self) -> Vec<ModelInfo>;
    fn convert_reasoning_effort(&self, model: &str, effort: &str) -> Result<String>;
    async fn execute(&self, request: ApiRequest, context: RequestContext)
    -> Result<ProviderOutput>;
}

pub(crate) fn validate_effort(effort: &str) -> Result<()> {
    ReasoningEffort::deserialize(effort.into_deserializer())
        .map(|_| ())
        .map_err(|_: serde::de::value::Error| Error::invalid("unsupported reasoning effort"))
}
