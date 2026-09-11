use crate::{Result, error::Error, models::config::StateCleanup, providers::openai::protocol};
use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD};
use eyre::WrapErr;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, HashSet},
    path::{Path, PathBuf},
    sync::{Arc, Weak},
    time::{Duration, SystemTime},
};
use tokio::{fs, sync::Mutex};

mod sqlite;

pub use sqlite::FILE as DATABASE_FILE;
use sqlite::{Database, Loaded};

pub const PREFIX: &str = "tinyllm:v1:";
pub(crate) const REFERENCE_FIELD: &str = "tinyllm_continuation";
pub(crate) const TOOL_PREFIX: &str = "toolu_tinyllm_";
const MAX_TOOL_ID_BYTES: usize = 1024;

pub(super) fn encode_tool_id(reference: &str, id: &str) -> Result<String> {
    let reference = Store::id(reference)?;
    if id.is_empty() || id.len() > MAX_TOOL_ID_BYTES {
        return Err(Error::upstream("tool call ID must contain 1–1024 bytes"));
    }
    Ok(format!(
        "{TOOL_PREFIX}{reference}_{}",
        URL_SAFE_NO_PAD.encode(id)
    ))
}

pub(super) fn decode_tool_id(id: &str) -> Result<Option<(String, String)>> {
    let Some(encoded) = id.strip_prefix(TOOL_PREFIX) else {
        return Ok(None);
    };
    let invalid = || Error::invalid("invalid tinyllm tool call ID");
    if encoded.len() > 33 + MAX_TOOL_ID_BYTES.div_ceil(3) * 4 {
        return Err(invalid());
    }
    let (uuid, encoded) = encoded.split_once('_').ok_or_else(invalid)?;
    let reference = format!("{PREFIX}{uuid}");
    Store::id(&reference)?;
    let bytes = URL_SAFE_NO_PAD.decode(encoded).map_err(|_| invalid())?;
    if bytes.is_empty()
        || bytes.len() > MAX_TOOL_ID_BYTES
        || URL_SAFE_NO_PAD.encode(&bytes) != encoded
    {
        return Err(invalid());
    }
    let native = String::from_utf8(bytes).map_err(|_| invalid())?;
    Ok(Some((reference, native)))
}

pub(super) fn mark_block(block: &mut Value, reference: &str) -> Result<()> {
    match block["type"].as_str() {
        Some("text") => block[REFERENCE_FIELD] = json!(reference),
        Some("tool_use") => {
            block["id"] = json!(encode_tool_id(reference, protocol::string(block, "id")?)?);
        }
        _ => {}
    }
    Ok(())
}

fn block_reference(block: &Value) -> Result<Option<(String, Option<String>)>> {
    if block.get(REFERENCE_FIELD).is_some() && block["type"] != "text" {
        return Err(Error::invalid(
            "continuation metadata must be on a text block",
        ));
    }
    let reference = match block["type"].as_str() {
        Some("redacted_thinking") => protocol::string(block, "data")?,
        Some("text") if block.get(REFERENCE_FIELD).is_some() => {
            protocol::string(block, REFERENCE_FIELD)?
        }
        Some("tool_use") => {
            return Ok(decode_tool_id(protocol::string(block, "id")?)?
                .map(|(reference, id)| (reference, Some(id))));
        }
        _ => return Ok(None),
    };
    Store::id(reference)?;
    Ok(Some((reference.to_owned(), None)))
}

fn normalize_content(content: Vec<Value>) -> Result<Vec<Value>> {
    let mut normalized = Vec::with_capacity(content.len());
    let mut current = None;
    for mut block in content {
        if let Some((reference, native_id)) = block_reference(&block)? {
            if block["type"] == "redacted_thinking" {
                current = Some(reference);
                normalized.push(block);
                continue;
            }
            if let Some(id) = native_id {
                block["id"] = json!(id);
            } else {
                block.as_object_mut().unwrap().remove(REFERENCE_FIELD);
            }
            if current.as_ref() != Some(&reference) {
                normalized.push(json!({"type":"redacted_thinking","data":reference}));
                current = Some(reference);
            }
        }
        normalized.push(block);
    }
    Ok(normalized)
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Usage {
    pub records: u64,
    pub temporary_files: u64,
    pub bytes: u64,
}

impl Usage {
    fn add(&mut self, bytes: u64, temporary: bool) -> eyre::Result<()> {
        self.bytes = self
            .bytes
            .checked_add(bytes)
            .ok_or_else(|| eyre::eyre!("state size overflow"))?;
        if temporary {
            self.temporary_files += 1;
        } else {
            self.records += 1;
        }
        Ok(())
    }

    pub fn is_high(&self, limit: u64) -> bool {
        self.bytes >= limit.saturating_sub(limit / 5)
    }
}

#[derive(Debug, Default)]
pub struct PruneReport {
    pub before: Usage,
    pub selected: Usage,
    pub after: Usage,
}

#[derive(Serialize, Deserialize)]
struct Record {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    provider: Option<String>,
    model: String,
    output: Vec<Value>,
    content: Vec<Value>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    tool_defaults: BTreeMap<String, BTreeMap<String, Value>>,
}

pub(super) fn tool_defaults(tools: &Value) -> BTreeMap<String, BTreeMap<String, Value>> {
    tools
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|tool| {
            let name = tool["name"].as_str()?;
            let schema = &tool["input_schema"];
            let required: HashSet<_> = schema["required"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .collect();
            let defaults: BTreeMap<_, _> = schema["properties"]
                .as_object()?
                .iter()
                .filter_map(|(name, property)| {
                    if required.contains(name.as_str()) {
                        return None;
                    }
                    Some((name.clone(), property.get("default")?.clone()))
                })
                .collect();
            (!defaults.is_empty()).then(|| (name.to_owned(), defaults))
        })
        .collect()
}

struct Bookkeeping {
    bytes: u64,
    pins: BTreeMap<String, Weak<()>>,
    database: Database,
}

#[derive(Clone)]
pub struct Store {
    _lock: Arc<std::fs::File>,
    used: Arc<Mutex<Bookkeeping>>,
    limit: u64,
    record_limit: usize,
    cleanup: Option<(Duration, SystemTime)>,
}

impl Store {
    pub async fn open(directory: PathBuf, limit: u64, record_limit: usize) -> eyre::Result<Self> {
        Self::open_with_cleanup(directory, limit, record_limit, None).await
    }

    pub async fn open_with_cleanup(
        directory: PathBuf,
        limit: u64,
        record_limit: usize,
        policy: Option<StateCleanup>,
    ) -> eyre::Result<Self> {
        let retention = policy.map(StateCleanup::retention).transpose()?;
        if !directory_exists(&directory).await? {
            fs::create_dir_all(&directory).await.wrap_err_with(|| {
                format!("cannot create state directory {}", directory.display())
            })?;
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&directory, std::fs::Permissions::from_mode(0o700)).await?;
        }
        tokio::task::spawn_blocking(move || {
            let lock = lock_directory(&directory)?;
            let cleanup = cleanup_grace(&directory, retention)?;
            let mut database = Database::open(&directory, true)?;
            database.migrate(&directory)?;
            let mut report = PruneReport {
                before: database.usage(None)?,
                ..Default::default()
            };
            scan_files(&directory, None, false, &mut report)?;
            let usage = report.before;
            tracing::info!(
                state_records = usage.records,
                state_temporary_files = usage.temporary_files,
                state_bytes = usage.bytes,
                max_state_bytes = limit,
                "continuation state usage"
            );
            warn_if_high(usage, limit);
            Ok(Self {
                _lock: Arc::new(lock),
                used: Arc::new(Mutex::new(Bookkeeping {
                    bytes: usage.bytes,
                    pins: BTreeMap::new(),
                    database,
                })),
                limit,
                record_limit,
                cleanup,
            })
        })
        .await?
    }

    pub(crate) async fn cleanup(&self) -> eyre::Result<Usage> {
        self.cleanup_at(SystemTime::now()).await
    }

    async fn cleanup_at(&self, now: SystemTime) -> eyre::Result<Usage> {
        let mut used = self.used.clone().lock_owned().await;
        let store = self.clone();
        tokio::task::spawn_blocking(move || store.cleanup_locked(&mut used, now, None)).await?
    }

    fn cleanup_locked(
        &self,
        used: &mut Bookkeeping,
        now: SystemTime,
        needed: Option<u64>,
    ) -> eyre::Result<Usage> {
        let removed = Usage::default();
        let Some((retention, grace)) = self.cleanup else {
            return Ok(removed);
        };
        let Some(cutoff) = now.checked_sub(retention) else {
            return Ok(removed);
        };
        used.pins.retain(|_, pin| pin.strong_count() > 0);
        if grace >= cutoff {
            return Ok(removed);
        }
        let removed = used
            .database
            .prune(cutoff, &used.pins, needed)
            .wrap_err_with(|| "cannot clean expired continuation state")?;
        used.bytes = used.bytes.saturating_sub(removed.bytes);
        if removed.records > 0 {
            tracing::info!(
                removed_records = removed.records,
                removed_bytes = removed.bytes,
                state_bytes = used.bytes,
                "expired continuation state removed"
            );
        }
        Ok(removed)
    }

    pub async fn pin(&self, reference: &str) -> Result<Option<Arc<()>>> {
        let id = Self::id(reference)?.to_owned();
        if self.cleanup.is_none() {
            return Ok(None);
        }
        let mut used = self.used.lock().await;
        used.pins.retain(|_, pin| pin.strong_count() > 0);
        Ok(Some(Self::pin_locked(&mut used, id)))
    }

    fn pin_locked(used: &mut Bookkeeping, id: String) -> Arc<()> {
        let entry = used.pins.entry(id).or_default();
        let pin = entry.upgrade().unwrap_or_else(|| Arc::new(()));
        *entry = Arc::downgrade(&pin);
        pin
    }

    pub async fn status(directory: &Path) -> eyre::Result<Usage> {
        if !directory_exists(directory).await? {
            return Ok(Usage::default());
        }
        let directory = directory.to_owned();
        tokio::task::spawn_blocking(move || Ok(scan(&directory, None, false)?.before)).await?
    }

    pub async fn prune(
        directory: &Path,
        cutoff: SystemTime,
        apply: bool,
    ) -> eyre::Result<PruneReport> {
        if !directory_exists(directory).await? {
            return Ok(PruneReport::default());
        }
        let directory = directory.to_owned();
        tokio::task::spawn_blocking(move || {
            let _lock = lock_directory(&directory)?;
            scan(&directory, Some(cutoff), apply)
        })
        .await?
    }

    pub fn reference() -> String {
        format!("{PREFIX}{}", uuid::Uuid::new_v4().simple())
    }

    fn id(reference: &str) -> Result<&str> {
        let id = reference
            .strip_prefix(PREFIX)
            .filter(|id| id.len() == 32 && id.bytes().all(|c| c.is_ascii_hexdigit()))
            .ok_or_else(|| Error::invalid("invalid tinyllm continuation reference"))?;
        Ok(id)
    }

    pub async fn save(
        &self,
        reference: &str,
        model: &str,
        response: &Value,
        content: Value,
    ) -> Result<()> {
        self.save_with_defaults(reference, model, None, response, content, &BTreeMap::new())
            .await
    }

    async fn save_with_defaults(
        &self,
        reference: &str,
        model: &str,
        provider: Option<&str>,
        response: &Value,
        content: Value,
        defaults: &BTreeMap<String, BTreeMap<String, Value>>,
    ) -> Result<()> {
        let content = content
            .as_array()
            .ok_or_else(|| Error::upstream("invalid response content"))?
            .clone();
        let mut tool_defaults = BTreeMap::new();
        for block in &content {
            if block["type"] == "tool_use"
                && let Some(name) = block["name"].as_str()
                && let Some(properties) = defaults.get(name)
            {
                tool_defaults
                    .entry(name.to_owned())
                    .or_insert_with(|| properties.clone());
            }
        }
        let record = Record {
            provider: provider.map(str::to_owned),
            model: model.into(),
            output: response["output"]
                .as_array()
                .ok_or_else(|| Error::upstream("missing native output"))?
                .clone(),
            content,
            tool_defaults,
        };
        let data = serde_json::to_vec(&record)
            .map_err(|_| Error::upstream("cannot serialize continuation"))?;
        if data.len() > self.record_limit {
            return Err(Error::upstream("continuation exceeds max_response_bytes"));
        }
        let id = Self::id(reference)?.to_owned();
        let mut used = self.used.clone().lock_owned().await;
        let store = self.clone();
        tokio::task::spawn_blocking(move || {
            if used.database.contains(&id).map_err(|error| state_error("cannot inspect continuation state", error))? {
                return Err(Error::upstream("continuation reference already exists"));
            }
            if data.len() as u64 > store.limit {
                return Err(Error::upstream("continuation exceeds max_state_bytes"));
            }
            if used.bytes.checked_add(data.len() as u64).is_none_or(|n| n > store.limit) {
                let needed = used.bytes.saturating_add(data.len() as u64).saturating_sub(store.limit);
                store.cleanup_locked(&mut used, SystemTime::now(), Some(needed))
                    .map_err(|error| state_error("cannot clean expired continuation state", error))?;
            }
            let next = used.bytes.checked_add(data.len() as u64)
                .filter(|n| *n <= store.limit)
                .ok_or_else(|| Error::upstream("continuation store is full; increase max_state_bytes or stop the gateway and run tinyllm state prune"))?;
            let was_high = Usage { bytes: used.bytes, ..Default::default() }.is_high(store.limit);
            used.database.save(&id, &data, SystemTime::now())
                .map_err(|error| state_error("cannot persist continuation state", error))?;
            used.bytes = next;
            if !was_high {
                warn_if_high(Usage { bytes: used.bytes, ..Default::default() }, store.limit);
            }
            Ok(())
        }).await.map_err(|error| state_error("continuation write failed", error))?
    }

    pub async fn save_scoped(
        &self,
        reference: &str,
        provider: &str,
        model: &str,
        response: &Value,
        content: Value,
        defaults: &BTreeMap<String, BTreeMap<String, Value>>,
    ) -> Result<()> {
        self.save_with_defaults(
            reference,
            &format!("{provider}/{model}"),
            Some(provider),
            response,
            content,
            defaults,
        )
        .await
    }

    pub async fn restore_scoped(
        &self,
        request: &Value,
        provider: &str,
        model: &str,
    ) -> Result<BTreeMap<usize, Vec<Value>>> {
        self.restore_scoped_pinned(request, provider, model, &mut Vec::new())
            .await
    }

    pub async fn restore_scoped_pinned(
        &self,
        request: &Value,
        provider: &str,
        model: &str,
        pins: &mut Vec<Arc<()>>,
    ) -> Result<BTreeMap<usize, Vec<Value>>> {
        self.restore_for(request, provider, model, pins).await
    }

    #[cfg(test)]
    pub async fn restore(
        &self,
        request: &Value,
        model: &str,
    ) -> Result<BTreeMap<usize, Vec<Value>>> {
        self.restore_for(request, "openai", model, &mut Vec::new())
            .await
    }

    async fn restore_for(
        &self,
        request: &Value,
        provider: &str,
        model: &str,
        pins: &mut Vec<Arc<()>>,
    ) -> Result<BTreeMap<usize, Vec<Value>>> {
        let mut restored = BTreeMap::new();
        let mut references = HashSet::new();
        let mut restored_bytes = 0usize;
        let Some(messages) = request["messages"].as_array() else {
            return Ok(restored);
        };
        if self.cleanup.is_some() {
            let mut used = self.used.lock().await;
            used.pins.retain(|_, pin| pin.strong_count() > 0);
            let mut pinned = HashSet::new();
            for block in messages
                .iter()
                .filter(|message| message["role"] == "assistant")
                .filter_map(|message| message["content"].as_array())
                .flatten()
            {
                if let Some((reference, _)) = block_reference(block)? {
                    let id = Self::id(&reference)?.to_owned();
                    if pinned.insert(id.clone()) {
                        pins.push(Self::pin_locked(&mut used, id));
                    }
                }
            }
        }
        let mut index = 0;
        let mut local_command = false;
        while index < messages.len() {
            if messages[index]["role"] != "assistant" {
                if !local_command && messages[index]["role"] == "user" {
                    let content = &messages[index]["content"];
                    local_command = content
                        .as_str()
                        .is_some_and(|text| text.starts_with("<local-command-stdout>"))
                        || content.as_array().is_some_and(|blocks| {
                            blocks.iter().any(|block| {
                                block["text"]
                                    .as_str()
                                    .is_some_and(|text| text.starts_with("<local-command-stdout>"))
                            })
                        });
                }
                index += 1;
                continue;
            }
            let first = index;
            let mut content = Vec::new();
            while index < messages.len() && messages[index]["role"] == "assistant" {
                content.extend(protocol::blocks(&messages[index]["content"])?);
                index += 1;
            }
            // A copied fork call may refer to an unfinished parent response.
            if !content.iter().any(|b| b["type"] == "redacted_thinking")
                && is_fork_bootstrap(&content, messages.get(index))
            {
                tracing::debug!(
                    assistant_message_index = first,
                    "Claude fork reasoning boundary"
                );
                continue;
            }
            let mut content = normalize_content(content)?;
            if !content.iter().any(|b| b["type"] == "redacted_thinking") {
                // Claude Code repeats this acknowledgement when resuming local commands.
                if local_command
                    && content.len() == 1
                    && content[0]["type"] == "text"
                    && content[0]["text"] == "No response requested."
                {
                    protocol::fields(&content[0], &["type", "text", "cache_control"])?;
                    continue;
                }
                let block_types: Vec<_> = content
                    .iter()
                    .take(8)
                    .map(|block| match block["type"].as_str() {
                        Some("text") => "text",
                        Some("tool_use") => "tool_use",
                        Some("thinking") => "thinking",
                        _ => "other",
                    })
                    .collect();
                tracing::warn!(
                    provider,
                    model,
                    assistant_message_index = first,
                    block_count = content.len(),
                    ?block_types,
                    "assistant history lacks a continuation reference"
                );
                return Err(Error::invalid(
                    "assistant history is missing tinyllm continuation references; imported histories or stripped references cannot preserve native reasoning; start a new conversation with a user summary",
                ));
            }
            for block in &mut content {
                if let Some(object) = block.as_object_mut() {
                    object.remove("cache_control");
                }
            }
            let mut native = Vec::new();
            let mut start = 0;
            while start < content.len() {
                if content[start]["type"] != "redacted_thinking" {
                    return Err(Error::invalid(
                        "continuation reference must precede its assistant content",
                    ));
                }
                protocol::fields(&content[start], &["type", "data"])?;
                let reference = protocol::string(&content[start], "data")?;
                if !references.insert(reference.to_owned()) {
                    return Err(Error::invalid("duplicate continuation reference"));
                }
                let id = Self::id(reference)?.to_owned();
                let used = self.used.clone().lock_owned().await;
                let limit = self.record_limit;
                let store = self.clone();
                let data = tokio::task::spawn_blocking(move || {
                    let _store = store;
                    used.database.load(&id, limit)
                })
                .await
                .map_err(|error| state_error("continuation read failed", error))?
                .map_err(|error| state_error("cannot read continuation state", error))?;
                let data = match data {
                    Loaded::Data(data) => data,
                    Loaded::Missing => {
                        return Err(Error::invalid(
                            "continuation state is missing; restore the state directory or start a new conversation",
                        ));
                    }
                    Loaded::TooLarge => {
                        return Err(Error::upstream(
                            "continuation record exceeds configured limit",
                        ));
                    }
                };
                restored_bytes = restored_bytes.checked_add(data.len()).filter(|n| *n <= self.record_limit).ok_or_else(|| Error::invalid("restored continuation history exceeds max_response_bytes; compact the conversation"))?;
                let record: Record = serde_json::from_slice(&data)
                    .map_err(|_| Error::upstream("corrupt continuation state"))?;
                let end = content[start + 1..]
                    .iter()
                    .position(|b| b["type"] == "redacted_thinking")
                    .map_or(content.len(), |p| start + 1 + p);
                let saved_provider = match record.provider.as_deref() {
                    Some(provider) => provider,
                    None if !record.model.contains('/') || record.model.starts_with("openai/") => {
                        "openai"
                    }
                    None => {
                        return Err(Error::invalid(
                            "legacy continuation has ambiguous provider scope; start a new conversation",
                        ));
                    }
                };
                if saved_provider != provider {
                    return Err(Error::invalid(
                        "continuation provider changed; keep its provider or start a new conversation",
                    ));
                }
                // Claude Code materializes omitted tool defaults before replaying calls.
                for (saved, replayed) in record.content.iter().zip(&mut content[start..end]) {
                    if saved["type"] == "tool_use"
                        && let Some(name) = saved["name"].as_str()
                        && let Some(defaults) = record.tool_defaults.get(name)
                        && let Some(original) = saved["input"].as_object()
                        && let Some(input) =
                            replayed.get_mut("input").and_then(Value::as_object_mut)
                    {
                        input.retain(|key, value| {
                            original.contains_key(key) || defaults.get(key) != Some(value)
                        });
                    }
                }
                if record.content != content[start..end] {
                    return Err(Error::invalid(
                        "assistant content differs from its continuation record; replay it unchanged or start a new conversation",
                    ));
                }
                native.extend(record.output);
                start = end;
            }
            restored.insert(first, native);
            for other in first + 1..index {
                restored.insert(other, Vec::new());
            }
        }
        if self.cleanup.is_some() && !references.is_empty() {
            let ids = references
                .iter()
                .map(|r| Self::id(r).map(str::to_owned))
                .collect::<Result<_>>()?;
            let mut used = self.used.clone().lock_owned().await;
            let store = self.clone();
            tokio::task::spawn_blocking(move || {
                let _store = store;
                used.database.touch(&ids, SystemTime::now())
            })
            .await
            .map_err(|error| state_error("continuation timestamp update failed", error))?
            .map_err(|error| state_error("cannot update continuation timestamp", error))?;
        }
        Ok(restored)
    }
}

fn state_error(message: &'static str, error: impl std::fmt::Display) -> Error {
    tracing::warn!(error = %format_args!("{error:#}"), "{message}");
    Error::upstream(message)
}

fn is_fork_bootstrap(content: &[Value], next: Option<&Value>) -> bool {
    let Some(results) = next
        .filter(|message| message["role"] == "user")
        .and_then(|message| message["content"].as_array())
    else {
        return false;
    };
    let Some(text) = results
        .get(content.len())
        .filter(|block| block["type"] == "text")
        .and_then(|block| block["text"].as_str())
    else {
        return false;
    };
    let Some((context, directive)) = text.split_once("</fork-boilerplate>\n\nYour directive: ")
    else {
        return false;
    };
    if !context.starts_with("<fork-boilerplate>\nYou are a worker fork. The transcript above is the parent's history — inherited reference, not your situation. You are NOT a continuation of that agent. Execute ONE directive, then stop.")
        || directive.trim().is_empty()
    {
        return false;
    }
    !content.is_empty()
        && content.iter().zip(results).all(|(call, result)| {
            call["type"] == "tool_use"
                && call["name"] == "Agent"
                && call["input"]["subagent_type"] == "fork"
                && call["id"].as_str().is_some_and(|id| !id.is_empty())
                && result["type"] == "tool_result"
                && result["tool_use_id"] == call["id"]
                && result.get("is_error").is_none_or(|value| value == false)
                && result["content"].as_array().is_some_and(|parts| {
                    parts.len() == 1
                        && parts[0]["type"] == "text"
                        && parts[0]["text"] == "Fork started — processing in background"
                })
        })
}

fn cleanup_grace(
    directory: &Path,
    retention: Option<Duration>,
) -> eyre::Result<Option<(Duration, SystemTime)>> {
    let path = directory.join(".cleanup-start");
    let existing = match std::fs::symlink_metadata(&path) {
        Ok(metadata) if metadata.is_file() => Some(metadata.modified()?),
        Ok(_) => eyre::bail!("cleanup grace marker must be a regular file"),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => return Err(error).wrap_err_with(|| "cannot inspect cleanup grace marker"),
    };
    let Some(retention) = retention else {
        if existing.is_some() {
            std::fs::remove_file(path)?;
        }
        return Ok(None);
    };
    let grace = if let Some(modified) = existing {
        modified
    } else {
        let mut options = std::fs::OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let file = options
            .open(path)
            .wrap_err_with(|| "cannot create cleanup grace marker")?;
        file.sync_all()?;
        file.metadata()?.modified()?
    };
    Ok(Some((retention, grace)))
}

fn record_name(name: &std::ffi::OsStr) -> Option<(&str, bool)> {
    let (id, extension) = name.to_str()?.rsplit_once('.')?;
    (matches!(extension, "json" | "tmp")
        && id.len() == 32
        && id.bytes().all(|b| b.is_ascii_hexdigit()))
    .then_some((id, extension == "tmp"))
}

async fn directory_exists(directory: &Path) -> eyre::Result<bool> {
    match fs::symlink_metadata(directory).await {
        Ok(metadata) if metadata.is_dir() => Ok(true),
        Ok(_) => eyre::bail!(
            "state directory must be a directory, not a symlink: {}",
            directory.display()
        ),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error)
            .wrap_err_with(|| format!("cannot inspect state directory {}", directory.display())),
    }
}

fn lock_directory(directory: &Path) -> eyre::Result<std::fs::File> {
    let path = directory.join(".lock");
    match std::fs::symlink_metadata(&path) {
        Ok(metadata) if metadata.is_file() => {}
        Ok(_) => eyre::bail!("state lock must be a regular file: {}", path.display()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(error)
                .wrap_err_with(|| format!("cannot inspect state lock {}", path.display()));
        }
    }
    let mut options = std::fs::OpenOptions::new();
    options.read(true).write(true).create(true).truncate(false);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let lock = options
        .open(&path)
        .wrap_err_with(|| format!("cannot open state lock {}", path.display()))?;
    lock.try_lock().wrap_err_with(|| format!("state directory {} is already in use or cannot be locked; stop the gateway before cleanup", directory.display()))?;
    Ok(lock)
}

fn scan(directory: &Path, cutoff: Option<SystemTime>, apply: bool) -> eyre::Result<PruneReport> {
    let mut report = PruneReport::default();
    let database_exists = match std::fs::symlink_metadata(directory.join(sqlite::FILE)) {
        Ok(_) => true,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => false,
        Err(error) => return Err(error).wrap_err_with(|| "cannot inspect state database"),
    };
    if database_exists {
        let mut database = Database::open(directory, apply)?;
        report.before = database.usage(None)?;
        report.selected = match cutoff {
            Some(cutoff) => database.usage(Some(cutoff))?,
            None => Usage::default(),
        };
        if apply && let Some(cutoff) = cutoff {
            database.prune(cutoff, &BTreeMap::new(), None)?;
        }
        report.after = database.usage(None)?;
    }
    scan_files(directory, cutoff, apply, &mut report)?;
    Ok(report)
}

fn scan_files(
    directory: &Path,
    cutoff: Option<SystemTime>,
    apply: bool,
    report: &mut PruneReport,
) -> eyre::Result<()> {
    for entry in std::fs::read_dir(directory)? {
        let entry = entry?;
        let name = entry.file_name();
        let Some((_, temporary)) = record_name(&name) else {
            continue;
        };
        let path = entry.path();
        let metadata = match std::fs::symlink_metadata(&path) {
            Ok(metadata) if metadata.is_file() => metadata,
            Ok(_) => continue,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => return Err(error).wrap_err_with(|| "cannot inspect continuation file"),
        };
        report.before.add(metadata.len(), temporary)?;
        let selected = match cutoff {
            Some(cutoff) => metadata.modified()? < cutoff,
            None => false,
        };
        if selected {
            report.selected.add(metadata.len(), temporary)?;
            if apply {
                std::fs::remove_file(&path).wrap_err_with(|| "cannot remove continuation file")?;
                continue;
            }
        }
        report.after.add(metadata.len(), temporary)?;
    }
    Ok(())
}

fn warn_if_high(usage: Usage, limit: u64) {
    if usage.is_high(limit) {
        tracing::warn!(
            state_bytes = usage.bytes,
            max_state_bytes = limit,
            "continuation state is at least 80% full; stop the gateway and preview tinyllm state prune before cleanup; pruned turns cannot resume"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn tool_reference_codec_is_bounded_canonical_and_reversible() {
        let reference = Store::reference();
        let nested = encode_tool_id(&Store::reference(), "native").unwrap();
        for native in [
            "call_abcdefghijklmnopqrstuvwxyz012345".to_owned(),
            "tool/🦀".to_owned(),
            "x".repeat(MAX_TOOL_ID_BYTES),
            nested,
        ] {
            let id = encode_tool_id(&reference, &native).unwrap();
            assert!(
                id.bytes()
                    .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-'))
            );
            assert_eq!(
                decode_tool_id(&id).unwrap(),
                Some((reference.clone(), native))
            );
        }
        assert!(decode_tool_id("call_native").unwrap().is_none());
        assert!(encode_tool_id(&reference, "").is_err());
        assert!(encode_tool_id(&reference, &"x".repeat(MAX_TOOL_ID_BYTES + 1)).is_err());
        let prefix = format!("{TOOL_PREFIX}{}_", Store::id(&reference).unwrap());
        for id in [
            TOOL_PREFIX.to_owned(),
            format!("{TOOL_PREFIX}invalid_YQ"),
            prefix.clone(),
            format!("{prefix}YQ=="),
            format!("{prefix}YR"),
            format!("{prefix}_w"),
            format!("{prefix}{}", "A".repeat(2000)),
        ] {
            assert!(decode_tool_id(&id).is_err(), "{id}");
        }
    }

    #[tokio::test]
    async fn carrier_normalization_rejects_conflicts_edits_and_partial_records() {
        let directory = std::env::temp_dir().join(format!(
            "tinyllm-carrier-validation-{}",
            uuid::Uuid::new_v4()
        ));
        let store = Store::open(directory.clone(), 100_000, 50_000)
            .await
            .unwrap();
        let mut contents = Vec::new();
        for _ in 0..2 {
            let reference = Store::reference();
            let content = json!([
                {"type":"redacted_thinking","data":reference},
                {"type":"text","text":"checking"},
                {"type":"tool_use","id":"call_a","name":"lookup","input":{}},
                {"type":"tool_use","id":"call_b","name":"lookup","input":{}}
            ]);
            store
                .save_scoped(
                    &reference,
                    "openai",
                    "old",
                    &json!({"output":[]}),
                    content.clone(),
                    &BTreeMap::new(),
                )
                .await
                .unwrap();
            let mut wire = content.as_array().unwrap().clone();
            for block in &mut wire {
                mark_block(block, &reference).unwrap();
            }
            contents.push(wire);
        }
        let restored =
            |content: Vec<Value>| json!({"messages":[{"role":"assistant","content":content}]});
        assert!(
            store
                .restore_scoped(&restored(contents[0].clone()), "openai", "new")
                .await
                .is_ok()
        );
        let mut stripped = contents[0][1..].to_vec();
        assert!(
            store
                .restore_scoped(&restored(stripped.clone()), "openai", "new")
                .await
                .is_ok()
        );
        let mut invalid = Vec::new();
        let mut conflict = contents[0].clone();
        conflict[1][REFERENCE_FIELD] = contents[1][0]["data"].clone();
        invalid.push(conflict);
        let mut conflict = stripped.clone();
        conflict[1]["id"] = contents[1][2]["id"].clone();
        invalid.push(conflict);
        let mut reordered = stripped.clone();
        reordered.swap(1, 2);
        invalid.push(reordered);
        invalid.push(stripped[..2].to_vec());
        let mut duplicate = stripped.clone();
        duplicate.extend(stripped.clone());
        invalid.push(duplicate);
        let mut extra = stripped.clone();
        extra.push(json!({"type":"text","text":"unreferenced extra"}));
        invalid.push(extra);
        let mut malformed = stripped.clone();
        malformed[0][REFERENCE_FIELD] = json!(42);
        invalid.push(malformed);
        let mut malformed = stripped.clone();
        malformed[1][REFERENCE_FIELD] = contents[0][0]["data"].clone();
        invalid.push(malformed);
        stripped[1]["input"] = json!({"unexpected":true});
        invalid.push(stripped);
        for content in invalid {
            assert!(
                store
                    .restore_scoped(&restored(content.clone()), "openai", "new")
                    .await
                    .is_err(),
                "{content:?}"
            );
        }
        let combined: Vec<_> = contents
            .iter()
            .flat_map(|content| content[1..].to_vec())
            .collect();
        assert!(
            store
                .restore_scoped(&restored(combined), "openai", "new")
                .await
                .is_ok()
        );
        let missing = json!({"messages":[{"role":"assistant","content":[{"type":"text","text":"missing","tinyllm_continuation":Store::reference()}]}]});
        assert!(
            store
                .restore_scoped(&missing, "openai", "new")
                .await
                .unwrap_err()
                .message
                .contains("state is missing")
        );
        drop(store);
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[tokio::test]
    async fn provider_scope_is_explicit_and_ambiguous_legacy_names_fail_closed() {
        let directory =
            std::env::temp_dir().join(format!("tinyllm-provider-scope-{}", uuid::Uuid::new_v4()));
        let store = Store::open(directory.clone(), 100_000, 50_000)
            .await
            .unwrap();
        let native = json!({"output":[]});
        for legacy in ["old", "openai/old", "other/old", "org/path/model"] {
            let reference = Store::reference();
            let content = json!([{"type":"redacted_thinking","data":reference}]);
            store
                .save(&reference, legacy, &native, content.clone())
                .await
                .unwrap();
            let req = json!({"messages":[{"role":"assistant","content":content}]});
            assert_eq!(
                store.restore_scoped(&req, "openai", "new").await.is_ok(),
                matches!(legacy, "old" | "openai/old")
            );
            assert!(store.restore_scoped(&req, "other", "old").await.is_err());
        }
        let reference = Store::reference();
        let content = json!([{"type":"redacted_thinking","data":reference}]);
        store
            .save_scoped(
                &reference,
                "other",
                "org/path/model",
                &native,
                content.clone(),
                &BTreeMap::new(),
            )
            .await
            .unwrap();
        let req = json!({"messages":[{"role":"assistant","content":content}]});
        assert!(store.restore_scoped(&req, "other", "new").await.is_ok());
        assert!(
            store
                .restore_scoped(&req, "openai", "other/org/path/model")
                .await
                .is_err()
        );
        drop(store);
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[tokio::test]
    async fn migration_survives_lowered_response_and_store_limits() {
        let directory =
            std::env::temp_dir().join(format!("tinyllm-migrate-limits-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir(&directory).unwrap();
        let reference = Store::reference();
        let content = json!([{"type":"redacted_thinking","data":reference}]);
        let output = json!([{"type":"reasoning","encrypted_content":"x".repeat(20000)}]);
        let record = json!({"model":"openai/m","content":content,"output":output});
        let path = directory.join(format!("{}.json", Store::id(&reference).unwrap()));
        let bytes = serde_json::to_vec(&record).unwrap();
        std::fs::write(&path, &bytes).unwrap();
        let request = json!({"messages":[{"role":"assistant","content":content}]});
        let store = Store::open(directory.clone(), 4096, 1024).await.unwrap();
        assert!(!path.exists());
        assert_eq!(
            Store::status(&directory).await.unwrap().bytes,
            bytes.len() as u64
        );
        assert!(
            store
                .restore_scoped(&request, "openai", "m")
                .await
                .unwrap_err()
                .message
                .contains("exceeds configured limit")
        );
        drop(store);
        let store = Store::open(directory.clone(), 100000, 100000)
            .await
            .unwrap();
        assert_eq!(
            store.restore_scoped(&request, "openai", "m").await.unwrap()[&0],
            *output.as_array().unwrap()
        );
        drop(store);
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[tokio::test]
    async fn sqlite_migrates_existing_references_and_keeps_credentials_as_files() {
        let directory =
            std::env::temp_dir().join(format!("tinyllm-migrate-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(directory.join("auth")).unwrap();
        std::fs::write(directory.join("auth/openai.json"), b"credentials").unwrap();
        let reference = "tinyllm:v1:00000000000000000000000000000001";
        let content = json!([{"type":"redacted_thinking","data":reference}]);
        let record = json!({"model":"openai/m", "content":content,
            "output":[{"type":"reasoning","encrypted_content":"original"}]});
        let legacy = directory.join("00000000000000000000000000000001.json");
        std::fs::write(&legacy, serde_json::to_vec(&record).unwrap()).unwrap();
        let request = json!({"messages":[{"role":"assistant","content":content}]});
        let store = Store::open(directory.clone(), 10000, 1000).await.unwrap();
        assert_eq!(
            store.restore_scoped(&request, "openai", "m").await.unwrap()[&0],
            record["output"].as_array().unwrap().clone()
        );
        assert!(
            !legacy.exists(),
            "committed migration must retire the source file"
        );
        assert!(
            std::fs::read(directory.join("state.sqlite"))
                .unwrap()
                .starts_with(b"SQLite format 3\0")
        );
        drop(store);
        let store = Store::open(directory.clone(), 10000, 1000).await.unwrap();
        assert!(store.restore_scoped(&request, "openai", "m").await.is_ok());
        assert_eq!(Store::status(&directory).await.unwrap().records, 1);
        assert_eq!(
            std::fs::read(directory.join("auth/openai.json")).unwrap(),
            b"credentials"
        );
        drop(store);
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[tokio::test]
    async fn duplicate_save_under_pressure_does_not_evict_history() {
        let directory =
            std::env::temp_dir().join(format!("tinyllm-duplicate-quota-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir(&directory).unwrap();
        let old = SystemTime::now() - Duration::from_secs(86400 * 60);
        write_at(&directory, ".cleanup-start", b"", old);
        let store =
            Store::open_with_cleanup(directory.clone(), 100, 100, Some(StateCleanup::default()))
                .await
                .unwrap();
        let reference = Store::reference();
        let response = json!({"output":[]});
        store
            .save(&reference, "m", &response, json!([]))
            .await
            .unwrap();
        set_last_used(&store, &reference, old).await;
        let second = Store::reference();
        store
            .save(&second, "m", &response, json!([]))
            .await
            .unwrap();
        assert!(
            store
                .save(&reference, "m", &response, json!([]))
                .await
                .is_err()
        );
        assert_eq!(
            last_used(&store, &reference).await,
            sqlite::timestamp(old).unwrap()
        );
        assert_eq!(Store::status(&directory).await.unwrap().records, 2);
        drop(store);
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[tokio::test]
    async fn concurrent_sqlite_saves_keep_quota_consistent_after_restart() {
        let directory = std::env::temp_dir().join(format!(
            "tinyllm-sqlite-concurrent-{}",
            uuid::Uuid::new_v4()
        ));
        let store = Store::open(directory.clone(), 200, 100).await.unwrap();
        let results = futures::future::join_all((0..20).map(|_| {
            let store = store.clone();
            async move {
                store
                    .save(&Store::reference(), "m", &json!({"output":[]}), json!([]))
                    .await
            }
        }))
        .await;
        let successful = results.iter().filter(|result| result.is_ok()).count();
        assert!(successful > 0 && successful < 20);
        let before = Store::status(&directory).await.unwrap();
        assert_eq!(before.records, successful as u64);
        assert!(before.bytes <= 200);
        assert_eq!(before.bytes, store.used.lock().await.bytes);
        drop(store);
        let store = Store::open(directory.clone(), 200, 100).await.unwrap();
        assert_eq!(Store::status(&directory).await.unwrap(), before);
        assert_eq!(store.used.lock().await.bytes, before.bytes);
        drop(store);
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[tokio::test]
    async fn rejected_histories_do_not_accumulate_dead_pins() {
        let directory =
            std::env::temp_dir().join(format!("tinyllm-dead-pins-{}", uuid::Uuid::new_v4()));
        let store = Store::open_with_cleanup(
            directory.clone(),
            10000,
            1000,
            Some(StateCleanup::default()),
        )
        .await
        .unwrap();
        for round in 0..10 {
            let content: Vec<_> = (0..256).map(|index| json!({
                "type":"redacted_thinking", "data":format!("{PREFIX}{:032x}", round * 256 + index)
            })).collect();
            let request = json!({"messages":[{"role":"assistant","content":content}]});
            let error = store
                .restore_scoped(&request, "openai", "m")
                .await
                .unwrap_err();
            assert_eq!(error.status, axum::http::StatusCode::BAD_REQUEST);
            let used = store.used.lock().await;
            assert!(
                used.pins.len() <= 256,
                "rejected histories must not accumulate between requests"
            );
            assert_eq!(used.bytes, 0);
        }
        store
            .restore_scoped(&json!({"messages":[]}), "openai", "m")
            .await
            .unwrap();
        assert!(store.used.lock().await.pins.is_empty());
        drop(store);
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[tokio::test]
    async fn new_output_pins_reclaim_dead_entries_and_preserve_shared_pins() {
        let directory =
            std::env::temp_dir().join(format!("tinyllm-output-pins-{}", uuid::Uuid::new_v4()));
        let store = Store::open_with_cleanup(
            directory.clone(),
            10000,
            1000,
            Some(StateCleanup::default()),
        )
        .await
        .unwrap();
        let reference = Store::reference();
        let first = store.pin(&reference).await.unwrap().unwrap();
        let second = store.pin(&reference).await.unwrap().unwrap();
        assert!(Arc::ptr_eq(&first, &second));
        drop(first);
        for _ in 0..10 {
            drop(store.pin(&Store::reference()).await.unwrap());
            let used = store.used.lock().await;
            assert!(
                used.pins.len() <= 2,
                "completed output pins must not accumulate"
            );
            assert!(Arc::ptr_eq(
                &used.pins[Store::id(&reference).unwrap()].upgrade().unwrap(),
                &second
            ));
        }
        drop(second);
        drop(store.pin(&Store::reference()).await.unwrap());
        assert_eq!(store.used.lock().await.pins.len(), 1);
        drop(store);
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn cancellation_keeps_admitted_mutations_and_accounting_together() {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .max_blocking_threads(1)
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(async {
            let directory =
                std::env::temp_dir().join(format!("tinyllm-cancel-state-{}", uuid::Uuid::new_v4()));
            let store = Store::open_with_cleanup(
                directory.clone(),
                10000,
                1000,
                Some(StateCleanup {
                    idle_days: 1,
                    interval_seconds: 1,
                }),
            )
            .await
            .unwrap();
            let reference = Store::reference();
            let native = json!({"output":[]});
            let guard = store.used.lock().await;
            let mut queued = Box::pin(store.save(&reference, "m", &native, json!([])));
            assert!(futures::poll!(&mut queued).is_pending());
            drop(queued);
            drop(guard);
            assert_eq!(
                store
                    .used
                    .lock()
                    .await
                    .database
                    .load(Store::id(&reference).unwrap(), 1000)
                    .unwrap(),
                Loaded::Missing
            );
            for cleanup in [false, true] {
                let (release, blocked) = std::sync::mpsc::channel();
                let (started, ready) = tokio::sync::oneshot::channel();
                let blocker = tokio::task::spawn_blocking(move || {
                    started.send(()).unwrap();
                    blocked.recv().unwrap();
                });
                ready.await.unwrap();
                let mut mutation: std::pin::Pin<Box<dyn std::future::Future<Output = ()> + '_>> =
                    if cleanup {
                        Box::pin(async {
                            store
                                .cleanup_at(SystemTime::now() + Duration::from_secs(86400 * 2))
                                .await
                                .unwrap();
                        })
                    } else {
                        Box::pin(async {
                            store
                                .save(&reference, "m", &native, json!([]))
                                .await
                                .unwrap();
                        })
                    };
                assert!(futures::poll!(&mut mutation).is_pending());
                assert!(
                    store.used.try_lock().is_err(),
                    "admitted job retains the lock"
                );
                drop(mutation);
                assert!(
                    store.used.try_lock().is_err(),
                    "cancellation must not release the admitted job's lock"
                );
                release.send(()).unwrap();
                blocker.await.unwrap();
                let used = store.used.lock().await;
                assert_eq!(
                    used.database
                        .contains(Store::id(&reference).unwrap())
                        .unwrap(),
                    !cleanup
                );
                assert_eq!(used.bytes, Store::status(&directory).await.unwrap().bytes);
            }
            tokio::task::spawn_blocking(|| ()).await.unwrap();
            assert_eq!(Arc::strong_count(&store._lock), 1);
            drop(store);
            lock_directory(&directory).unwrap();
            std::fs::remove_dir_all(directory).unwrap();
        });
    }

    #[tokio::test]
    async fn restore_pins_the_entire_history_before_reading_files() {
        let directory =
            std::env::temp_dir().join(format!("tinyllm-restore-pins-{}", uuid::Uuid::new_v4()));
        let store = Store::open_with_cleanup(
            directory.clone(),
            10000,
            1000,
            Some(StateCleanup::default()),
        )
        .await
        .unwrap();
        let first = Store::reference();
        let second = Store::reference();
        let request = json!({"messages":[{"role":"assistant","content":[
            {"type":"redacted_thinking","data":first},
            {"type":"text","text":"first","tinyllm_continuation":first},
            {"type":"tool_use","id":encode_tool_id(&second, "call_second").unwrap(),"name":"lookup","input":{}}
        ]}]});
        let mut pins = Vec::new();
        let mut restore = Box::pin(store.restore_scoped_pinned(&request, "openai", "m", &mut pins));
        let _ = futures::poll!(&mut restore);
        drop(restore);
        assert_eq!(
            pins.len(),
            2,
            "even an interrupted read protects all requested records"
        );
        let used = store.used.lock().await;
        for reference in [first, second] {
            assert!(
                used.pins[Store::id(&reference).unwrap()]
                    .upgrade()
                    .is_some()
            );
        }
        drop(used);
        drop(pins);
        store
            .restore_scoped(&json!({"messages":[]}), "openai", "m")
            .await
            .unwrap();
        assert!(
            store.used.lock().await.pins.is_empty(),
            "abandoned restore pins must be reclaimed on admission"
        );
        drop(store);
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[tokio::test]
    async fn failed_save_preserves_existing_files_and_quota() {
        let directory =
            std::env::temp_dir().join(format!("tinyllm-save-error-{}", uuid::Uuid::new_v4()));
        let store = Store::open(directory.clone(), 10000, 1000).await.unwrap();
        let reference = Store::reference();
        let native = json!({"output":[]});
        store
            .save(&reference, "m", &native, json!([]))
            .await
            .unwrap();
        let before = Store::status(&directory).await.unwrap();
        assert!(
            store
                .save(&reference, "m", &native, json!([]))
                .await
                .is_err()
        );
        let reference = Store::reference();
        store.used.lock().await.database.connection.execute_batch(
            "CREATE TEMP TRIGGER fail_access BEFORE INSERT ON access BEGIN SELECT RAISE(ABORT, 'fixture'); END;"
        ).unwrap();
        assert!(
            store
                .save(&reference, "m", &native, json!([]))
                .await
                .is_err()
        );
        let used = store.used.lock().await;
        assert_eq!(
            used.database
                .load(Store::id(&reference).unwrap(), 1000)
                .unwrap(),
            Loaded::Missing
        );
        assert_eq!(used.bytes, before.bytes);
        assert_eq!(used.bytes, Store::status(&directory).await.unwrap().bytes);
        drop(used);
        drop(store);
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn cleanup_rejects_symlink_grace_marker() {
        let directory =
            std::env::temp_dir().join(format!("tinyllm-grace-link-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(directory.join("auth")).unwrap();
        std::fs::write(directory.join("auth/openai.json"), b"private").unwrap();
        std::os::unix::fs::symlink(
            directory.join("auth/openai.json"),
            directory.join(".cleanup-start"),
        )
        .unwrap();
        for policy in [None, Some(StateCleanup::default())] {
            assert!(
                Store::open_with_cleanup(directory.clone(), 10000, 1000, policy)
                    .await
                    .is_err()
            );
        }
        assert_eq!(
            std::fs::read(directory.join("auth/openai.json")).unwrap(),
            b"private"
        );
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[tokio::test]
    async fn cleanup_grace_survives_restart_and_resets_after_disable() {
        let directory =
            std::env::temp_dir().join(format!("tinyllm-grace-{}", uuid::Uuid::new_v4()));
        let policy = crate::models::config::StateCleanup {
            idle_days: 1,
            interval_seconds: 1,
        };
        let day = std::time::Duration::from_secs(86400);
        let now = SystemTime::now();
        std::fs::create_dir(&directory).unwrap();
        let name = "00000000000000000000000000000001.json";
        write_at(&directory, name, b"old", now - day * 60);
        let store = Store::open_with_cleanup(directory.clone(), 10000, 1000, Some(policy))
            .await
            .unwrap();
        assert_eq!(store.cleanup_at(now).await.unwrap().records, 0);
        drop(store);
        let store = Store::open_with_cleanup(directory.clone(), 10000, 1000, Some(policy))
            .await
            .unwrap();
        assert_eq!(store.cleanup_at(now + day * 2).await.unwrap().records, 1);
        drop(store);
        write_at(&directory, name, b"old", now - day * 60);
        let store = Store::open(directory.clone(), 10000, 1000).await.unwrap();
        assert_eq!(store.cleanup_at(now + day * 100).await.unwrap().records, 0);
        assert!(!directory.join(".cleanup-start").exists());
        drop(store);
        let store = Store::open_with_cleanup(directory.clone(), 10000, 1000, Some(policy))
            .await
            .unwrap();
        assert_eq!(store.cleanup_at(now).await.unwrap().records, 0);
        drop(store);
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[tokio::test]
    async fn cleanup_preserves_active_records_and_refreshes_only_valid_history() {
        let directory = std::env::temp_dir().join(format!("tinyllm-idle-{}", uuid::Uuid::new_v4()));
        let now = SystemTime::now();
        let day = std::time::Duration::from_secs(86400);
        let old = now - day * 60;
        std::fs::create_dir_all(directory.join("auth")).unwrap();
        write_at(&directory, ".cleanup-start", b"", old);
        for name in [
            "unrelated.json",
            "auth/openai.json",
            "00000000000000000000000000000001.tmp",
        ] {
            write_at(&directory, name, b"keep", old);
        }
        #[cfg(unix)]
        std::os::unix::fs::symlink(
            directory.join("auth/openai.json"),
            directory.join("00000000000000000000000000000002.json"),
        )
        .unwrap();
        let store = Store::open_with_cleanup(
            directory.clone(),
            10000,
            1000,
            Some(crate::models::config::StateCleanup::default()),
        )
        .await
        .unwrap();
        let reference = Store::reference();
        let content =
            json!([{"type":"redacted_thinking","data":reference},{"type":"text","text":"ok"}]);
        store
            .save(
                &reference,
                "openai/gpt-test",
                &json!({"output":[]}),
                content.clone(),
            )
            .await
            .unwrap();
        set_last_used(&store, &reference, old).await;
        let history = json!({"messages":[{"role":"assistant","content":content}]});
        let mut invalid = history.clone();
        invalid["messages"][0]["content"][1]["text"] = json!("edited");
        assert!(
            store
                .restore_scoped(&invalid, "openai", "gpt-test")
                .await
                .is_err()
        );
        assert_eq!(
            last_used(&store, &reference).await,
            sqlite::timestamp(old).unwrap()
        );
        let mut first = Vec::new();
        let mut second = Vec::new();
        store
            .restore_scoped_pinned(&history, "openai", "gpt-test", &mut first)
            .await
            .unwrap();
        store
            .restore_scoped_pinned(&history, "openai", "gpt-test", &mut second)
            .await
            .unwrap();
        assert!(last_used(&store, &reference).await >= sqlite::timestamp(now).unwrap());
        assert!(Arc::ptr_eq(&first[0], &second[0]));
        assert_eq!(store.cleanup_at(now + day).await.unwrap().records, 0);
        drop(first);
        assert_eq!(store.cleanup_at(now + day * 61).await.unwrap().records, 0);
        drop(second);
        let removed = store.cleanup_at(now + day * 61).await.unwrap();
        assert_eq!(removed.records, 1);
        assert_eq!(store.used.lock().await.bytes, 4);
        assert_eq!(Store::status(&directory).await.unwrap().bytes, 4);
        for name in [
            "unrelated.json",
            "auth/openai.json",
            "00000000000000000000000000000001.tmp",
        ] {
            assert_eq!(std::fs::read(directory.join(name)).unwrap(), b"keep");
        }
        assert!(store.used.lock().await.pins.is_empty());
        drop(store);
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[tokio::test]
    async fn capacity_cleanup_only_reclaims_expired_unpinned_records() {
        let directory =
            std::env::temp_dir().join(format!("tinyllm-quota-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir(&directory).unwrap();
        let old = SystemTime::now() - std::time::Duration::from_secs(86400 * 60);
        write_at(&directory, ".cleanup-start", b"", old);
        let reference = Store::reference();
        let filename = format!("{}.json", reference.strip_prefix(PREFIX).unwrap());
        write_at(&directory, &filename, &[b'x'; 500], old);
        let store = Store::open_with_cleanup(
            directory.clone(),
            500,
            500,
            Some(crate::models::config::StateCleanup::default()),
        )
        .await
        .unwrap();
        let pin = store.pin(&reference).await.unwrap();
        let fresh = Store::reference();
        assert!(
            store
                .save(&fresh, "m", &json!({"output":[]}), json!([]))
                .await
                .is_err()
        );
        drop(pin);
        store
            .save(&fresh, "m", &json!({"output":[]}), json!([]))
            .await
            .unwrap();
        assert!(!directory.join(filename).exists());
        assert_eq!(
            store.used.lock().await.bytes,
            Store::status(&directory).await.unwrap().bytes
        );
        let pin = store.pin(&fresh).await.unwrap();
        assert_eq!(
            store
                .cleanup_at(SystemTime::now() + std::time::Duration::from_secs(86400 * 60))
                .await
                .unwrap()
                .records,
            0
        );
        drop(pin);
        drop(store);
        std::fs::remove_dir_all(directory).unwrap();
    }

    async fn set_last_used(store: &Store, reference: &str, time: SystemTime) {
        store
            .used
            .lock()
            .await
            .database
            .connection
            .execute(
                "UPDATE access SET last_used = ?2 WHERE id = ?1",
                (
                    Store::id(reference).unwrap(),
                    sqlite::timestamp(time).unwrap(),
                ),
            )
            .unwrap();
    }

    async fn last_used(store: &Store, reference: &str) -> i64 {
        store
            .used
            .lock()
            .await
            .database
            .connection
            .query_row(
                "SELECT last_used FROM access WHERE id = ?1",
                [Store::id(reference).unwrap()],
                |row| row.get(0),
            )
            .unwrap()
    }

    fn write_at(directory: &Path, name: &str, content: &[u8], modified: SystemTime) {
        let path = directory.join(name);
        std::fs::write(&path, content).unwrap();
        std::fs::File::options()
            .write(true)
            .open(path)
            .unwrap()
            .set_times(std::fs::FileTimes::new().set_modified(modified))
            .unwrap();
    }

    #[tokio::test]
    async fn prune_previews_and_applies_only_old_regular_continuation_files() {
        let directory =
            std::env::temp_dir().join(format!("tinyllm-prune-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(directory.join("auth")).unwrap();
        let cutoff = SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(1_000_000);
        let old = cutoff - std::time::Duration::from_secs(1);
        let new = cutoff + std::time::Duration::from_secs(1);
        write_at(
            &directory,
            "00000000000000000000000000000001.json",
            b"123",
            old,
        );
        write_at(
            &directory,
            "00000000000000000000000000000002.tmp",
            b"12345",
            old,
        );
        write_at(
            &directory,
            "00000000000000000000000000000003.json",
            b"1234567",
            cutoff,
        );
        write_at(
            &directory,
            "00000000000000000000000000000004.json",
            b"12345678901",
            new,
        );
        for name in [
            "unrelated.json",
            "00000000000000000000000000000005.txt",
            "0000000000000000000000000000000z.json",
            "auth/00000000000000000000000000000006.json",
        ] {
            write_at(&directory, name, b"keep", old);
        }
        std::fs::create_dir(directory.join("00000000000000000000000000000007.json")).unwrap();
        #[cfg(unix)]
        std::os::unix::fs::symlink(
            directory.join("unrelated.json"),
            directory.join("00000000000000000000000000000008.json"),
        )
        .unwrap();
        let before = Usage {
            records: 3,
            temporary_files: 1,
            bytes: 26,
        };
        let selected = Usage {
            records: 1,
            temporary_files: 1,
            bytes: 8,
        };
        let preview = Store::prune(&directory, cutoff, false).await.unwrap();
        assert_eq!(preview.before, before);
        assert_eq!(preview.selected, selected);
        assert_eq!(preview.after, before);
        assert_eq!(Store::status(&directory).await.unwrap(), before);
        let report = Store::prune(&directory, cutoff, true).await.unwrap();
        assert_eq!(report.before, before);
        assert_eq!(report.selected, selected);
        assert_eq!(
            report.after,
            Usage {
                records: 2,
                temporary_files: 0,
                bytes: 18
            }
        );
        assert_eq!(Store::status(&directory).await.unwrap(), report.after);
        for name in [
            "00000000000000000000000000000001.json",
            "00000000000000000000000000000002.tmp",
        ] {
            assert!(!directory.join(name).exists());
        }
        for name in [
            "unrelated.json",
            "00000000000000000000000000000005.txt",
            "0000000000000000000000000000000z.json",
            "auth/00000000000000000000000000000006.json",
        ] {
            assert_eq!(std::fs::read(directory.join(name)).unwrap(), b"keep");
        }
        assert!(
            directory
                .join("00000000000000000000000000000007.json")
                .is_dir()
        );
        #[cfg(unix)]
        assert!(
            std::fs::symlink_metadata(directory.join("00000000000000000000000000000008.json"))
                .unwrap()
                .is_symlink()
        );
        let store = Store::open(directory.clone(), 100_000, 10_000)
            .await
            .unwrap();
        assert_eq!(store.used.lock().await.bytes, 18);
        assert_eq!(Store::status(&directory).await.unwrap(), report.after);
        drop(store);
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[tokio::test]
    async fn absent_state_is_not_created_by_status_or_prune() {
        let directory =
            std::env::temp_dir().join(format!("tinyllm-absent-{}", uuid::Uuid::new_v4()));
        assert_eq!(Store::status(&directory).await.unwrap(), Usage::default());
        for apply in [false, true] {
            let report = Store::prune(&directory, SystemTime::now(), apply)
                .await
                .unwrap();
            assert_eq!(report.before, Usage::default());
            assert_eq!(report.selected, Usage::default());
            assert_eq!(report.after, Usage::default());
        }
        assert!(!directory.exists());
    }

    #[tokio::test]
    async fn live_status_tolerates_temporary_file_renames() {
        let directory =
            std::env::temp_dir().join(format!("tinyllm-snapshot-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir(&directory).unwrap();
        let path = directory.join("00000000000000000000000000000001.json");
        let writer = async {
            for _ in 0..100 {
                fs::write(path.with_extension("tmp"), b"123").await.unwrap();
                fs::rename(path.with_extension("tmp"), &path).await.unwrap();
            }
        };
        let reader = async {
            for _ in 0..100 {
                Store::status(&directory).await.unwrap();
            }
        };
        tokio::join!(writer, reader);
        assert_eq!(
            Store::status(&directory).await.unwrap(),
            Usage {
                records: 1,
                temporary_files: 0,
                bytes: 3
            }
        );
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn state_directory_and_lock_symlinks_are_rejected() {
        use std::os::unix::fs::symlink;
        let directory =
            std::env::temp_dir().join(format!("tinyllm-symlinks-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(directory.join("real")).unwrap();
        symlink(directory.join("real"), directory.join("linked")).unwrap();
        assert!(Store::status(&directory.join("linked")).await.is_err());
        for apply in [false, true] {
            assert!(
                Store::prune(&directory.join("linked"), SystemTime::now(), apply)
                    .await
                    .is_err()
            );
        }
        assert!(!directory.join("real/.lock").exists());
        std::fs::write(directory.join("untouched"), b"keep").unwrap();
        symlink(directory.join("untouched"), directory.join("real/.lock")).unwrap();
        for apply in [false, true] {
            assert!(
                Store::prune(&directory.join("real"), SystemTime::now(), apply)
                    .await
                    .is_err()
            );
        }
        assert_eq!(std::fs::read(directory.join("untouched")).unwrap(), b"keep");
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn high_water_boundary_does_not_overflow() {
        for (bytes, limit, high) in [
            (79, 100, false),
            (80, 100, true),
            (101, 100, true),
            (80, 101, false),
            (81, 101, true),
            (0, u64::MAX, false),
            (u64::MAX, u64::MAX, true),
        ] {
            assert_eq!(
                Usage {
                    bytes,
                    ..Usage::default()
                }
                .is_high(limit),
                high
            );
        }
    }

    #[tokio::test]
    async fn acknowledgement_exception_requires_a_prior_user_marker_and_exact_content() {
        let directory =
            std::env::temp_dir().join(format!("tinyllm-marker-{}", uuid::Uuid::new_v4()));
        let store = Store::open(directory.clone(), 100_000, 10_000)
            .await
            .unwrap();
        let ack = json!({"role":"assistant", "content":[{"type":"text", "text":"No response requested.", "cache_control":{"type":"ephemeral"}}]});
        for (content, valid) in [
            (json!("<local-command-stdout>ready"), true),
            (json!([{"text":"<local-command-stdout>ready"}]), true),
            (
                json!([{"type":"image", "text":"<local-command-stdout>ready"}]),
                true,
            ),
            (json!({"text":"<local-command-stdout>ready"}), false),
            (json!("prefix <local-command-stdout>ready"), false),
            (json!(42), false),
        ] {
            let request = json!({"messages":[{"role":"user", "content":content}, {"role":"system", "content":"reminder"}, {"role":"user", "content":"ordinary"}, ack, {"role":"user", "content":"continue"}, ack]});
            assert_eq!(
                store
                    .restore_scoped(&request, "openai", "gpt-test")
                    .await
                    .is_ok(),
                valid
            );
        }
        let marker = json!({"role":"user", "content":"<local-command-stdout>ready"});
        for messages in [
            json!([ack, marker]),
            json!([{"role":"system", "content":"<local-command-stdout>ready"}, ack]),
            json!([marker, ack, ack]),
            json!([marker, {"role":"assistant", "content":"Different response"}]),
            json!([marker, {"role":"assistant", "content":[{"type":"text", "text":"No response requested.", "extra":true}]}]),
            json!([marker, {"role":"assistant", "content":[{"type":"redacted_thinking", "data":Store::reference()}, {"type":"text", "text":"No response requested."}]}]),
        ] {
            assert!(
                store
                    .restore_scoped(&json!({"messages":messages}), "openai", "gpt-test")
                    .await
                    .is_err()
            );
        }
        drop(store);
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[tokio::test]
    async fn startup_usage_counts_only_continuation_files() {
        let directory =
            std::env::temp_dir().join(format!("tinyllm-usage-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(directory.join("auth")).unwrap();
        std::fs::write(
            directory.join("00000000000000000000000000000001.json"),
            b"123",
        )
        .unwrap();
        std::fs::write(
            directory.join("00000000000000000000000000000002.tmp"),
            b"12345",
        )
        .unwrap();
        std::fs::write(directory.join("unrelated.json"), b"1234567890").unwrap();
        std::fs::write(
            directory.join("auth/00000000000000000000000000000003.json"),
            b"private",
        )
        .unwrap();
        let store = Store::open(directory.clone(), 100_000, 10_000)
            .await
            .unwrap();
        assert_eq!(store.used.lock().await.bytes, 8);
        drop(store);
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[tokio::test]
    async fn long_local_acknowledgement_history_stays_within_cpu_budget() {
        let directory =
            std::env::temp_dir().join(format!("tinyllm-linear-{}", uuid::Uuid::new_v4()));
        let store = Store::open(directory.clone(), 100_000, 10_000)
            .await
            .unwrap();
        let mut messages = vec![json!({"role":"user", "content":"<local-command-stdout>ready"})];
        for _ in 0..4_000 {
            messages.push(json!({"role":"user", "content":"ordinary"}));
            messages.push(json!({"role":"assistant", "content":"No response requested."}));
        }
        let request = json!({"messages":messages});
        let start = std::time::Instant::now();
        assert!(
            store
                .restore_scoped(&request, "openai", "gpt-test")
                .await
                .unwrap()
                .is_empty()
        );
        let elapsed = start.elapsed();
        drop(store);
        std::fs::remove_dir_all(directory).unwrap();
        assert!(
            elapsed < std::time::Duration::from_secs(2),
            "history restore took {elapsed:?}"
        );
    }

    #[tokio::test]
    async fn continuation_is_scoped_to_provider_and_legacy_openai_is_readable() {
        let directory =
            std::env::temp_dir().join(format!("tinyllm-scope-{}", uuid::Uuid::new_v4()));
        let store = Store::open(directory.clone(), 100_000, 10_000)
            .await
            .unwrap();
        let reference = Store::reference();
        let content =
            json!([{"type":"redacted_thinking","data":reference},{"type":"text","text":"same"}]);
        let native = json!({"output":[{"type":"reasoning","encrypted_content":"opaque"}]});
        store
            .save_scoped(
                &reference,
                "openai",
                "gpt-test",
                &native,
                content.clone(),
                &BTreeMap::new(),
            )
            .await
            .unwrap();
        let request = json!({"messages":[{"role":"assistant","content":content}]});
        assert!(
            store
                .restore_scoped(&request, "openai", "gpt-test")
                .await
                .is_ok()
        );
        assert!(
            store
                .restore_scoped(&request, "other", "gpt-test")
                .await
                .is_err()
        );
        let legacy = Store::reference();
        let content = json!([{"type":"redacted_thinking","data":legacy}]);
        store
            .save(&legacy, "gpt-test", &native, content.clone())
            .await
            .unwrap();
        let request = json!({"messages":[{"role":"assistant","content":content}]});
        assert!(
            store
                .restore_scoped(&request, "openai", "gpt-test")
                .await
                .is_ok()
        );
        assert!(
            store
                .restore_scoped(&request, "other", "gpt-test")
                .await
                .is_err()
        );
        drop(store);
        let report = Store::prune(
            &directory,
            SystemTime::now() + std::time::Duration::from_secs(1),
            true,
        )
        .await
        .unwrap();
        assert_eq!(report.selected.records, 2);
        assert_eq!(report.after, Usage::default());
        let store = Store::open(directory.clone(), 100_000, 10_000)
            .await
            .unwrap();
        assert!(
            store
                .restore_scoped(&request, "openai", "gpt-test")
                .await
                .unwrap_err()
                .message
                .contains("state is missing")
        );
        drop(store);
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[tokio::test]
    async fn repeated_calls_do_not_multiply_default_storage() {
        let directory =
            std::env::temp_dir().join(format!("tinyllm-default-size-{}", uuid::Uuid::new_v4()));
        let store = Store::open(directory.clone(), 100_000, 40_000)
            .await
            .unwrap();
        let reference = Store::reference();
        let mut content = vec![json!({"type":"redacted_thinking","data":reference})];
        content.extend((0..100).map(
            |id| json!({"type":"tool_use","id":format!("call_{id}"),"name":"tool","input":{}}),
        ));
        let defaults = tool_defaults(
            &json!([{"name":"tool","input_schema":{"properties":{"omitted":{"default":"x".repeat(1024)}}}}]),
        );
        store
            .save_scoped(
                &reference,
                "openai",
                "gpt-test",
                &json!({"output":[]}),
                json!(content),
                &defaults,
            )
            .await
            .unwrap();
        drop(store);
        std::fs::remove_dir_all(directory).unwrap();
    }
}
