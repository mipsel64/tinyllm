use crate::{Result, error::Error, models::config::StateCleanup, providers::openai::protocol};
use eyre::WrapErr;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::{BTreeMap, HashSet},
    path::{Path, PathBuf},
    sync::{Arc, Weak},
    time::{Duration, SystemTime},
};
use tokio::{fs, sync::Mutex};

pub const PREFIX: &str = "tinyllm:v1:";

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

#[derive(Default)]
struct Bookkeeping {
    bytes: u64,
    pins: BTreeMap<PathBuf, Weak<()>>,
}

#[derive(Clone)]
pub struct Store {
    directory: PathBuf,
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
        let lock = lock_directory(&directory)?;
        let cleanup = cleanup_grace(&directory, retention)?;
        let usage = Self::status(&directory).await?;
        tracing::info!(
            state_records = usage.records,
            state_temporary_files = usage.temporary_files,
            state_bytes = usage.bytes,
            max_state_bytes = limit,
            "continuation state usage"
        );
        warn_if_high(usage, limit);
        Ok(Self {
            directory,
            _lock: Arc::new(lock),
            used: Arc::new(Mutex::new(Bookkeeping {
                bytes: usage.bytes,
                ..Default::default()
            })),
            limit,
            record_limit,
            cleanup,
        })
    }

    pub(crate) async fn cleanup(&self) -> eyre::Result<Usage> {
        self.cleanup_at(SystemTime::now()).await
    }

    async fn cleanup_at(&self, now: SystemTime) -> eyre::Result<Usage> {
        let mut used = self.used.clone().lock_owned().await;
        let store = self.clone();
        tokio::task::spawn_blocking(move || store.cleanup_locked(&mut used, now)).await?
    }

    fn cleanup_locked(&self, used: &mut Bookkeeping, now: SystemTime) -> eyre::Result<Usage> {
        let mut removed = Usage::default();
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
        // ponytail: scan under the write lock; index records if directory scans become costly.
        let result = (|| -> eyre::Result<()> {
            for entry in std::fs::read_dir(&self.directory)? {
                let entry = entry?;
                let name = entry.file_name();
                if !matches!(record_name(&name), Some((_, false))) {
                    continue;
                }
                let path = entry.path();
                if used.pins.contains_key(&path) {
                    continue;
                }
                let metadata = match std::fs::symlink_metadata(&path) {
                    Ok(metadata) if metadata.is_file() => metadata,
                    Ok(_) => continue,
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
                    Err(error) => return Err(error.into()),
                };
                if metadata.modified()? >= cutoff {
                    continue;
                }
                std::fs::remove_file(&path)?;
                used.bytes = used.bytes.saturating_sub(metadata.len());
                removed.add(metadata.len(), false)?;
            }
            Ok(())
        })();
        if removed.records > 0 {
            tracing::info!(
                removed_records = removed.records,
                removed_bytes = removed.bytes,
                state_bytes = used.bytes,
                "expired continuation state removed"
            );
        }
        result.wrap_err_with(|| "cannot clean expired continuation state")?;
        Ok(removed)
    }

    pub async fn pin(&self, reference: &str) -> Result<Option<Arc<()>>> {
        let path = self.path(reference)?;
        if self.cleanup.is_none() {
            return Ok(None);
        }
        let mut used = self.used.lock().await;
        used.pins.retain(|_, pin| pin.strong_count() > 0);
        Ok(Some(Self::pin_locked(&mut used, path)))
    }

    fn pin_locked(used: &mut Bookkeeping, path: PathBuf) -> Arc<()> {
        let entry = used.pins.entry(path).or_default();
        let pin = entry.upgrade().unwrap_or_else(|| Arc::new(()));
        *entry = Arc::downgrade(&pin);
        pin
    }

    pub async fn status(directory: &Path) -> eyre::Result<Usage> {
        Ok(scan(directory, None, false).await?.before)
    }

    pub async fn prune(
        directory: &Path,
        cutoff: SystemTime,
        apply: bool,
    ) -> eyre::Result<PruneReport> {
        if !directory_exists(directory).await? {
            return Ok(PruneReport::default());
        }
        let _lock = lock_directory(directory)?;
        scan(directory, Some(cutoff), apply).await
    }

    pub fn reference() -> String {
        format!("{PREFIX}{}", uuid::Uuid::new_v4().simple())
    }

    fn path(&self, reference: &str) -> Result<PathBuf> {
        let id = reference
            .strip_prefix(PREFIX)
            .filter(|id| id.len() == 32 && id.bytes().all(|c| c.is_ascii_hexdigit()))
            .ok_or_else(|| Error::invalid("invalid tinyllm continuation reference"))?;
        Ok(self.directory.join(format!("{id}.json")))
    }

    pub async fn save(
        &self,
        reference: &str,
        model: &str,
        response: &Value,
        content: Value,
    ) -> Result<()> {
        self.save_with_defaults(reference, model, response, content, &BTreeMap::new())
            .await
    }

    async fn save_with_defaults(
        &self,
        reference: &str,
        model: &str,
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
        let path = self.path(reference)?;
        let mut used = self.used.clone().lock_owned().await;
        let store = self.clone();
        tokio::task::spawn_blocking(move || {
            if used.bytes.checked_add(data.len() as u64).is_none_or(|n| n > store.limit) {
                store.cleanup_locked(&mut used, SystemTime::now())
                    .map_err(|_| Error::upstream("cannot clean expired continuation state"))?;
            }
            let next = used.bytes.checked_add(data.len() as u64)
                .filter(|n| *n <= store.limit)
                .ok_or_else(|| Error::upstream("continuation store is full; increase max_state_bytes or stop the gateway and run tinyllm state prune"))?;
            let was_high = Usage { bytes: used.bytes, ..Default::default() }.is_high(store.limit);
            let temporary = path.with_extension("tmp");
            if path.try_exists().map_err(|_| Error::upstream("cannot inspect continuation state"))? {
                return Err(Error::upstream("continuation reference already exists"));
            }
            let mut options = std::fs::OpenOptions::new();
            options.write(true).create_new(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                options.mode(0o600);
            }
            let mut file = options.open(&temporary)
                .map_err(|_| Error::upstream("cannot create continuation state"))?;
            use std::io::Write;
            let write = file.write_all(&data)
                .and_then(|_| file.sync_all())
                .and_then(|_| std::fs::rename(&temporary, &path));
            if write.is_err() {
                if std::fs::remove_file(&temporary).is_err() {
                    used.bytes = used.bytes.saturating_add(file.metadata().map_or(data.len() as u64, |m| m.len()));
                }
                return Err(Error::upstream("cannot persist continuation state"));
            }
            used.bytes = next;
            if !was_high {
                warn_if_high(Usage { bytes: used.bytes, ..Default::default() }, store.limit);
            }
            Ok(())
        }).await.map_err(|_| Error::upstream("continuation write failed"))?
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
        self.restore_for(
            request,
            &format!("{provider}/{model}"),
            (provider == "openai").then_some(model),
            pins,
        )
        .await
    }

    #[cfg(test)]
    pub async fn restore(
        &self,
        request: &Value,
        model: &str,
    ) -> Result<BTreeMap<usize, Vec<Value>>> {
        self.restore_for(request, model, None, &mut Vec::new())
            .await
    }

    async fn restore_for(
        &self,
        request: &Value,
        model: &str,
        legacy_model: Option<&str>,
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
            for block in messages
                .iter()
                .filter(|message| message["role"] == "assistant")
                .filter_map(|message| message["content"].as_array())
                .flatten()
                .filter(|block| block["type"] == "redacted_thinking")
            {
                let path = self.path(protocol::string(block, "data")?)?;
                pins.push(Self::pin_locked(&mut used, path));
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
                let path = self.path(reference)?;
                if !fs::symlink_metadata(&path).await.is_ok_and(|m| m.is_file()) {
                    return Err(Error::invalid(
                        "continuation state is missing or not a regular file; restore the state directory or start a new conversation",
                    ));
                }
                let file = fs::File::open(&path).await.map_err(|_| Error::invalid("continuation state is missing; restore the state directory or start a new conversation"))?;
                use tokio::io::AsyncReadExt;
                let mut data = Vec::new();
                file.take(self.record_limit as u64 + 1)
                    .read_to_end(&mut data)
                    .await
                    .map_err(|_| Error::upstream("cannot read continuation state"))?;
                if data.len() > self.record_limit {
                    return Err(Error::upstream(
                        "continuation record exceeds configured limit",
                    ));
                }
                restored_bytes = restored_bytes.checked_add(data.len()).filter(|n| *n <= self.record_limit).ok_or_else(|| Error::invalid("restored continuation history exceeds max_response_bytes; compact the conversation"))?;
                let record: Record = serde_json::from_slice(&data)
                    .map_err(|_| Error::upstream("corrupt continuation state"))?;
                let end = content[start + 1..]
                    .iter()
                    .position(|b| b["type"] == "redacted_thinking")
                    .map_or(content.len(), |p| start + 1 + p);
                if record.model != model && legacy_model != Some(record.model.as_str()) {
                    return Err(Error::invalid(
                        "continuation model changed; keep its model mapping or start a new conversation",
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
            let paths: Vec<_> = references
                .iter()
                .map(|r| self.path(r))
                .collect::<Result<_>>()?;
            let used = self.used.clone().lock_owned().await;
            let store = self.clone();
            tokio::task::spawn_blocking(move || -> std::io::Result<()> {
                let (_used, _store) = (used, store);
                let now = SystemTime::now();
                for path in paths {
                    let file = std::fs::File::open(path)?;
                    file.set_modified(now.max(file.metadata()?.modified()?))?;
                }
                Ok(())
            })
            .await
            .map_err(|_| Error::upstream("continuation timestamp update failed"))?
            .map_err(|_| Error::upstream("cannot update continuation timestamp"))?;
        }
        Ok(restored)
    }
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

async fn scan(
    directory: &Path,
    cutoff: Option<SystemTime>,
    apply: bool,
) -> eyre::Result<PruneReport> {
    let mut report = PruneReport::default();
    if !directory_exists(directory).await? {
        return Ok(report);
    }
    let mut entries = fs::read_dir(directory)
        .await
        .wrap_err_with(|| format!("cannot read state directory {}", directory.display()))?;
    while let Some(entry) = entries
        .next_entry()
        .await
        .wrap_err_with(|| format!("cannot read state directory {}", directory.display()))?
    {
        let name = entry.file_name();
        let Some((_, temporary)) = record_name(&name) else {
            continue;
        };
        let path = entry.path();
        let metadata = match fs::symlink_metadata(&path).await {
            Ok(metadata) if metadata.is_file() => metadata,
            Ok(_) => continue,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => {
                return Err(error)
                    .wrap_err_with(|| format!("cannot inspect state file {}", path.display()));
            }
        };
        report.before.add(metadata.len(), temporary)?;
        let selected = match cutoff {
            Some(cutoff) => {
                metadata.modified().wrap_err_with(|| {
                    format!("cannot read modification time for {}", path.display())
                })? < cutoff
            }
            None => false,
        };
        if selected {
            report.selected.add(metadata.len(), temporary)?;
            if apply {
                fs::remove_file(&path)
                    .await
                    .wrap_err_with(|| format!("cannot remove state file {}", path.display()))?;
                continue;
            }
        }
        report.after.add(metadata.len(), temporary)?;
    }
    Ok(report)
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
                &used.pins[&store.path(&reference).unwrap()]
                    .upgrade()
                    .unwrap(),
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
            assert!(!store.path(&reference).unwrap().exists());
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
                assert_eq!(store.path(&reference).unwrap().exists(), !cleanup);
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
            {"type":"redacted_thinking","data":second}
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
                used.pins[&store.path(&reference).unwrap()]
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
        let temporary = store.path(&reference).unwrap().with_extension("tmp");
        std::fs::write(&temporary, b"unfinished").unwrap();
        store.used.lock().await.bytes += 10;
        assert!(
            store
                .save(&reference, "m", &native, json!([]))
                .await
                .is_err()
        );
        assert_eq!(std::fs::read(temporary).unwrap(), b"unfinished");
        assert_eq!(store.used.lock().await.bytes, before.bytes + 10);
        assert_eq!(
            store.used.lock().await.bytes,
            Store::status(&directory).await.unwrap().bytes
        );
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
        let path = store.path(&reference).unwrap();
        std::fs::File::open(&path)
            .unwrap()
            .set_modified(old)
            .unwrap();
        let history = json!({"messages":[{"role":"assistant","content":content}]});
        let mut invalid = history.clone();
        invalid["messages"][0]["content"][1]["text"] = json!("edited");
        assert!(
            store
                .restore_scoped(&invalid, "openai", "gpt-test")
                .await
                .is_err()
        );
        assert_eq!(std::fs::metadata(&path).unwrap().modified().unwrap(), old);
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
        assert!(std::fs::metadata(&path).unwrap().modified().unwrap() >= now);
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

    #[tokio::test]
    async fn prune_requires_gateway_process_lock() {
        const CHILD_DIRECTORY: &str = "TINYLLM_TEST_LOCKED_STATE";
        if let Some(directory) = std::env::var_os(CHILD_DIRECTORY) {
            for apply in [false, true] {
                let error = Store::prune(Path::new(&directory), SystemTime::now(), apply)
                    .await
                    .unwrap_err();
                assert!(error.to_string().contains("already in use"));
            }
            return;
        }
        let directory = std::env::temp_dir().join(format!("tinyllm-lock-{}", uuid::Uuid::new_v4()));
        let store = Store::open(directory.clone(), 100_000, 10_000)
            .await
            .unwrap();
        let result = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "providers::openai::state::tests::prune_requires_gateway_process_lock",
            ])
            .env(CHILD_DIRECTORY, &directory)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}{}",
            String::from_utf8_lossy(&result.stdout),
            String::from_utf8_lossy(&result.stderr)
        );
        drop(store);
        Store::prune(&directory, SystemTime::now(), false)
            .await
            .unwrap();
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
