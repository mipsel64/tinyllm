use super::{Usage, record_name};
use eyre::WrapErr;
use rusqlite::{Connection, OpenFlags, OptionalExtension, Transaction};
use std::{
    collections::{BTreeMap, HashSet},
    io::{Read, Write},
    path::Path,
    sync::Weak,
    time::SystemTime,
};

pub const FILE: &str = "state.sqlite";
const TOUCH_SECONDS: i64 = 300;

#[derive(Debug, PartialEq)]
pub(super) enum Loaded {
    Missing,
    TooLarge,
    Data(Vec<u8>),
}

pub(super) struct Database {
    pub connection: Connection,
}

impl Database {
    pub fn open(directory: &Path, writable: bool) -> eyre::Result<Self> {
        for name in [
            FILE,
            "state.sqlite-journal",
            "state.sqlite-wal",
            "state.sqlite-shm",
        ] {
            match std::fs::symlink_metadata(directory.join(name)) {
                Ok(metadata) if metadata.is_file() => {}
                Ok(_) => eyre::bail!("state database files must be regular files"),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => return Err(error.into()),
            }
        }
        let path = directory.join(FILE);
        if writable && !path.exists() {
            let mut options = std::fs::OpenOptions::new();
            options.write(true).create_new(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                options.mode(0o600);
            }
            options.open(&path)?;
        }
        #[cfg(unix)]
        if writable {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600))?;
        }
        let flags = if writable {
            OpenFlags::SQLITE_OPEN_READ_WRITE
        } else {
            OpenFlags::SQLITE_OPEN_READ_ONLY
        };
        let mut connection = Connection::open_with_flags(path, flags)?;
        connection.execute_batch("PRAGMA foreign_keys = ON; PRAGMA trusted_schema = OFF;")?;
        let version: i64 = connection.query_row("PRAGMA user_version", [], |row| row.get(0))
            .wrap_err_with(|| "cannot read continuation database; after an interrupted write, start the gateway to recover it")?;
        eyre::ensure!(
            matches!(version, 0 | 1),
            "unsupported continuation database version {version}"
        );
        if writable {
            // EXTRA makes journal retirement durable before migrated source files are removed.
            connection
                .execute_batch("PRAGMA journal_mode = DELETE; PRAGMA synchronous = EXTRA;")?;
        }
        match version {
            0 if writable => {
                let transaction = connection.transaction()?;
                transaction.execute_batch(
                    "CREATE TABLE records (id TEXT PRIMARY KEY, data BLOB NOT NULL);
                     CREATE TABLE access (id TEXT PRIMARY KEY REFERENCES records(id) ON DELETE CASCADE,
                         last_used INTEGER NOT NULL, bytes INTEGER NOT NULL CHECK(bytes >= 0));
                     CREATE INDEX access_lru ON access(last_used, id);
                     PRAGMA user_version = 1;"
                )?;
                transaction.commit()?;
            }
            1 => {}
            _ => eyre::bail!(
                "continuation database is uninitialized; start the gateway to initialize it"
            ),
        }
        Ok(Self { connection })
    }

    pub fn migrate(&mut self, directory: &Path) -> eyre::Result<()> {
        let mut total = 0;
        loop {
            let transaction = self.connection.transaction()?;
            let mut imported = Vec::new();
            for entry in std::fs::read_dir(directory)? {
                let entry = entry?;
                let name = entry.file_name();
                let Some((id, false)) = record_name(&name) else {
                    continue;
                };
                let path = entry.path();
                if !std::fs::symlink_metadata(&path)?.is_file() {
                    continue;
                }
                let mut file = std::fs::File::open(&path)?;
                let metadata = file.metadata()?;
                let size = i32::try_from(metadata.len()).wrap_err_with(
                    || "legacy continuation exceeds SQLite's blob size limit; source retained",
                )?;
                let existing: Option<i64> = transaction
                    .query_row("SELECT rowid FROM records WHERE id = ?1", [id], |row| {
                        row.get(0)
                    })
                    .optional()?;
                let rowid = if let Some(rowid) = existing {
                    rowid
                } else {
                    transaction.execute(
                        "INSERT INTO records(id, data) VALUES (?1, ?2)",
                        (id, rusqlite::blob::ZeroBlob(size)),
                    )?;
                    transaction.last_insert_rowid()
                };
                let mut blob = transaction.blob_open(
                    rusqlite::MAIN_DB,
                    "records",
                    "data",
                    rowid,
                    existing.is_some(),
                )?;
                eyre::ensure!(
                    blob.size() == size,
                    "legacy continuation conflicts with the database; source files retained"
                );
                // Import independently of current quotas, without buffering whole legacy records.
                let mut buffer = [0; 8192];
                let mut comparison = [0; 8192];
                let mut remaining = size as usize;
                while remaining > 0 {
                    let length = remaining.min(buffer.len());
                    file.read_exact(&mut buffer[..length])?;
                    if existing.is_some() {
                        blob.read_exact(&mut comparison[..length])?;
                        eyre::ensure!(
                            buffer[..length] == comparison[..length],
                            "legacy continuation conflicts with the database; source files retained"
                        );
                    } else {
                        blob.write_all(&buffer[..length])?;
                    }
                    remaining -= length;
                }
                eyre::ensure!(
                    file.read(&mut buffer[..1])? == 0,
                    "legacy continuation changed during migration; source retained"
                );
                blob.close()?;
                if existing.is_none() {
                    transaction.execute(
                        "INSERT INTO access(id, last_used, bytes) VALUES (?1, ?2, ?3)",
                        (id, timestamp(metadata.modified()?)?, size),
                    )?;
                }
                imported.push(path);
                if imported.len() == 256 {
                    break;
                }
            }
            transaction.commit()?;
            // A restart can repeat committed imports before retiring their source files.
            for path in &imported {
                std::fs::remove_file(path)
                    .wrap_err_with(|| "cannot retire migrated continuation file")?;
            }
            if imported.is_empty() {
                break;
            }
            total += imported.len();
        }
        if total > 0 {
            tracing::info!(records = total, "migrated continuation files to SQLite");
        }
        Ok(())
    }

    pub fn usage(&self, cutoff: Option<SystemTime>) -> eyre::Result<Usage> {
        let cutoff = cutoff.map(timestamp).transpose()?;
        let (records, bytes): (i64, i64) = self.connection.query_row(
            "SELECT count(*), coalesce(sum(bytes), 0) FROM access WHERE ?1 IS NULL OR last_used < ?1",
            [cutoff], |row| Ok((row.get(0)?, row.get(1)?))
        )?;
        Ok(Usage {
            records: records.try_into()?,
            bytes: bytes.try_into()?,
            temporary_files: 0,
        })
    }

    pub fn save(&mut self, id: &str, data: &[u8], now: SystemTime) -> eyre::Result<()> {
        let transaction = self.connection.transaction()?;
        insert(&transaction, id, data, timestamp(now)?)?;
        transaction.commit()?;
        Ok(())
    }

    pub fn contains(&self, id: &str) -> eyre::Result<bool> {
        Ok(self.connection.query_row(
            "SELECT EXISTS(SELECT 1 FROM records WHERE id = ?1)",
            [id],
            |row| row.get(0),
        )?)
    }

    pub fn load(&self, id: &str, limit: usize) -> eyre::Result<Loaded> {
        let row: Option<(i64, Option<Vec<u8>>)> = self.connection.query_row(
            "SELECT length(data), CASE WHEN length(data) <= ?2 THEN data END FROM records WHERE id = ?1",
            (id, i64::try_from(limit).unwrap_or(i64::MAX)), |row| Ok((row.get(0)?, row.get(1)?))
        ).optional()?;
        let Some((bytes, data)) = row else {
            return Ok(Loaded::Missing);
        };
        if bytes > i64::try_from(limit).unwrap_or(i64::MAX) {
            return Ok(Loaded::TooLarge);
        }
        Ok(Loaded::Data(data.ok_or_else(|| {
            eyre::eyre!("corrupt continuation state")
        })?))
    }

    pub fn touch(&mut self, ids: &HashSet<String>, now: SystemTime) -> eyre::Result<()> {
        let now = timestamp(now)?;
        let rounded = now
            .checked_add(TOUCH_SECONDS - 1)
            .ok_or_else(|| eyre::eyre!("state timestamp overflow"))?
            / TOUCH_SECONDS
            * TOUCH_SECONDS;
        let transaction = self.connection.transaction()?;
        {
            let mut update = transaction
                .prepare("UPDATE access SET last_used = ?2 WHERE id = ?1 AND last_used < ?3")?;
            for id in ids {
                update.execute((id, rounded, now))?;
            }
        }
        transaction.commit()?;
        Ok(())
    }

    pub fn prune(
        &mut self,
        cutoff: SystemTime,
        pins: &BTreeMap<String, Weak<()>>,
        needed: Option<u64>,
    ) -> eyre::Result<Usage> {
        let mut removed = Usage::default();
        let transaction = self.connection.transaction()?;
        {
            let mut select = transaction.prepare("SELECT id, bytes, last_used FROM access WHERE last_used < ?1 AND (last_used, id) > (?2, ?3) ORDER BY last_used, id LIMIT 256")?;
            let mut delete = transaction.prepare("DELETE FROM records WHERE id = ?1")?;
            let mut cursor = (i64::MIN, String::new());
            loop {
                let rows = select
                    .query_map((timestamp(cutoff)?, cursor.0, &cursor.1), |row| {
                        Ok((
                            row.get::<_, String>(0)?,
                            row.get::<_, i64>(1)?,
                            row.get::<_, i64>(2)?,
                        ))
                    })?
                    .collect::<rusqlite::Result<Vec<_>>>()?;
                if rows.is_empty() {
                    break;
                }
                for (id, bytes, last_used) in rows {
                    cursor = (last_used, id.clone());
                    if needed.is_some_and(|needed| removed.bytes >= needed) {
                        break;
                    }
                    if pins.get(&id).is_some_and(|pin| pin.strong_count() > 0) {
                        continue;
                    }
                    removed.add(bytes.try_into()?, false)?;
                    delete.execute([id])?;
                }
                if needed.is_some_and(|needed| removed.bytes >= needed) {
                    break;
                }
            }
        }
        transaction.commit()?;
        Ok(removed)
    }
}

fn insert(
    transaction: &Transaction<'_>,
    id: &str,
    data: &[u8],
    last_used: i64,
) -> eyre::Result<()> {
    transaction.execute("INSERT INTO records(id, data) VALUES (?1, ?2)", (id, data))?;
    transaction.execute(
        "INSERT INTO access(id, last_used, bytes) VALUES (?1, ?2, ?3)",
        (id, last_used, data.len() as i64),
    )?;
    Ok(())
}

pub(super) fn timestamp(time: SystemTime) -> eyre::Result<i64> {
    Ok(i64::try_from(
        time.duration_since(SystemTime::UNIX_EPOCH)?.as_secs(),
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn empty_database_status_explains_recovery_without_writing() {
        let directory =
            std::env::temp_dir().join(format!("tinyllm-sqlite-empty-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir(&directory).unwrap();
        std::fs::write(directory.join(FILE), []).unwrap();
        let error = Database::open(&directory, false).err().unwrap().to_string();
        assert!(error.contains("start the gateway"), "{error}");
        assert_eq!(std::fs::metadata(directory.join(FILE)).unwrap().len(), 0);
        let database = Database::open(&directory, true).unwrap();
        assert_eq!(
            database
                .connection
                .query_row("PRAGMA synchronous", [], |row| row.get::<_, i64>(0))
                .unwrap(),
            3
        );
        drop(database);
        assert_eq!(
            Database::open(&directory, false)
                .unwrap()
                .usage(None)
                .unwrap()
                .records,
            0
        );
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn replay_and_idle_cleanup_do_not_churn_the_database() {
        let directory =
            std::env::temp_dir().join(format!("tinyllm-sqlite-io-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir(&directory).unwrap();
        let mut database = Database::open(&directory, true).unwrap();
        let now = SystemTime::UNIX_EPOCH + Duration::from_secs(1_000_201);
        let payload = vec![b'x'; 512 * 1024];
        let ids = HashSet::from(["fixture".to_owned()]);
        database
            .save("fixture", &payload, now - Duration::from_secs(86400))
            .unwrap();
        database.touch(&ids, now).unwrap();
        let before = std::fs::read(directory.join(FILE)).unwrap();
        let changes = database.connection.total_changes();
        for second in 1..100 {
            database
                .touch(&ids, now + Duration::from_secs(second))
                .unwrap();
            assert_eq!(
                database
                    .prune(now - Duration::from_secs(86400), &BTreeMap::new(), None)
                    .unwrap()
                    .records,
                0
            );
            assert_eq!(database.usage(None).unwrap().records, 1);
            assert_eq!(
                database.load("fixture", payload.len()).unwrap(),
                Loaded::Data(payload.clone())
            );
        }
        assert_eq!(database.connection.total_changes(), changes);
        assert_eq!(std::fs::read(directory.join(FILE)).unwrap(), before);
        assert!(!directory.join("state.sqlite-wal").exists());
        assert!(!directory.join("state.sqlite-journal").exists());
        database
            .touch(&ids, now + Duration::from_secs(301))
            .unwrap();
        assert_eq!(database.connection.total_changes(), changes + 1);
        assert_eq!(
            database.load("fixture", payload.len()).unwrap(),
            Loaded::Data(payload.clone())
        );
        assert_eq!(
            database.load("fixture", payload.len() - 1).unwrap(),
            Loaded::TooLarge
        );
        drop(database);
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn lru_prune_is_bounded_and_skips_active_records() {
        let directory =
            std::env::temp_dir().join(format!("tinyllm-sqlite-lru-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir(&directory).unwrap();
        let mut database = Database::open(&directory, true).unwrap();
        let now = SystemTime::UNIX_EPOCH + Duration::from_secs(1_000_000);
        let transaction = database.connection.transaction().unwrap();
        for index in 0..600 {
            insert(
                &transaction,
                &format!("{index:032x}"),
                b"fixture",
                index + 1,
            )
            .unwrap();
        }
        transaction.commit().unwrap();
        let pin = std::sync::Arc::new(());
        let pins = BTreeMap::from([(format!("{:032x}", 0), std::sync::Arc::downgrade(&pin))]);
        assert_eq!(database.prune(now, &pins, Some(7)).unwrap().records, 1);
        assert!(!database.contains(&format!("{:032x}", 1)).unwrap());
        assert!(database.contains(&format!("{:032x}", 2)).unwrap());
        assert_eq!(database.prune(now, &pins, None).unwrap().records, 598);
        assert_eq!(database.usage(None).unwrap().records, 1);
        drop(pin);
        assert_eq!(database.prune(now, &pins, None).unwrap().records, 1);
        assert_eq!(database.usage(None).unwrap(), Usage::default());
        drop(database);
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn migration_resumes_and_rolls_back_without_deleting_sources() {
        let directory =
            std::env::temp_dir().join(format!("tinyllm-sqlite-recovery-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir(&directory).unwrap();
        let mut database = Database::open(&directory, true).unwrap();
        let id = "00000000000000000000000000000001";
        let path = directory.join(format!("{id}.json"));
        std::fs::write(&path, b"original").unwrap();
        database.connection.execute_batch("CREATE TEMP TRIGGER fail_import BEFORE INSERT ON access BEGIN SELECT RAISE(ABORT, 'fixture'); END;").unwrap();
        assert!(database.migrate(&directory).is_err());
        assert!(!database.contains(id).unwrap());
        assert_eq!(std::fs::read(&path).unwrap(), b"original");
        database
            .connection
            .execute_batch("DROP TRIGGER fail_import;")
            .unwrap();
        database.save(id, b"different", SystemTime::now()).unwrap();
        assert!(database.migrate(&directory).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), b"original");
        database
            .connection
            .execute("DELETE FROM records", [])
            .unwrap();
        database.save(id, b"original", SystemTime::now()).unwrap();
        drop(database);
        let mut database = Database::open(&directory, true).unwrap();
        database.migrate(&directory).unwrap();
        assert!(!path.exists());
        assert_eq!(
            database.load(id, 100).unwrap(),
            Loaded::Data(b"original".to_vec())
        );
        assert_eq!(database.usage(None).unwrap().records, 1);
        std::fs::write(&path, [b'x'; 101]).unwrap();
        assert!(database.migrate(&directory).is_err());
        assert_eq!(std::fs::metadata(&path).unwrap().len(), 101);
        assert_eq!(
            database.load(id, 100).unwrap(),
            Loaded::Data(b"original".to_vec())
        );
        std::fs::remove_file(&path).unwrap();
        for index in 2..259 {
            std::fs::write(directory.join(format!("{index:032x}.json")), b"batch").unwrap();
        }
        database.migrate(&directory).unwrap();
        assert_eq!(database.usage(None).unwrap().records, 258);
        assert!(std::fs::read_dir(&directory).unwrap().all(|entry| {
            entry
                .unwrap()
                .path()
                .extension()
                .is_none_or(|extension| extension != "json")
        }));
        drop(database);
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn migration_compares_duplicate_blobs_across_chunks() {
        let directory =
            std::env::temp_dir().join(format!("tinyllm-sqlite-duplicate-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir(&directory).unwrap();
        let mut database = Database::open(&directory, true).unwrap();
        let id = "00000000000000000000000000000001";
        let path = directory.join(format!("{id}.json"));
        let original = vec![b'x'; 20000];
        database.save(id, &original, SystemTime::now()).unwrap();
        let mut conflicting = original.clone();
        *conflicting.last_mut().unwrap() = b'y';
        std::fs::write(&path, &conflicting).unwrap();
        assert!(
            database
                .migrate(&directory)
                .unwrap_err()
                .to_string()
                .contains("conflicts")
        );
        assert_eq!(std::fs::read(&path).unwrap(), conflicting);
        assert_eq!(
            database.load(id, original.len()).unwrap(),
            Loaded::Data(original.clone())
        );
        std::fs::write(&path, &original).unwrap();
        database.migrate(&directory).unwrap();
        assert!(!path.exists());
        assert_eq!(database.usage(None).unwrap().records, 1);
        drop(database);
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn database_rejects_unknown_versions_and_unsafe_files() {
        let directory =
            std::env::temp_dir().join(format!("tinyllm-sqlite-files-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir(&directory).unwrap();
        let database = Database::open(&directory, true).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                std::fs::metadata(directory.join(FILE))
                    .unwrap()
                    .permissions()
                    .mode()
                    & 0o777,
                0o600
            );
        }
        database
            .connection
            .pragma_update(None, "user_version", 2)
            .unwrap();
        drop(database);
        let before = std::fs::read(directory.join(FILE)).unwrap();
        for writable in [false, true] {
            assert!(Database::open(&directory, writable).is_err());
        }
        assert_eq!(std::fs::read(directory.join(FILE)).unwrap(), before);
        std::fs::remove_file(directory.join(FILE)).unwrap();
        #[cfg(unix)]
        {
            std::fs::write(directory.join("credentials"), b"private").unwrap();
            for name in [
                FILE,
                "state.sqlite-journal",
                "state.sqlite-wal",
                "state.sqlite-shm",
            ] {
                std::os::unix::fs::symlink(directory.join("credentials"), directory.join(name))
                    .unwrap();
                assert!(Database::open(&directory, true).is_err());
                assert_eq!(
                    std::fs::read(directory.join("credentials")).unwrap(),
                    b"private"
                );
                std::fs::remove_file(directory.join(name)).unwrap();
            }
        }
        std::fs::remove_dir_all(directory).unwrap();
    }
}
