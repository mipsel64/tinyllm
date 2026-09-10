use serde_json::json;
use std::{
    path::PathBuf,
    sync::Mutex,
    time::{Duration, SystemTime},
};
use tinyllm::providers::openai::state::Store;

#[tokio::test]
async fn database_failures_log_causes_without_exposing_them_to_clients() {
    let directory =
        std::env::temp_dir().join(format!("tinyllm-sql-errors-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir(&directory).unwrap();
    let log = directory.join("warnings.log");
    tracing::subscriber::set_global_default(
        tracing_subscriber::fmt()
            .with_ansi(false)
            .with_max_level(tracing::Level::WARN)
            .with_writer(Mutex::new(std::fs::File::create(&log).unwrap()))
            .finish(),
    )
    .unwrap();
    let store =
        Store::open_with_cleanup(directory.clone(), 100000, 10000, Some(Default::default()))
            .await
            .unwrap();
    let reference = Store::reference();
    let content = json!([{"type":"redacted_thinking","data":reference}]);
    let response = json!({"output":[{"type":"reasoning","encrypted_content":"private-payload"}]});
    let database = rusqlite::Connection::open(directory.join("state.sqlite")).unwrap();
    database.execute_batch("CREATE TRIGGER fail_save BEFORE INSERT ON access BEGIN SELECT RAISE(ABORT, 'fixture-save-cause'); END;").unwrap();
    let error = store
        .save(&reference, "openai/m", &response, content.clone())
        .await
        .unwrap_err();
    assert_eq!(error.message, "cannot persist continuation state");
    database.execute_batch("DROP TRIGGER fail_save").unwrap();
    store
        .save(&reference, "openai/m", &response, content.clone())
        .await
        .unwrap();
    database.execute_batch("UPDATE access SET last_used = 1; CREATE TRIGGER fail_touch BEFORE UPDATE ON access BEGIN SELECT RAISE(ABORT, 'fixture-touch-cause'); END;").unwrap();
    let request = json!({"messages":[{"role":"assistant","content":content}]});
    let error = store
        .restore_scoped(&request, "openai", "m")
        .await
        .unwrap_err();
    assert_eq!(error.message, "cannot update continuation timestamp");
    database
        .execute_batch("DROP TRIGGER fail_touch; DROP TABLE records;")
        .unwrap();
    assert_eq!(
        store
            .restore_scoped(&request, "openai", "m")
            .await
            .unwrap_err()
            .message,
        "cannot read continuation state"
    );
    assert_eq!(
        store
            .save(&Store::reference(), "openai/m", &response, json!([]))
            .await
            .unwrap_err()
            .message,
        "cannot inspect continuation state"
    );
    let periodic = directory.join("periodic");
    std::fs::create_dir(&periodic).unwrap();
    let marker = std::fs::File::create(periodic.join(".cleanup-start")).unwrap();
    marker
        .set_modified(SystemTime::now() - Duration::from_secs(86400 * 60))
        .unwrap();
    let config = serde_json::from_value(json!({
        "server":{"state_dir":periodic,"state_cleanup":{"idle_days":1,"interval_seconds":1}},
        "providers":{"openai":{"type":"openai","auth":{"type":"ApiKey","options":"fixture"}}}
    }))
    .unwrap();
    let registry = tinyllm::providers::registry::Registry::new(&config)
        .await
        .unwrap();
    let periodic_db = rusqlite::Connection::open(periodic.join("state.sqlite")).unwrap();
    periodic_db.execute_batch("BEGIN; INSERT INTO records VALUES ('fixture', x'00'); INSERT INTO access VALUES ('fixture', 1, 1); CREATE TRIGGER fail_cleanup BEFORE DELETE ON records BEGIN SELECT RAISE(ABORT, 'fixture-cleanup-cause'); END; COMMIT;").unwrap();
    tokio::time::timeout(Duration::from_secs(5), async {
        while !std::fs::read_to_string(&log)
            .unwrap()
            .contains("continuation cleanup failed")
        {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    drop(registry);
    tokio::task::yield_now().await;
    drop(periodic_db);
    let logged = std::fs::read_to_string(log).unwrap();
    for cause in [
        "fixture-save-cause",
        "fixture-touch-cause",
        "no such table: records",
        "fixture-cleanup-cause",
    ] {
        assert!(logged.contains(cause), "missing {cause}: {logged}");
    }
    assert!(!logged.contains("private-payload"));
    drop(database);
    drop(store);
    std::fs::remove_dir_all(directory).unwrap();
}

#[tokio::test]
async fn interrupted_write_status_explains_recovery_without_writing() {
    const CHILD_DIRECTORY: &str = "TINYLLM_TEST_CRASH_STATE";
    if let Some(directory) = std::env::var_os(CHILD_DIRECTORY) {
        let connection =
            rusqlite::Connection::open(PathBuf::from(directory).join("state.sqlite")).unwrap();
        connection.execute_batch("PRAGMA cache_size = 10; BEGIN IMMEDIATE; UPDATE records SET data = randomblob(2000000);").unwrap();
        std::process::exit(9);
    }
    let directory =
        std::env::temp_dir().join(format!("tinyllm-sql-crash-{}", uuid::Uuid::new_v4()));
    let store = Store::open(directory.clone(), 100000, 10000).await.unwrap();
    let reference = Store::reference();
    let content = json!([{"type":"redacted_thinking","data":reference}]);
    let response = json!({"output":[{"type":"reasoning","encrypted_content":"original"}]});
    store
        .save(&reference, "openai/m", &response, content.clone())
        .await
        .unwrap();
    drop(store);
    let result = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "interrupted_write_status_explains_recovery_without_writing",
        ])
        .env(CHILD_DIRECTORY, &directory)
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(9));
    let before = std::fs::read(directory.join("state.sqlite")).unwrap();
    let journal = std::fs::read(directory.join("state.sqlite-journal")).unwrap();
    let error = Store::status(&directory).await.unwrap_err().to_string();
    assert!(error.contains("start the gateway"), "{error}");
    assert_eq!(
        std::fs::read(directory.join("state.sqlite")).unwrap(),
        before
    );
    assert_eq!(
        std::fs::read(directory.join("state.sqlite-journal")).unwrap(),
        journal
    );
    let store = Store::open(directory.clone(), 100000, 10000).await.unwrap();
    let restored = store
        .restore_scoped(
            &json!({"messages":[{"role":"assistant","content":content}]}),
            "openai",
            "m",
        )
        .await
        .unwrap();
    assert_eq!(restored[&0], *response["output"].as_array().unwrap());
    assert_eq!(Store::status(&directory).await.unwrap().records, 1);
    drop(store);
    std::fs::remove_dir_all(directory).unwrap();
}
