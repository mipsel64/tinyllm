use std::{path::Path, time::SystemTime};
use tinyllm::providers::openai::state::Store;

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
        .args(["--exact", "prune_requires_gateway_process_lock"])
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
