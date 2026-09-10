#[test]
fn state_directory_defaults_follow_xdg() {
    if let Some(expected) = std::env::var_os("TINYLLM_TEST_EXPECTED_STATE_DIR") {
        assert_eq!(
            tinyllm::config::Server::default().state_dir,
            std::path::PathBuf::from(expected)
        );
        return;
    }
    let fallback =
        std::path::PathBuf::from(std::env::var_os("HOME").unwrap()).join(".local/state/tinyllm");
    let custom = std::env::temp_dir().join("tinyllm-xdg-test");
    for (xdg, expected) in [
        (None, fallback.clone()),
        (Some(std::path::PathBuf::from("relative")), fallback),
        (Some(custom.clone()), custom.join("tinyllm")),
    ] {
        let mut child = std::process::Command::new(std::env::current_exe().unwrap());
        child
            .args(["--exact", "state_directory_defaults_follow_xdg"])
            .env("TINYLLM_TEST_EXPECTED_STATE_DIR", expected)
            .env_remove("XDG_STATE_HOME");
        if let Some(xdg) = xdg {
            child.env("XDG_STATE_HOME", xdg);
        }
        let output = child.output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}
