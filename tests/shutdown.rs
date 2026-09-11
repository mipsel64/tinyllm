#![cfg(unix)]

use std::{
    io::{BufRead, BufReader},
    process::{Child, Command, Stdio},
    sync::mpsc,
    time::{Duration, Instant},
};

struct Gateway(Child, std::path::PathBuf);

impl Drop for Gateway {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
        let _ = std::fs::remove_dir_all(&self.1);
    }
}

#[test]
fn service_signals_shut_down_cleanly() {
    for signal in ["TERM", "INT"] {
        let directory = std::env::temp_dir().join(format!("tinyllm-stop-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir(&directory).unwrap();
        let config = directory.join("config.toml");
        std::fs::write(&config, "[server]\nbind='127.0.0.1:0'\nstate_dir='state'\n[providers.fixture]\ntype='openai'\nbase_url='http://127.0.0.1:9'\n[providers.fixture.auth]\ntype='ApiKey'\noptions='fixture-key'\n").unwrap();
        let child = Command::new(env!("CARGO_BIN_EXE_tinyllm"))
            .arg("--config")
            .arg(&config)
            .args(["--log-format", "json", "--log-level", "tinyllm=info"])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let mut gateway = Gateway(child, directory);
        let stderr = gateway.0.stderr.take().unwrap();
        let (send, receive) = mpsc::channel();
        std::thread::spawn(move || {
            for line in BufReader::new(stderr).lines() {
                let line = line.unwrap();
                let event: serde_json::Value = serde_json::from_str(&line).unwrap();
                if send
                    .send(event["fields"]["message"].as_str().unwrap().to_owned())
                    .is_err()
                {
                    break;
                }
            }
        });
        while receive.recv_timeout(Duration::from_secs(5)).unwrap() != "tinyllm listening" {}
        assert!(
            Command::new("kill")
                .args(["-s", signal, &gateway.0.id().to_string()])
                .status()
                .unwrap()
                .success()
        );
        let deadline = Instant::now() + Duration::from_secs(5);
        let status = loop {
            if let Some(status) = gateway.0.try_wait().unwrap() {
                break status;
            }
            assert!(Instant::now() < deadline, "shutdown timed out for {signal}");
            std::thread::sleep(Duration::from_millis(10));
        };
        assert!(status.success(), "{signal} must exit cleanly: {status}");
        let messages: Vec<_> = receive.iter().collect();
        assert!(
            messages
                .iter()
                .any(|message| message == "shutdown requested")
        );
        assert!(messages.iter().any(|message| message == "server stopped"));
    }
}
