// Format/filter conventions: https://github.com/MeteoraAg/methub/blob/main/crates/logger/src/lib.rs
use eyre::{Result, WrapErr, bail};
use serde::Deserialize;
use std::io::IsTerminal;
use tracing_subscriber::{
    Layer, filter::Targets, fmt::MakeWriter, layer::SubscriberExt, util::SubscriberInitExt,
};

#[derive(Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub level: String,
    pub format: Format,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            level: "tinyllm=info".into(),
            format: Format::Compact,
        }
    }
}

#[derive(Clone, Copy, Default, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum Format {
    #[default]
    Compact,
    Full,
    Json,
    Pretty,
}

pub fn init(config: &Config) -> Result<()> {
    subscriber(config, std::io::stderr)?
        .try_init()
        .map_err(|_| eyre::eyre!("cannot initialize logging"))
}

fn targets(level: &str) -> Result<Targets> {
    if level.trim().is_empty() {
        bail!("logging.level must not be empty; use off to disable logging");
    }
    level
        .parse()
        .wrap_err_with(|| "invalid logging.level or RUST_LOG")
}

fn subscriber<W>(config: &Config, writer: W) -> Result<impl tracing::Subscriber + Send + Sync>
where
    W: for<'a> MakeWriter<'a> + Send + Sync + 'static,
{
    let format = tracing_subscriber::fmt::layer()
        .with_writer(writer)
        .with_ansi(std::io::stderr().is_terminal())
        .with_line_number(true);
    let format = match config.format {
        Format::Compact => format.compact().boxed(),
        Format::Full => format.boxed(),
        Format::Json => format.json().boxed(),
        Format::Pretty => format.pretty().boxed(),
    };
    Ok(tracing_subscriber::registry()
        .with(targets(&config.level)?)
        .with(format))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{self, Write};
    use std::sync::{Arc, Mutex};

    struct Capture(Arc<Mutex<Vec<u8>>>);

    impl Write for Capture {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.0.lock().unwrap().write(bytes)
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn rejects_invalid_filters_without_echoing_them() {
        for level in ["", " ", "info,tinyllm=secret-key"] {
            let error = format!("{:?}", targets(level).unwrap_err());
            assert!(!error.contains("secret-key"));
        }
        assert!(targets("warn,tinyllm=debug").is_ok());
    }

    #[test]
    fn formats_allowed_events_and_filters_dependency_details_by_default() {
        for format in [Format::Compact, Format::Full, Format::Json, Format::Pretty] {
            let output = Arc::new(Mutex::new(Vec::new()));
            let config = Config {
                format,
                ..Config::default()
            };
            let captured = output.clone();
            let subscriber = subscriber(&config, move || Capture(captured.clone())).unwrap();
            tracing::subscriber::with_default(subscriber, || {
                tracing::info!(target: "reqwest", url = "https://private.invalid", authorization = "secret-key", body = "secret-prompt", "transport");
                tracing::debug!(target: "tinyllm::proxy", "debug details");
                tracing::info!(target: "tinyllm::proxy", status = 200u16, elapsed_ms = 12u64, "request completed");
            });
            let text = String::from_utf8(output.lock().unwrap().clone()).unwrap();
            assert!(text.contains("request completed"));
            assert!(!text.contains("debug details"));
            assert!(!text.contains("private.invalid"));
            assert!(!text.contains("secret-"));
            if matches!(format, Format::Json) {
                let record: serde_json::Value = serde_json::from_str(&text).unwrap();
                assert_eq!(record["fields"]["status"], 200);
                assert_eq!(record["fields"]["elapsed_ms"], 12);
            }
        }
    }

    #[tokio::test]
    async fn warns_when_gateway_authentication_is_disabled() {
        use tracing::instrument::WithSubscriber;

        for token in [None, Some("fixture-local-secret")] {
            let output = Arc::new(Mutex::new(Vec::new()));
            let captured = output.clone();
            let subscriber = subscriber(
                &Config {
                    format: Format::Json,
                    ..Config::default()
                },
                move || Capture(captured.clone()),
            )
            .unwrap();
            let config = serde_json::from_value(serde_json::json!({
                "server":{"auth_token":token},
                "providers":{"fixture":{"type":"openrouter","api_key":"fixture-upstream-secret"}}
            }))
            .unwrap();
            drop(
                crate::server::router(config)
                    .with_subscriber(subscriber)
                    .await
                    .unwrap(),
            );
            let text = String::from_utf8(output.lock().unwrap().clone()).unwrap();
            assert_eq!(
                text.contains("gateway authentication is disabled"),
                token.is_none()
            );
            assert!(!text.contains("secret"));
            if token.is_none() {
                let record: serde_json::Value = serde_json::from_str(&text).unwrap();
                assert_eq!(record["level"], "WARN");
                assert_eq!(record["fields"]["bind"], "127.0.0.1:8080");
            }
        }
    }

    #[test]
    fn unknown_field_diagnostics_are_bounded_and_omit_values() {
        let output = Arc::new(Mutex::new(Vec::new()));
        let captured = output.clone();
        let subscriber = subscriber(
            &Config {
                format: Format::Json,
                ..Config::default()
            },
            move || Capture(captured.clone()),
        )
        .unwrap();
        tracing::subscriber::with_default(subscriber, || {
            for key in [
                "future_control".into(),
                "x".repeat(1000),
                "bad\nfield".into(),
            ] {
                let value = serde_json::json!({key: "secret-prompt"});
                assert!(crate::providers::openai::protocol::fields(&value, &[]).is_err());
            }
        });
        let text = String::from_utf8(output.lock().unwrap().clone()).unwrap();
        let events = text
            .lines()
            .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(events.len(), 3);
        assert_eq!(events[0]["fields"]["field"], "future_control");
        assert_eq!(events[1]["fields"]["field"].as_str().unwrap().len(), 64);
        assert!(
            !events[2]["fields"]["field"]
                .as_str()
                .unwrap()
                .contains('\n')
        );
        assert!(!text.contains("secret-prompt"));
    }
}
