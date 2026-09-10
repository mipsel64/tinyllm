use clap::{Parser, Subcommand};
use std::path::PathBuf;
use tinyllm::{config, logging, providers, server};

#[derive(Parser)]
#[command(version = env!("TINYLLM_VERSION"), about)]
struct Cli {
    #[arg(
        short,
        long,
        global = true,
        help = "TOML or YAML config file [default: ~/.config/tinyllm/config.toml]"
    )]
    config: Option<PathBuf>,
    #[arg(
        long,
        global = true,
        env = "RUST_LOG",
        help = "Log filter [default: tinyllm=info]"
    )]
    log_level: Option<String>,
    #[arg(
        long,
        global = true,
        env = "LOG_FORMAT",
        value_enum,
        help = "Log format [default: compact]"
    )]
    log_format: Option<logging::Format>,
    #[command(subcommand)]
    command: Option<Command>,
}

impl Cli {
    fn config_path(&self) -> eyre::Result<PathBuf> {
        if let Some(path) = &self.config {
            return Ok(path.clone());
        }
        let home = std::env::var_os("HOME")
            .map(PathBuf::from)
            .filter(|path| path.is_absolute())
            .ok_or_else(|| eyre::eyre!("set HOME or provide --config"))?;
        Ok(home.join(".config/tinyllm/config.toml"))
    }
}

#[derive(Subcommand)]
enum Command {
    #[command(
        name = "openai",
        subcommand,
        about = "Manage OpenAI subscription authentication"
    )]
    OpenAi(OpenAiCommand),
    #[command(subcommand, about = "Inspect and explicitly prune continuation state")]
    State(StateCommand),
}

#[derive(Subcommand)]
enum StateCommand {
    /// Report a snapshot of continuation file counts and bytes.
    Status,
    /// Preview old continuation files; stop the gateway before cleanup.
    Prune {
        #[arg(long, value_parser = clap::value_parser!(u64).range(1..), help = "Select files last modified more than N days ago")]
        older_than_days: u64,
        #[arg(long, help = "Delete selected files; pruned turns cannot resume")]
        apply: bool,
    },
}

#[derive(Subcommand)]
enum OpenAiCommand {
    /// Sign in with a ChatGPT subscription.
    Login {
        #[arg(
            long,
            help = "Use a device code instead of a localhost browser callback"
        )]
        device_auth: bool,
        #[arg(long, default_value = "openai")]
        provider: String,
    },
    /// Remove tinyllm's locally stored subscription credentials.
    Logout {
        #[arg(long, default_value = "openai")]
        provider: String,
    },
}

#[tokio::main]
async fn main() -> eyre::Result<()> {
    let cli = Cli::parse();
    let mut config = config::Config::load(&cli.config_path()?)?;
    if let Some(level) = cli.log_level {
        config.logging.level = level;
    }
    if let Some(format) = cli.log_format {
        config.logging.format = format;
    }
    logging::init(&config.logging)?;
    if let Some(Command::State(command)) = &cli.command {
        return state_command(command, &config.server).await;
    }
    if let Some(Command::OpenAi(command)) = cli.command {
        let prefix = match &command {
            OpenAiCommand::Login { provider, .. } | OpenAiCommand::Logout { provider } => provider,
        };
        let Some(config::ProviderConfig::OpenAi(openai)) = config.providers.get(prefix) else {
            eyre::bail!("login and logout require a configured OpenAI provider");
        };
        let providers::openai::models::OpenAiAuth::Subscription(options) = &openai.auth else {
            eyre::bail!("set the provider auth.type to Subscription before login or logout");
        };
        let session = providers::openai::auth::Session::open(&options.credentials_dir)?;
        match command {
            OpenAiCommand::Login { device_auth, .. } => {
                tokio::select! {
                    result = session.login(device_auth) => result?,
                    _ = tokio::signal::ctrl_c() => eyre::bail!("login cancelled"),
                }
                println!("Subscription login saved. Start tinyllm with the same config.");
            }
            OpenAiCommand::Logout { .. } => {
                session.logout()?;
                println!("Local subscription credentials removed.");
            }
        }
        return Ok(());
    }
    let bind = config.server.bind;
    let app = server::router(config).await?;
    let listener = tokio::net::TcpListener::bind(bind).await?;
    #[cfg(unix)]
    let (mut interrupt, mut terminate) = {
        use tokio::signal::unix::{SignalKind, signal};
        (
            signal(SignalKind::interrupt())?,
            signal(SignalKind::terminate())?,
        )
    };
    tracing::info!(
        address = %format_args!("http://{}/anthropic", listener.local_addr()?),
        "tinyllm listening"
    );
    axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            #[cfg(unix)]
            tokio::select! {
                _ = interrupt.recv() => {},
                _ = terminate.recv() => {},
            }
            #[cfg(not(unix))]
            let _ = tokio::signal::ctrl_c().await;
            tracing::info!("shutdown requested");
        })
        .await?;
    tracing::info!("server stopped");
    Ok(())
}

async fn state_command(command: &StateCommand, config: &config::Server) -> eyre::Result<()> {
    use providers::openai::state::{DATABASE_FILE, Store};
    println!("State: {}", config.state_dir.display());
    match command {
        StateCommand::Status => {
            let usage = Store::status(&config.state_dir).await?;
            print_usage("Snapshot", usage, config.max_state_bytes);
            match tokio::fs::symlink_metadata(config.state_dir.join(DATABASE_FILE)).await {
                Ok(metadata) if metadata.is_file() => println!(
                    "SQLite file: {} bytes, including reusable free pages (journal excluded)",
                    metadata.len()
                ),
                Ok(_) => eyre::bail!("state database must be a regular file"),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => return Err(error.into()),
            }
            if usage.is_high(config.max_state_bytes) {
                eprintln!(
                    "State is at least 80% full. Stop the gateway and preview tinyllm state prune before cleanup."
                );
            }
        }
        StateCommand::Prune {
            older_than_days,
            apply,
        } => {
            let seconds = older_than_days
                .checked_mul(86_400)
                .ok_or_else(|| eyre::eyre!("--older-than-days is too large"))?;
            let cutoff = std::time::SystemTime::now()
                .checked_sub(std::time::Duration::from_secs(seconds))
                .ok_or_else(|| eyre::eyre!("--older-than-days is too large"))?;
            eprintln!("Stop the gateway before cleanup. Pruned turns cannot resume.");
            let report = Store::prune(&config.state_dir, cutoff, *apply).await?;
            print_usage(
                if *apply { "Removed" } else { "Preview" },
                report.selected,
                config.max_state_bytes,
            );
            print_usage("Remaining", report.after, config.max_state_bytes);
            if !apply {
                println!("No records removed. Repeat with --apply to delete the selection.");
            }
        }
    }
    Ok(())
}

fn print_usage(label: &str, usage: providers::openai::state::Usage, limit: u64) {
    println!(
        "{label}: {} records, {} temporary files, {} / {limit} record bytes",
        usage.records, usage.temporary_files, usage.bytes
    );
}

#[cfg(test)]
mod tests {
    #[test]
    fn cli_version_includes_commit_and_build_date() {
        use super::Cli;
        use clap::{Parser, error::ErrorKind};
        for flag in ["--version", "-V"] {
            let error = Cli::try_parse_from(["tinyllm", "--config", "missing.toml", flag])
                .err()
                .unwrap();
            assert_eq!(error.kind(), ErrorKind::DisplayVersion);
            let output = error.to_string();
            let metadata = output
                .trim_end()
                .strip_prefix(&format!("tinyllm {}+", env!("CARGO_PKG_VERSION")))
                .expect("version must include build metadata");
            let (commit, date) = metadata.split_once(' ').unwrap();
            assert!(
                commit == "unknown"
                    || (commit.len() == 7 && commit.bytes().all(|b| b.is_ascii_hexdigit()))
            );
            assert_eq!(date.len(), 20);
            assert!(date.bytes().enumerate().all(|(i, b)| match i {
                4 | 7 => b == b'-',
                10 => b == b'T',
                13 | 16 => b == b':',
                19 => b == b'Z',
                _ => b.is_ascii_digit(),
            }));
        }
    }

    #[test]
    fn cli_accepts_explicit_state_management() {
        use super::{Cli, Command, StateCommand};
        use clap::{Parser, error::ErrorKind};
        assert!(matches!(
            Cli::try_parse_from(["tinyllm", "state", "status"])
                .unwrap()
                .command,
            Some(Command::State(StateCommand::Status))
        ));
        for apply in [false, true] {
            let mut args = vec![
                "tinyllm",
                "state",
                "-c",
                "state.toml",
                "prune",
                "--older-than-days",
                "30",
            ];
            if apply {
                args.push("--apply");
            }
            let cli = Cli::try_parse_from(args).unwrap();
            assert_eq!(
                cli.config_path().unwrap(),
                std::path::Path::new("state.toml")
            );
            assert!(
                matches!(cli.command, Some(Command::State(StateCommand::Prune { older_than_days: 30, apply: selected })) if selected == apply)
            );
        }
        for args in [
            vec!["tinyllm", "state"],
            vec!["tinyllm", "state", "status", "--apply"],
            vec!["tinyllm", "state", "prune"],
            vec!["tinyllm", "state", "prune", "--apply"],
            vec!["tinyllm", "state", "prune", "--older-than-days"],
        ] {
            assert!(Cli::try_parse_from(args).is_err());
        }
        for days in ["0", "-1", "1.5", "many", "18446744073709551616"] {
            assert!(
                Cli::try_parse_from(["tinyllm", "state", "prune", "--older-than-days", days])
                    .is_err()
            );
        }
        for action in ["status", "prune"] {
            assert_eq!(
                Cli::try_parse_from(["tinyllm", "state", action, "--help"])
                    .err()
                    .unwrap()
                    .kind(),
                ErrorKind::DisplayHelp
            );
        }
    }

    #[tokio::test]
    async fn state_prune_rejects_age_overflow_without_creating_state() {
        let directory =
            std::env::temp_dir().join(format!("tinyllm-cli-state-{}", uuid::Uuid::new_v4()));
        let config = tinyllm::config::Server {
            state_dir: directory.clone(),
            ..Default::default()
        };
        for apply in [false, true] {
            assert!(
                super::state_command(
                    &super::StateCommand::Prune {
                        older_than_days: u64::MAX,
                        apply
                    },
                    &config
                )
                .await
                .unwrap_err()
                .to_string()
                .contains("too large")
            );
        }
        assert!(!directory.exists());
    }

    #[test]
    fn cli_accepts_config_paths_and_reports_usage() {
        use super::{Cli, Command, OpenAiCommand};
        use clap::{Parser, error::ErrorKind};
        assert_eq!(
            Cli::try_parse_from(["tinyllm"])
                .unwrap()
                .config_path()
                .unwrap(),
            std::path::PathBuf::from(std::env::var_os("HOME").unwrap())
                .join(".config/tinyllm/config.toml")
        );
        assert!(Cli::try_parse_from(["tinyllm"]).unwrap().command.is_none());
        for flag in ["-c", "--config"] {
            assert_eq!(
                Cli::try_parse_from(["tinyllm", flag, "config with spaces.yaml"])
                    .unwrap()
                    .config_path()
                    .unwrap(),
                std::path::Path::new("config with spaces.yaml")
            );
        }
        for args in [
            vec![
                "tinyllm",
                "openai",
                "login",
                "--device-auth",
                "-c",
                "subscription.yaml",
            ],
            vec!["tinyllm", "-c", "subscription.yaml", "openai", "login"],
            vec!["tinyllm", "openai", "-c", "subscription.yaml", "login"],
        ] {
            let device = args.contains(&"--device-auth");
            let cli = Cli::try_parse_from(args).unwrap();
            assert!(matches!(&cli.command,
                Some(Command::OpenAi(OpenAiCommand::Login { device_auth, provider }))
                    if *device_auth == device && provider == "openai"));
            assert_eq!(
                cli.config_path().unwrap(),
                std::path::Path::new("subscription.yaml")
            );
        }
        for provider in [None, Some("codex")] {
            for action in ["login", "logout"] {
                let mut args = vec!["tinyllm", "openai", action];
                if let Some(provider) = provider {
                    args.extend(["--provider", provider]);
                }
                let Some(Command::OpenAi(command)) = Cli::try_parse_from(args).unwrap().command
                else {
                    panic!("expected OpenAI command")
                };
                let selected = match (action, command) {
                    (
                        "login",
                        OpenAiCommand::Login {
                            provider,
                            device_auth: false,
                        },
                    )
                    | ("logout", OpenAiCommand::Logout { provider }) => provider,
                    _ => panic!("wrong OpenAI command"),
                };
                assert_eq!(selected, provider.unwrap_or("openai"));
            }
        }
        for args in [
            vec!["tinyllm", "openai"],
            vec!["tinyllm", "login"],
            vec!["tinyllm", "logout"],
        ] {
            assert!(Cli::try_parse_from(args).is_err());
        }
        for args in [
            vec!["tinyllm", "openai", "--help"],
            vec!["tinyllm", "openai", "login", "--help"],
            vec!["tinyllm", "openai", "logout", "--help"],
        ] {
            assert_eq!(
                Cli::try_parse_from(args).err().unwrap().kind(),
                ErrorKind::DisplayHelp
            );
        }
        for (flag, expected) in [
            ("--help", ErrorKind::DisplayHelp),
            ("--version", ErrorKind::DisplayVersion),
            ("--unknown", ErrorKind::UnknownArgument),
            ("--config", ErrorKind::InvalidValue),
        ] {
            assert_eq!(
                Cli::try_parse_from(["tinyllm", flag]).err().unwrap().kind(),
                expected
            );
        }
    }
}
