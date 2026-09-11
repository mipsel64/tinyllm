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
