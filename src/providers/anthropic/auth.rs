use super::models::AnthropicAuth;
use axum::{Router, extract::Query, http::StatusCode, routing::get};
use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD};
use eyre::{OptionExt, Result, WrapErr, bail};
use futures::StreamExt;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    fs::{File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use subtle::ConstantTimeEq;
use tokio::sync::{Mutex, OwnedMutexGuard, mpsc};

const AUTHORIZE_URL: &str = "https://claude.ai/oauth/authorize";
const TOKEN_URL: &str = "https://platform.claude.com/v1/oauth/token";
const CLIENT_ID: &str = "9d1c250a-e61b-44d9-88ed-5944d1962f5e";
const REDIRECT_URI: &str = "http://localhost:53692/callback";
const SCOPES: &str = "user:profile user:inference";
const MAX_AUTH_BYTES: usize = 64 * 1024;

pub enum Auth {
    ApiKey(String),
    Subscription(Arc<Session>),
}

impl Auth {
    pub fn open(config: &AnthropicAuth) -> Result<Self> {
        match config {
            AnthropicAuth::ApiKey(_) => Ok(Self::ApiKey(config.api_key()?.to_owned())),
            AnthropicAuth::Subscription(options) => {
                let session = Session::open(&options.credentials_dir)?;
                if session.load()?.is_none() {
                    bail!(
                        "no subscription credentials; run tinyllm anthropic login with this config"
                    );
                }
                Ok(Self::Subscription(Arc::new(session)))
            }
        }
    }

    pub fn is_subscription(&self) -> bool {
        matches!(self, Self::Subscription(_))
    }

    #[cfg(test)]
    pub(crate) fn set_token_url(&mut self, token_url: String) {
        if let Self::Subscription(session) = self {
            Arc::get_mut(session).unwrap().token_url = token_url;
        }
    }

    pub async fn access(&self, rejected: Option<&str>) -> Result<String> {
        match self {
            Self::ApiKey(key) => Ok(key.clone()),
            Self::Subscription(session) => {
                let state = session.credentials.clone().lock_owned().await;
                let session = session.clone();
                let rejected = rejected.map(str::to_owned);
                tokio::spawn(async move { session.access(state, rejected.as_deref()).await })
                    .await
                    .wrap_err_with(|| "subscription refresh task failed")?
            }
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
struct Credentials {
    access_token: String,
    refresh_token: String,
    expires_at: u64,
}

#[derive(Deserialize)]
struct TokenResponse {
    access_token: String,
    refresh_token: Option<String>,
    expires_in: u64,
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn valid_secret(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= MAX_AUTH_BYTES
        && value.bytes().all(|byte| byte.is_ascii_graphic())
}

impl Credentials {
    fn from_response(response: TokenResponse, previous: Option<&Self>) -> Result<Self> {
        let credentials = Self {
            access_token: response.access_token,
            refresh_token: response
                .refresh_token
                .or_else(|| previous.map(|credentials| credentials.refresh_token.clone()))
                .ok_or_eyre("OAuth response omitted the refresh token; login cannot be saved")?,
            expires_at: now().saturating_add(response.expires_in),
        };
        credentials.validate()?;
        if credentials.expires_at <= now().saturating_add(120) {
            bail!("OAuth returned an expired or near-expiry access token; log in again");
        }
        Ok(credentials)
    }

    fn validate(&self) -> Result<()> {
        if !valid_secret(&self.access_token) || !valid_secret(&self.refresh_token) {
            bail!("subscription credentials contain an empty or invalid token");
        }
        Ok(())
    }
}

pub struct Session {
    directory: PathBuf,
    _lock: File,
    credentials: Arc<Mutex<CredentialState>>,
    client: reqwest::Client,
    token_url: String,
}

#[derive(Default)]
struct CredentialState {
    current: Option<Credentials>,
    unsaved: bool,
}

impl Session {
    pub fn open(directory: &Path) -> Result<Self> {
        let mut builder = std::fs::DirBuilder::new();
        builder.recursive(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::DirBuilderExt;
            builder.mode(0o700);
        }
        builder
            .create(directory)
            .wrap_err_with(|| "cannot create Anthropic subscription credential directory")?;
        private_path(directory, true)?;
        let lock_path = directory.join("anthropic.lock");
        match std::fs::symlink_metadata(&lock_path) {
            Ok(_) => private_path(&lock_path, false)?,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
        let mut options = OpenOptions::new();
        options.read(true).write(true).create(true).truncate(false);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let lock = options
            .open(lock_path)
            .wrap_err_with(|| "cannot open Anthropic subscription credential lock")?;
        lock.try_lock().map_err(|_| {
            eyre::eyre!(
                "Anthropic subscription credentials are in use; stop the other tinyllm process before login/logout"
            )
        })?;
        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .connect_timeout(Duration::from_secs(10))
            .timeout(Duration::from_secs(30))
            .user_agent(concat!("tinyllm/", env!("CARGO_PKG_VERSION")))
            .build()?;
        Ok(Self {
            directory: directory.into(),
            _lock: lock,
            credentials: Arc::new(Mutex::new(CredentialState::default())),
            client,
            token_url: TOKEN_URL.into(),
        })
    }

    fn path(&self) -> PathBuf {
        self.directory.join("anthropic.json")
    }

    fn load(&self) -> Result<Option<Credentials>> {
        let path = self.path();
        if !path.try_exists()? {
            return Ok(None);
        }
        private_path(&path, false)?;
        let mut bytes = Vec::new();
        File::open(path)?
            .take(MAX_AUTH_BYTES as u64 + 1)
            .read_to_end(&mut bytes)?;
        if bytes.len() > MAX_AUTH_BYTES {
            bail!("Anthropic subscription credential file exceeds 64 KiB");
        }
        let credentials: Credentials = serde_json::from_slice(&bytes).map_err(|_| {
            eyre::eyre!(
                "invalid Anthropic subscription credential file; run tinyllm anthropic login again"
            )
        })?;
        credentials.validate()?;
        Ok(Some(credentials))
    }

    fn save(&self, credentials: &Credentials) -> Result<()> {
        credentials.validate()?;
        let bytes = serde_json::to_vec(credentials)?;
        if bytes.len() > MAX_AUTH_BYTES {
            bail!("Anthropic subscription credentials exceed 64 KiB");
        }
        let temporary = self
            .directory
            .join(format!(".anthropic.{}.tmp", uuid::Uuid::new_v4().simple()));
        let result = (|| -> Result<()> {
            let mut options = OpenOptions::new();
            options.write(true).create_new(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                options.mode(0o600);
            }
            let mut file = options.open(&temporary)?;
            file.write_all(&bytes)?;
            file.sync_all()?;
            std::fs::rename(&temporary, self.path())?;
            #[cfg(unix)]
            File::open(&self.directory)?.sync_all()?;
            Ok(())
        })();
        if result.is_err() {
            let _ = std::fs::remove_file(temporary);
        }
        result.wrap_err_with(|| "cannot persist Anthropic subscription credentials")
    }

    async fn access(
        &self,
        mut state: OwnedMutexGuard<CredentialState>,
        rejected: Option<&str>,
    ) -> Result<String> {
        if state.current.is_none() {
            state.current = Some(
                self.load()?
                    .ok_or_eyre("no subscription credentials; run tinyllm anthropic login")?,
            );
        }
        if state.unsaved {
            self.save(state.current.as_ref().unwrap())?;
            state.unsaved = false;
        }
        let credentials = state.current.as_ref().unwrap();
        if credentials.expires_at <= now().saturating_add(120)
            || rejected == Some(credentials.access_token.as_str())
        {
            let response = self
                .client
                .post(&self.token_url)
                .json(&json!({
                    "grant_type": "refresh_token",
                    "client_id": CLIENT_ID,
                    "refresh_token": credentials.refresh_token,
                }))
                .send()
                .await
                .map_err(|_| eyre::eyre!("cannot connect to Anthropic for subscription refresh"))?;
            let response = read_json(
                response,
                "Anthropic subscription refresh failed; run tinyllm anthropic login again",
            )
            .await?;
            state.current = Some(Credentials::from_response(response, Some(credentials))?);
            state.unsaved = true;
            self.save(state.current.as_ref().unwrap())?;
            state.unsaved = false;
        }
        Ok(state.current.as_ref().unwrap().access_token.clone())
    }

    async fn exchange(&self, code: &str, state: &str, verifier: &str) -> Result<Credentials> {
        let response = self
            .client
            .post(&self.token_url)
            .json(&json!({
                "grant_type": "authorization_code",
                "client_id": CLIENT_ID,
                "code": code,
                "state": state,
                "redirect_uri": REDIRECT_URI,
                "code_verifier": verifier,
            }))
            .send()
            .await
            .map_err(|_| eyre::eyre!("cannot connect to Anthropic for login exchange"))?;
        Credentials::from_response(
            read_json(response, "Anthropic login exchange failed").await?,
            None,
        )
    }

    pub async fn login(&self, headless: bool) -> Result<()> {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:53692")
            .await
            .wrap_err_with(
                || "cannot bind OAuth callback port 53692; close the other Anthropic login",
            )?;
        let verifier = format!(
            "{}{}",
            uuid::Uuid::new_v4().simple(),
            uuid::Uuid::new_v4().simple()
        );
        let challenge = URL_SAFE_NO_PAD.encode(Sha256::digest(verifier.as_bytes()));
        let state = format!(
            "{}{}",
            uuid::Uuid::new_v4().simple(),
            uuid::Uuid::new_v4().simple()
        );
        let url = authorize_url(AUTHORIZE_URL, &challenge, &state)?;
        println!("Open this URL in your browser to sign in:\n\n{url}\n");
        crate::providers::browser::open(url.as_str(), headless);
        let (code, returned_state) =
            tokio::time::timeout(Duration::from_secs(300), callback(listener, state))
                .await
                .wrap_err_with(|| "browser login timed out after five minutes")??;
        let credentials = self.exchange(&code, &returned_state, &verifier).await?;
        self.save(&credentials)
    }

    pub fn logout(&self) -> Result<()> {
        match std::fs::remove_file(self.path()) {
            Ok(()) => {
                #[cfg(unix)]
                File::open(&self.directory)?.sync_all()?;
                Ok(())
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(error)
                .wrap_err_with(|| "cannot remove local Anthropic subscription credentials"),
        }
    }
}

#[cfg(test)]
pub(crate) fn save_fixture(
    directory: &Path,
    access_token: &str,
    refresh_token: &str,
    expires_at: u64,
) {
    let session = Session::open(directory).unwrap();
    session
        .save(&Credentials {
            access_token: access_token.into(),
            refresh_token: refresh_token.into(),
            expires_at,
        })
        .unwrap();
}

#[cfg(test)]
pub(crate) fn fixture_expiry() -> u64 {
    now() + 3600
}

fn private_path(path: &Path, directory: bool) -> Result<()> {
    let metadata = std::fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink()
        || if directory {
            !metadata.is_dir()
        } else {
            !metadata.is_file()
        }
    {
        bail!("subscription credential paths must be regular files/directories, not symlinks");
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if metadata.permissions().mode() & 0o077 != 0 {
            bail!("subscription credentials must be private: directory mode 0700, files 0600");
        }
    }
    Ok(())
}

async fn read_json<T: DeserializeOwned>(response: reqwest::Response, failure: &str) -> Result<T> {
    if !response.status().is_success() {
        bail!("{failure} (HTTP {})", response.status().as_u16());
    }
    let mut chunks = response.bytes_stream();
    let mut bytes = Vec::new();
    while let Some(chunk) = chunks.next().await {
        let chunk = chunk.map_err(|_| eyre::eyre!("OAuth response interrupted"))?;
        if bytes.len().saturating_add(chunk.len()) > MAX_AUTH_BYTES {
            bail!("OAuth response exceeds 64 KiB");
        }
        bytes.extend_from_slice(&chunk);
    }
    serde_json::from_slice(&bytes).map_err(|_| eyre::eyre!("invalid OAuth response"))
}

fn authorize_url(base: &str, challenge: &str, state: &str) -> Result<reqwest::Url> {
    let mut url = reqwest::Url::parse(base)?;
    url.query_pairs_mut().extend_pairs([
        ("code", "true"),
        ("client_id", CLIENT_ID),
        ("response_type", "code"),
        ("redirect_uri", REDIRECT_URI),
        ("scope", SCOPES),
        ("code_challenge", challenge),
        ("code_challenge_method", "S256"),
        ("state", state),
    ]);
    Ok(url)
}

#[derive(Deserialize)]
struct Callback {
    code: Option<String>,
    state: Option<String>,
    error: Option<String>,
}

async fn callback(listener: tokio::net::TcpListener, state: String) -> Result<(String, String)> {
    use std::future::IntoFuture;
    let (sender, mut receiver) = mpsc::channel(1);
    let app = Router::new().route(
        "/callback",
        get(move |Query(query): Query<Callback>| {
            let sender = sender.clone();
            let state = state.clone();
            async move {
                if !query
                    .state
                    .as_ref()
                    .is_some_and(|value| bool::from(value.as_bytes().ct_eq(state.as_bytes())))
                {
                    return (
                        StatusCode::BAD_REQUEST,
                        "Login state mismatch. Return to the original login link.",
                    );
                }
                let result = if query.error.is_some() {
                    Err(eyre::eyre!("Anthropic login was denied or cancelled"))
                } else {
                    query
                        .code
                        .filter(|code| !code.is_empty() && code.len() <= 8192)
                        .map(|code| (code, state))
                        .ok_or_eyre("OAuth callback omitted the authorization code")
                };
                let failed = result.is_err();
                let _ = sender.try_send(result);
                if failed {
                    (StatusCode::BAD_REQUEST, "Anthropic login did not complete.")
                } else {
                    (
                        StatusCode::OK,
                        "Authorization received. Return to the terminal for the login result.",
                    )
                }
            }
        }),
    );
    tokio::select! {
        result = receiver.recv() => result.ok_or_eyre("OAuth callback closed")?,
        result = axum::serve(listener, app).into_future() => {
            result.wrap_err_with(|| "OAuth callback server failed")?;
            bail!("OAuth callback server stopped before login");
        }
    }
}

#[cfg(test)]
#[path = "auth_tests.rs"]
mod tests;
