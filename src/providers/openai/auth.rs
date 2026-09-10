use super::models::OpenAiAuth;
use axum::{Router, extract::Query, http::StatusCode, routing::get};
use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD};
use eyre::{OptionExt, Result, WrapErr, bail};
use futures::StreamExt;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::{Value, json};
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

const ISSUER: &str = "https://auth.openai.com";
const CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";
const REDIRECT_URI: &str = "http://localhost:1455/auth/callback";
const MAX_AUTH_BYTES: usize = 64 * 1024;

pub enum Auth {
    ApiKey(String),
    Subscription(Arc<Session>),
}

impl Auth {
    pub fn open(config: &OpenAiAuth) -> Result<Self> {
        match config {
            OpenAiAuth::ApiKey(_) => Ok(Self::ApiKey(config.api_key()?.to_owned())),
            OpenAiAuth::Subscription(options) => {
                let session = Session::open(&options.credentials_dir)?;
                if session.load()?.is_none() {
                    bail!("no subscription credentials; run tinyllm openai login with this config");
                }
                Ok(Self::Subscription(Arc::new(session)))
            }
        }
    }

    pub async fn access(&self, rejected: Option<&str>) -> Result<(String, Option<String>)> {
        match self {
            Self::ApiKey(key) => Ok((key.clone(), None)),
            Self::Subscription(session) => {
                let state = session.credentials.clone().lock_owned().await;
                let session = session.clone();
                let rejected = rejected.map(str::to_owned);
                // Persist rotated tokens even if the waiting inference is cancelled.
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
    account_id: String,
    expires_at: u64,
}

#[derive(Deserialize)]
struct TokenResponse {
    access_token: String,
    refresh_token: Option<String>,
    id_token: Option<String>,
    expires_in: Option<u64>,
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
        && value.bytes().all(|b| b.is_ascii_graphic())
}

fn claims(token: &str) -> Option<Value> {
    let mut parts = token.split('.');
    parts.next()?;
    let payload = parts.next()?;
    if parts.next()?.is_empty() || parts.next().is_some() {
        return None;
    }
    serde_json::from_slice(&URL_SAFE_NO_PAD.decode(payload).ok()?).ok()
}

fn account_id(claims: &Value) -> Option<&str> {
    claims["https://api.openai.com/auth"]["chatgpt_account_id"]
        .as_str()
        .or_else(|| claims["chatgpt_account_id"].as_str())
}

impl Credentials {
    fn from_response(response: TokenResponse, previous: Option<&Self>) -> Result<Self> {
        let access_claims = claims(&response.access_token).unwrap_or(Value::Null);
        let id_claims = response
            .id_token
            .as_deref()
            .and_then(claims)
            .unwrap_or(Value::Null);
        if [&access_claims, &id_claims]
            .iter()
            .any(|v| v["https://api.openai.com/auth"]["chatgpt_account_is_fedramp"] == true)
        {
            bail!("this subscription requires a FedRAMP endpoint, which tinyllm does not support");
        }
        let account = account_id(&id_claims)
            .or_else(|| account_id(&access_claims))
            .or_else(|| previous.map(|p| p.account_id.as_str()))
            .ok_or_eyre("OAuth response omitted the ChatGPT account ID")?;
        if [&access_claims, &id_claims]
            .iter()
            .filter_map(|v| account_id(v))
            .any(|id| id != account)
            || previous.is_some_and(|p| p.account_id != account)
        {
            bail!("subscription account changed during refresh; run tinyllm openai login again");
        }
        let expires_at = access_claims["exp"]
            .as_u64()
            .unwrap_or_else(|| now().saturating_add(response.expires_in.unwrap_or(3600)));
        let credentials = Self {
            access_token: response.access_token,
            refresh_token: response
                .refresh_token
                .or_else(|| previous.map(|p| p.refresh_token.clone()))
                .ok_or_eyre("OAuth response omitted the refresh token; login cannot be saved")?,
            account_id: account.into(),
            expires_at,
        };
        credentials.validate()?;
        if expires_at <= now().saturating_add(120) {
            bail!("OAuth returned an expired or near-expiry access token; log in again");
        }
        Ok(credentials)
    }

    fn validate(&self) -> Result<()> {
        if ![&self.access_token, &self.refresh_token, &self.account_id]
            .into_iter()
            .all(|s| valid_secret(s))
        {
            bail!("subscription credentials contain an empty or invalid token/account ID");
        }
        Ok(())
    }
}

pub struct Session {
    directory: PathBuf,
    _lock: File,
    credentials: Arc<Mutex<CredentialState>>,
    client: reqwest::Client,
    issuer: String,
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
            .wrap_err_with(|| "cannot create subscription credential directory")?;
        private_path(directory, true)?;
        let lock_path = directory.join(".lock");
        if lock_path.try_exists()? {
            private_path(&lock_path, false)?;
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
            .wrap_err_with(|| "cannot open subscription credential lock")?;
        lock.try_lock().map_err(|_| eyre::eyre!("subscription credentials are in use; stop the other tinyllm process before login/logout"))?;
        let current = directory.join("openai.json");
        let legacy = directory.join("auth.json");
        if !current.try_exists()? && legacy.try_exists()? {
            private_path(&legacy, false)?;
            std::fs::rename(&legacy, &current)
                .wrap_err_with(|| "cannot migrate subscription credentials to openai.json")?;
            #[cfg(unix)]
            File::open(directory)?.sync_all()?;
        }
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
            issuer: ISSUER.into(),
        })
    }

    fn load(&self) -> Result<Option<Credentials>> {
        let path = self.directory.join("openai.json");
        if !path.try_exists()? {
            return Ok(None);
        }
        private_path(&path, false)?;
        let mut bytes = Vec::new();
        File::open(path)?
            .take(MAX_AUTH_BYTES as u64 + 1)
            .read_to_end(&mut bytes)?;
        if bytes.len() > MAX_AUTH_BYTES {
            bail!("subscription credential file exceeds 64 KiB");
        }
        let credentials: Credentials = serde_json::from_slice(&bytes).map_err(|_| {
            eyre::eyre!("invalid subscription credential file; run tinyllm openai login again")
        })?;
        credentials.validate()?;
        Ok(Some(credentials))
    }

    fn save(&self, credentials: &Credentials) -> Result<()> {
        credentials.validate()?;
        let bytes = serde_json::to_vec(credentials)?;
        if bytes.len() > MAX_AUTH_BYTES {
            bail!("subscription credentials exceed 64 KiB");
        }
        let temporary = self
            .directory
            .join(format!(".{}.tmp", uuid::Uuid::new_v4().simple()));
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
            std::fs::rename(&temporary, self.directory.join("openai.json"))?;
            #[cfg(unix)]
            File::open(&self.directory)?.sync_all()?;
            Ok(())
        })();
        if result.is_err() {
            let _ = std::fs::remove_file(temporary);
        }
        result.wrap_err_with(|| "cannot persist subscription credentials")
    }

    async fn access(
        &self,
        mut state: OwnedMutexGuard<CredentialState>,
        rejected: Option<&str>,
    ) -> Result<(String, Option<String>)> {
        if state.current.is_none() {
            state.current = Some(
                self.load()?
                    .ok_or_eyre("no subscription credentials; run tinyllm openai login")?,
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
            let response = self.client.post(format!("{}/oauth/token", self.issuer))
                .json(&json!({"client_id":CLIENT_ID,"grant_type":"refresh_token","refresh_token":credentials.refresh_token}))
                .send().await.map_err(|_| eyre::eyre!("cannot connect to OpenAI for subscription refresh"))?;
            let response = read_json(
                response,
                "subscription refresh failed; run tinyllm openai login again",
            )
            .await?;
            state.current = Some(Credentials::from_response(response, Some(credentials))?);
            state.unsaved = true;
            self.save(state.current.as_ref().unwrap())?;
            state.unsaved = false;
        }
        let credentials = state.current.as_ref().unwrap();
        Ok((
            credentials.access_token.clone(),
            Some(credentials.account_id.clone()),
        ))
    }

    async fn exchange(&self, code: &str, redirect: &str, verifier: &str) -> Result<Credentials> {
        let response = self
            .client
            .post(format!("{}/oauth/token", self.issuer))
            .form(&[
                ("grant_type", "authorization_code"),
                ("code", code),
                ("redirect_uri", redirect),
                ("client_id", CLIENT_ID),
                ("code_verifier", verifier),
            ])
            .send()
            .await
            .map_err(|_| eyre::eyre!("cannot connect to OpenAI for login exchange"))?;
        Credentials::from_response(
            read_json(response, "OpenAI login exchange failed").await?,
            None,
        )
    }

    pub async fn login(&self, device: bool) -> Result<()> {
        let credentials = if device {
            self.login_device().await?
        } else {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:1455").await
                .wrap_err_with(|| "cannot bind OAuth callback port 1455; close the other login or use --device-auth")?;
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
            let url = authorize_url(&self.issuer, &challenge, &state)?;
            println!("Open this URL in your browser to sign in:\n\n{url}\n");
            let code = tokio::time::timeout(Duration::from_secs(300), callback(listener, state))
                .await
                .wrap_err_with(|| "browser login timed out after five minutes")??;
            self.exchange(&code, REDIRECT_URI, &verifier).await?
        };
        self.save(&credentials)
    }

    async fn login_device(&self) -> Result<Credentials> {
        let response = self
            .client
            .post(format!("{}/api/accounts/deviceauth/usercode", self.issuer))
            .json(&json!({"client_id":CLIENT_ID}))
            .send()
            .await
            .map_err(|_| eyre::eyre!("cannot connect to OpenAI for device login"))?;
        let device: DeviceCode = read_json(
            response,
            "device login unavailable; enable it in ChatGPT security settings or use browser login",
        )
        .await?;
        let interval = device
            .interval
            .as_u64()
            .or_else(|| device.interval.as_str().and_then(|s| s.parse().ok()))
            .unwrap_or(5)
            .clamp(1, 60);
        let expires = device.expires_in.unwrap_or(900).clamp(1, 900);
        if !valid_secret(&device.user_code) || !valid_secret(&device.device_auth_id) {
            bail!("OpenAI returned an invalid device code");
        }
        println!(
            "Open {}/codex/device and enter {} (expires in {expires} seconds).",
            self.issuer, device.user_code
        );
        tokio::time::timeout(Duration::from_secs(expires), async {
            loop {
                let response = self.client.post(format!("{}/api/accounts/deviceauth/token", self.issuer))
                    .json(&json!({"device_auth_id":device.device_auth_id,"user_code":device.user_code}))
                    .send().await.map_err(|_| eyre::eyre!("cannot poll OpenAI device login"))?;
                if matches!(response.status().as_u16(), 403 | 404) {
                    tokio::time::sleep(Duration::from_secs(interval)).await;
                    continue;
                }
                let code: DeviceToken = read_json(response, "device authorization failed").await?;
                return self.exchange(&code.authorization_code, &format!("{}/deviceauth/callback", self.issuer), &code.code_verifier).await;
            }
        }).await.wrap_err_with(|| "device code expired; run tinyllm openai login again")?
    }

    pub fn logout(&self) -> Result<()> {
        for name in ["auth.json", "openai.json"] {
            match std::fs::remove_file(self.directory.join(name)) {
                Ok(()) => (),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => (),
                Err(e) => {
                    return Err(e).wrap_err_with(|| "cannot remove local subscription credentials");
                }
            }
        }
        Ok(())
    }
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

fn authorize_url(issuer: &str, challenge: &str, state: &str) -> Result<reqwest::Url> {
    let mut url = reqwest::Url::parse(&format!("{issuer}/oauth/authorize"))?;
    url.query_pairs_mut().extend_pairs([
        ("response_type", "code"),
        ("client_id", CLIENT_ID),
        ("redirect_uri", REDIRECT_URI),
        ("scope", "openid profile email offline_access"),
        ("code_challenge", challenge),
        ("code_challenge_method", "S256"),
        ("id_token_add_organizations", "true"),
        ("codex_cli_simplified_flow", "true"),
        ("state", state),
        ("originator", "tinyllm"),
    ]);
    Ok(url)
}

#[derive(Deserialize)]
struct DeviceCode {
    device_auth_id: String,
    #[serde(alias = "usercode")]
    user_code: String,
    #[serde(default)]
    interval: Value,
    expires_in: Option<u64>,
}

#[derive(Deserialize)]
struct DeviceToken {
    authorization_code: String,
    code_verifier: String,
}

#[derive(Deserialize)]
struct Callback {
    code: Option<String>,
    state: Option<String>,
    error: Option<String>,
}

async fn callback(listener: tokio::net::TcpListener, state: String) -> Result<String> {
    use std::future::IntoFuture;
    let (sender, mut receiver) = mpsc::channel(1);
    let app = Router::new().route(
        "/auth/callback",
        get(move |Query(query): Query<Callback>| {
            let sender = sender.clone();
            let state = state.clone();
            async move {
                if !query
                    .state
                    .as_ref()
                    .is_some_and(|s| bool::from(s.as_bytes().ct_eq(state.as_bytes())))
                {
                    return (
                        StatusCode::BAD_REQUEST,
                        "Login state mismatch. Return to the original login link.",
                    );
                }
                let code = if query.error.is_some() {
                    Err(eyre::eyre!("OpenAI login was denied or cancelled"))
                } else {
                    query
                        .code
                        .filter(|code| !code.is_empty() && code.len() <= 8192)
                        .ok_or_eyre("OAuth callback omitted the authorization code")
                };
                let _ = sender.try_send(code);
                (
                    StatusCode::OK,
                    "Authorization received. Return to the terminal for the login result.",
                )
            }
        }),
    );
    tokio::select! {
        code = receiver.recv() => code.ok_or_eyre("OAuth callback closed")?,
        result = axum::serve(listener, app).into_future() => {
            result.wrap_err_with(|| "OAuth callback server failed")?;
            bail!("OAuth callback server stopped before login");
        }
    }
}

#[cfg(test)]
#[path = "auth_tests.rs"]
mod tests;
