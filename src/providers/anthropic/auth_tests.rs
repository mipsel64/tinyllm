use super::*;
use axum::{Json, routing::post};
use serde_json::{Value, json};
use std::{
    collections::HashMap,
    sync::atomic::{AtomicUsize, Ordering},
};

fn directory() -> PathBuf {
    std::env::temp_dir().join(format!("tinyllm-anthropic-auth-{}", uuid::Uuid::new_v4()))
}

async fn server(router: Router) -> (String, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let task = tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    (url, task)
}

fn response(access: &str, refresh: Option<&str>) -> Value {
    let mut value = json!({"access_token":access,"expires_in":3600});
    if let Some(refresh) = refresh {
        value["refresh_token"] = json!(refresh);
    }
    value
}

#[tokio::test]
async fn browser_callback_checks_state_and_reports_missing_or_denied() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let wait = tokio::spawn(callback(listener, "expected-state".into()));
    let client = reqwest::Client::new();
    assert_eq!(
        client
            .get(format!(
                "http://{address}/callback?state=wrong&code=ignored"
            ))
            .send()
            .await
            .unwrap()
            .status(),
        StatusCode::BAD_REQUEST
    );
    assert!(!wait.is_finished());
    assert_eq!(
        client
            .get(format!("http://{address}/callback?state=expected-state"))
            .send()
            .await
            .unwrap()
            .status(),
        StatusCode::BAD_REQUEST
    );
    assert!(
        wait.await
            .unwrap()
            .unwrap_err()
            .to_string()
            .contains("omitted")
    );

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let wait = tokio::spawn(callback(listener, "expected-state".into()));
    assert_eq!(
        client
            .get(format!(
                "http://{address}/callback?state=expected-state&error=denied"
            ))
            .send()
            .await
            .unwrap()
            .status(),
        StatusCode::BAD_REQUEST
    );
    assert!(
        wait.await
            .unwrap()
            .unwrap_err()
            .to_string()
            .contains("denied")
    );

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let wait = tokio::spawn(callback(listener, "expected-state".into()));
    assert_eq!(
        client
            .get(format!(
                "http://{address}/callback?state=expected-state&code=a%2Bb%2Fc%3D"
            ))
            .send()
            .await
            .unwrap()
            .status(),
        StatusCode::OK
    );
    assert_eq!(
        wait.await.unwrap().unwrap(),
        ("a+b/c=".into(), "expected-state".into())
    );
}

#[tokio::test]
async fn authorization_and_exchange_use_pkce_state_and_minimal_scopes() {
    let (token_url, task) = server(Router::new().route(
        "/v1/oauth/token",
        post(|Json(body): Json<Value>| async move {
            assert_eq!(body["grant_type"], "authorization_code");
            assert_eq!(body["client_id"], CLIENT_ID);
            assert_eq!(body["code"], "code");
            assert_eq!(body["state"], "state");
            assert_eq!(body["redirect_uri"], REDIRECT_URI);
            assert_eq!(body["code_verifier"], "verifier");
            Json(response("access", Some("refresh")))
        }),
    ))
    .await;
    let path = directory();
    let mut session = Session::open(&path).unwrap();
    session.token_url = format!("{token_url}/v1/oauth/token");
    let credentials = session.exchange("code", "state", "verifier").await.unwrap();
    assert_eq!(credentials.access_token, "access");
    assert_eq!(credentials.refresh_token, "refresh");
    assert!(credentials.expires_at > now());

    let url = authorize_url(AUTHORIZE_URL, "challenge", "state").unwrap();
    assert_eq!(url.as_str().split('?').next().unwrap(), AUTHORIZE_URL);
    let params: HashMap<_, _> = url.query_pairs().collect();
    assert_eq!(params["client_id"], CLIENT_ID);
    assert_eq!(params["redirect_uri"], REDIRECT_URI);
    assert_eq!(params["scope"], SCOPES);
    assert_eq!(params["scope"], "user:profile user:inference");
    assert_eq!(params["code_challenge"], "challenge");
    assert_eq!(params["code_challenge_method"], "S256");
    assert_eq!(params["state"], "state");
    assert_eq!(params["code"], "true");
    task.abort();
    drop(session);
    std::fs::remove_dir_all(path).unwrap();
}

#[tokio::test]
async fn refresh_rotates_once_for_concurrent_and_cancelled_waiters() {
    let calls = Arc::new(AtomicUsize::new(0));
    let started = Arc::new(tokio::sync::Notify::new());
    let (token_url, task) = server(Router::new().route(
        "/v1/oauth/token",
        post({
            let calls = calls.clone();
            let started = started.clone();
            move |Json(body): Json<Value>| {
                let calls = calls.clone();
                let started = started.clone();
                async move {
                    assert_eq!(body["grant_type"], "refresh_token");
                    assert_eq!(body["client_id"], CLIENT_ID);
                    assert_eq!(body["refresh_token"], "refresh-old");
                    assert!(body.get("scope").is_none());
                    calls.fetch_add(1, Ordering::SeqCst);
                    started.notify_one();
                    tokio::time::sleep(Duration::from_millis(80)).await;
                    Json(response("access-new", Some("refresh-new")))
                }
            }
        }),
    ))
    .await;
    let path = directory();
    let mut session = Session::open(&path).unwrap();
    session.token_url = format!("{token_url}/v1/oauth/token");
    session
        .save(&Credentials {
            access_token: "access-old".into(),
            refresh_token: "refresh-old".into(),
            expires_at: 0,
        })
        .unwrap();
    let session = Arc::new(session);
    let auth = Arc::new(Auth::Subscription(session.clone()));
    let cancelled = {
        let auth = auth.clone();
        tokio::spawn(async move { auth.access(None).await })
    };
    tokio::time::timeout(Duration::from_secs(2), started.notified())
        .await
        .unwrap();
    let waiting = {
        let auth = auth.clone();
        tokio::spawn(async move { auth.access(None).await })
    };
    cancelled.abort();
    assert!(cancelled.await.unwrap_err().is_cancelled());
    assert_eq!(waiting.await.unwrap().unwrap(), "access-new");
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    let saved = session.load().unwrap().unwrap();
    assert_eq!(saved.access_token, "access-new");
    assert_eq!(saved.refresh_token, "refresh-new");
    task.abort();
    drop(auth);
    drop(session);
    std::fs::remove_dir_all(path).unwrap();
}

#[tokio::test]
async fn failed_persistence_keeps_rotated_credentials_in_memory_for_retry() {
    let calls = Arc::new(AtomicUsize::new(0));
    let (token_url, task) = server(Router::new().route(
        "/v1/oauth/token",
        post({
            let calls = calls.clone();
            move |Json(body): Json<Value>| {
                let calls = calls.clone();
                async move {
                    calls.fetch_add(1, Ordering::SeqCst);
                    assert_eq!(body["refresh_token"], "refresh-old");
                    Json(response("access-new", Some("refresh-new")))
                }
            }
        }),
    ))
    .await;
    let path = directory();
    let mut session = Session::open(&path).unwrap();
    session.token_url = format!("{token_url}/v1/oauth/token");
    session
        .save(&Credentials {
            access_token: "access-old".into(),
            refresh_token: "refresh-old".into(),
            expires_at: now() + 3600,
        })
        .unwrap();
    let session = Arc::new(session);
    let auth = Auth::Subscription(session.clone());
    assert_eq!(auth.access(None).await.unwrap(), "access-old");
    std::fs::rename(session.path(), path.join("previous.json")).unwrap();
    std::fs::create_dir(session.path()).unwrap();
    let error = auth
        .access(Some("access-old"))
        .await
        .unwrap_err()
        .to_string();
    assert!(error.contains("persist"));
    assert!(!error.contains("refresh-new"));
    std::fs::remove_dir(session.path()).unwrap();
    assert_eq!(auth.access(None).await.unwrap(), "access-new");
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        session.load().unwrap().unwrap().refresh_token,
        "refresh-new"
    );
    task.abort();
    drop(auth);
    drop(session);
    std::fs::remove_dir_all(path).unwrap();
}

#[test]
fn credentials_are_private_locked_and_logout_is_idempotent() {
    let missing = directory();
    let error = match Auth::open(&AnthropicAuth::Subscription(
        super::super::models::SubscriptionOptions {
            credentials_dir: missing.clone(),
        },
    )) {
        Ok(_) => panic!("missing credentials unexpectedly opened"),
        Err(error) => error.to_string(),
    };
    assert!(error.contains("no subscription credentials"));
    std::fs::remove_dir_all(missing).unwrap();

    let path = directory();
    let session = Session::open(&path).unwrap();
    assert!(Session::open(&path).is_err());
    session
        .save(&Credentials {
            access_token: "access".into(),
            refresh_token: "refresh".into(),
            expires_at: now() + 3600,
        })
        .unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(
            std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o700
        );
        assert_eq!(
            std::fs::metadata(session.path())
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o600
        );
        std::fs::set_permissions(session.path(), std::fs::Permissions::from_mode(0o644)).unwrap();
        assert!(session.load().is_err());
        std::fs::set_permissions(session.path(), std::fs::Permissions::from_mode(0o600)).unwrap();
    }
    session.logout().unwrap();
    session.logout().unwrap();
    assert!(session.load().unwrap().is_none());
    drop(session);

    #[cfg(unix)]
    {
        use std::os::unix::fs::{PermissionsExt, symlink};
        let unsafe_path = directory();
        std::fs::create_dir(&unsafe_path).unwrap();
        std::fs::set_permissions(&unsafe_path, std::fs::Permissions::from_mode(0o755)).unwrap();
        assert!(Session::open(&unsafe_path).is_err());
        std::fs::remove_dir(&unsafe_path).unwrap();
        symlink(&path, &unsafe_path).unwrap();
        assert!(Session::open(&unsafe_path).is_err());
        std::fs::remove_file(&unsafe_path).unwrap();
    }
    std::fs::remove_dir_all(path).unwrap();
}

#[cfg(unix)]
#[test]
fn dangling_lock_symlink_is_rejected_without_creating_its_target() {
    use std::os::unix::fs::{DirBuilderExt, symlink};

    let path = directory();
    let mut builder = std::fs::DirBuilder::new();
    builder.mode(0o700).create(&path).unwrap();
    let target = path.join("lock-target");
    symlink(&target, path.join("anthropic.lock")).unwrap();

    assert!(Session::open(&path).is_err());
    assert!(!target.exists());
    std::fs::remove_dir_all(path).unwrap();
}

#[tokio::test]
async fn oauth_failures_are_bounded_and_do_not_echo_response_bodies() {
    let (token_url, task) = server(Router::new().route(
        "/v1/oauth/token",
        post(|| async {
            (
                StatusCode::BAD_REQUEST,
                "secret-access secret-refresh secret-code",
            )
        }),
    ))
    .await;
    let path = directory();
    let mut session = Session::open(&path).unwrap();
    session.token_url = format!("{token_url}/v1/oauth/token");
    let error = match session.exchange("secret-code", "state", "verifier").await {
        Ok(_) => panic!("exchange unexpectedly succeeded"),
        Err(error) => error.to_string(),
    };
    assert!(error.contains("HTTP 400"));
    for secret in ["secret-access", "secret-refresh", "secret-code"] {
        assert!(!error.contains(secret));
    }
    task.abort();
    drop(session);
    std::fs::remove_dir_all(path).unwrap();
}
