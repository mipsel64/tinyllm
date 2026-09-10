use super::*;
use axum::{Json, extract::Form, routing::post};
use std::{
    collections::HashMap,
    sync::atomic::{AtomicUsize, Ordering},
};

fn directory() -> PathBuf {
    std::env::temp_dir().join(format!("tinyllm-auth-{}", uuid::Uuid::new_v4()))
}

#[test]
fn legacy_credentials_migrate_without_overwriting_new_credentials() {
    let path = directory();
    let session = Session::open(&path).unwrap();
    session
        .save(&Credentials {
            access_token: "current-access".into(),
            refresh_token: "refresh".into(),
            account_id: "account".into(),
            expires_at: now() + 3600,
        })
        .unwrap();
    let current = path.join("openai.json");
    let legacy = path.join("auth.json");
    assert!(current.is_file());
    std::fs::rename(&current, &legacy).unwrap();
    drop(session);
    let session = Session::open(&path).unwrap();
    assert!(current.is_file());
    assert!(!legacy.exists());
    assert_eq!(
        session.load().unwrap().unwrap().access_token,
        "current-access"
    );
    session
        .save(&Credentials {
            access_token: "new-access".into(),
            refresh_token: "new-refresh".into(),
            account_id: "account".into(),
            expires_at: now() + 3600,
        })
        .unwrap();
    std::fs::copy(&current, &legacy).unwrap();
    std::fs::write(&legacy, b"invalid legacy credentials").unwrap();
    drop(session);
    let session = Session::open(&path).unwrap();
    assert_eq!(session.load().unwrap().unwrap().access_token, "new-access");
    session.logout().unwrap();
    assert!(!current.exists());
    assert!(!legacy.exists());
    assert!(session.load().unwrap().is_none());
    drop(session);
    std::fs::remove_dir_all(path).unwrap();
}

fn token_response(access: &str) -> Value {
    let payload = URL_SAFE_NO_PAD.encode(
        json!({"https://api.openai.com/auth":{"chatgpt_account_id":"account"}}).to_string(),
    );
    json!({"access_token":access,"refresh_token":"refresh-new","id_token":format!("h.{payload}.s"),"expires_in":3600})
}

async fn server(router: Router) -> (String, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let task = tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    (url, task)
}

#[tokio::test]
async fn browser_callback_binds_state_and_exchanges_encoded_code() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let wait = tokio::spawn(callback(listener, "expected-state".into()));
    let client = reqwest::Client::new();
    assert_eq!(
        client
            .get(format!("http://{address}/favicon.ico"))
            .send()
            .await
            .unwrap()
            .status(),
        404
    );
    assert_eq!(
        client
            .get(format!(
                "http://{address}/auth/callback?state=wrong&code=ignored"
            ))
            .send()
            .await
            .unwrap()
            .status(),
        400
    );
    assert!(!wait.is_finished());
    assert_eq!(
        client
            .get(format!(
                "http://{address}/auth/callback?state=expected-state&code=a%2Bb%2Fc%3D"
            ))
            .send()
            .await
            .unwrap()
            .status(),
        200
    );
    let code = wait.await.unwrap().unwrap();
    assert_eq!(code, "a+b/c=");
    let (issuer, task) = server(Router::new().route(
        "/oauth/token",
        post(|Form(form): Form<HashMap<String, String>>| async move {
            assert_eq!(form["grant_type"], "authorization_code");
            assert_eq!(form["code"], "a+b/c=");
            assert_eq!(form["code_verifier"], "verifier");
            assert_eq!(form["redirect_uri"], REDIRECT_URI);
            assert_eq!(form["client_id"], CLIENT_ID);
            Json(token_response("access"))
        }),
    ))
    .await;
    let path = directory();
    let mut session = Session::open(&path).unwrap();
    session.issuer = issuer;
    let credentials = session
        .exchange(&code, REDIRECT_URI, "verifier")
        .await
        .unwrap();
    assert_eq!(credentials.account_id, "account");
    let url = authorize_url(ISSUER, "challenge", "state").unwrap();
    let params: HashMap<_, _> = url.query_pairs().collect();
    assert_eq!(params["code_challenge_method"], "S256");
    assert_eq!(params["originator"], "tinyllm");
    assert_eq!(params["scope"], "openid profile email offline_access");
    assert_eq!(params["state"], "state");
    assert_eq!(params["code_challenge"], "challenge");
    let listener = tokio::net::TcpListener::bind(address).await.unwrap();
    let wait = tokio::spawn(callback(listener, "state".into()));
    wait.abort();
    let _ = wait.await;
    assert!(tokio::net::TcpListener::bind(address).await.is_ok());
    task.abort();
    drop(session);
    std::fs::remove_dir_all(path).unwrap();
}

#[tokio::test]
async fn device_login_polls_persists_and_reopens_private_credentials() {
    let polls = Arc::new(AtomicUsize::new(0));
    let router = Router::new()
        .route("/api/accounts/deviceauth/usercode", post(|Json(body): Json<Value>| async move {
            assert_eq!(body["client_id"], CLIENT_ID);
            Json(json!({"device_auth_id":"device","user_code":"CODE","interval":"1","expires_in":30}))
        }))
        .route("/api/accounts/deviceauth/token", post({let polls=polls.clone(); move |Json(body): Json<Value>| {
            let polls=polls.clone(); async move {
                assert_eq!(body["device_auth_id"], "device");
                assert_eq!(body["user_code"], "CODE");
                if polls.fetch_add(1, Ordering::SeqCst) == 0 {
                    return (StatusCode::FORBIDDEN, Json(json!({})));
                }
                (StatusCode::OK, Json(json!({"authorization_code":"code","code_verifier":"device-verifier"})))
            }
        }}))
        .route("/oauth/token", post(|Form(form): Form<HashMap<String,String>>| async move {
            assert_eq!(form["code_verifier"], "device-verifier");
            assert!(form["redirect_uri"].ends_with("/deviceauth/callback"));
            Json(token_response("device-access"))
        }));
    let (issuer, task) = server(router).await;
    let path = directory();
    let mut session = Session::open(&path).unwrap();
    session.issuer = issuer;
    assert!(Session::open(&path).is_err());
    session.login(true).await.unwrap();
    assert_eq!(polls.load(Ordering::SeqCst), 2);
    drop(session);
    let session = Session::open(&path).unwrap();
    assert_eq!(
        session.load().unwrap().unwrap().access_token,
        "device-access"
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(
            std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o700
        );
        assert_eq!(
            std::fs::metadata(path.join("openai.json"))
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o600
        );
    }
    session.logout().unwrap();
    assert!(session.load().unwrap().is_none());
    session.logout().unwrap();
    task.abort();
    drop(session);
    std::fs::remove_dir_all(path).unwrap();
}

#[tokio::test]
async fn failed_persistence_keeps_the_rotated_refresh_token() {
    let calls = Arc::new(AtomicUsize::new(0));
    let (issuer, task) = server(Router::new().route(
        "/oauth/token",
        post({
            let calls = calls.clone();
            move || {
                let calls = calls.clone();
                async move {
                    calls.fetch_add(1, Ordering::SeqCst);
                    Json(token_response("new-access"))
                }
            }
        }),
    ))
    .await;
    let path = directory();
    let mut session = Session::open(&path).unwrap();
    session.issuer = issuer;
    session
        .save(&Credentials {
            access_token: "old-access".into(),
            refresh_token: "old-refresh".into(),
            account_id: "account".into(),
            expires_at: now() + 3600,
        })
        .unwrap();
    let session = Arc::new(session);
    let auth = Auth::Subscription(session.clone());
    auth.access(None).await.unwrap();
    std::fs::rename(path.join("openai.json"), path.join("previous.json")).unwrap();
    std::fs::create_dir(path.join("openai.json")).unwrap();
    assert!(
        auth.access(Some("old-access"))
            .await
            .err()
            .unwrap()
            .to_string()
            .contains("persist")
    );
    std::fs::remove_dir(path.join("openai.json")).unwrap();
    assert_eq!(auth.access(None).await.unwrap().0, "new-access");
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

#[tokio::test]
async fn cancelled_refresh_waiters_do_not_retry_failed_refresh() {
    let calls = Arc::new(AtomicUsize::new(0));
    let started = Arc::new(tokio::sync::Notify::new());
    let release = Arc::new(tokio::sync::Notify::new());
    let (issuer, task) = server(Router::new().route(
        "/oauth/token",
        post({
            let calls = calls.clone();
            let started = started.clone();
            let release = release.clone();
            move || {
                let calls = calls.clone();
                let started = started.clone();
                let release = release.clone();
                async move {
                    if calls.fetch_add(1, Ordering::SeqCst) == 0 {
                        started.notify_one();
                        release.notified().await;
                    }
                    StatusCode::BAD_REQUEST
                }
            }
        }),
    ))
    .await;
    let path = directory();
    let mut session = Session::open(&path).unwrap();
    session.issuer = issuer;
    session
        .save(&Credentials {
            access_token: "expired-access".into(),
            refresh_token: "refresh".into(),
            account_id: "account".into(),
            expires_at: 0,
        })
        .unwrap();
    let session = Arc::new(session);
    let auth = Arc::new(Auth::Subscription(session.clone()));
    let first = {
        let auth = auth.clone();
        tokio::spawn(async move { auth.access(None).await })
    };
    tokio::time::timeout(Duration::from_secs(2), started.notified())
        .await
        .unwrap();
    let (ready, mut waiting) = mpsc::channel(8);
    let mut waiters = Vec::new();
    for _ in 0..8 {
        let auth = auth.clone();
        let ready = ready.clone();
        waiters.push(tokio::spawn(async move {
            let access = auth.access(None);
            tokio::pin!(access);
            assert!(futures::poll!(&mut access).is_pending());
            ready.send(()).await.unwrap();
            access.await
        }));
    }
    drop(ready);
    for _ in 0..8 {
        waiting.recv().await.unwrap();
    }
    for waiter in waiters {
        waiter.abort();
        assert!(waiter.await.unwrap_err().is_cancelled());
    }
    {
        let drained = session.credentials.lock();
        tokio::pin!(drained);
        assert!(futures::poll!(&mut drained).is_pending());
        release.notify_one();
        assert!(first.await.unwrap().is_err());
        let _guard = tokio::time::timeout(Duration::from_secs(2), drained)
            .await
            .unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }
    task.abort();
    drop(auth);
    drop(session);
    std::fs::remove_dir_all(path).unwrap();
}

#[tokio::test]
async fn refresh_is_shared_survives_cancellation_and_redacts_failures() {
    let calls = Arc::new(AtomicUsize::new(0));
    let (issuer, task) = server(Router::new().route(
        "/oauth/token",
        post({
            let calls = calls.clone();
            move |Json(body): Json<Value>| {
                let calls = calls.clone();
                async move {
                    assert_eq!(body["grant_type"], "refresh_token");
                    assert_eq!(body["client_id"], CLIENT_ID);
                    assert!(body.get("scope").is_none());
                    let call = calls.fetch_add(1, Ordering::SeqCst);
                    tokio::time::sleep(Duration::from_millis(80)).await;
                    if call >= 2 {
                        return (
                            StatusCode::BAD_REQUEST,
                            Json(json!({"error":"do not expose refresh-new"})),
                        );
                    }
                    let mut response = token_response(if call == 0 {
                        "access-new"
                    } else {
                        "access-newer"
                    });
                    if call == 1 {
                        response.as_object_mut().unwrap().remove("refresh_token");
                    }
                    (StatusCode::OK, Json(response))
                }
            }
        }),
    ))
    .await;
    let path = directory();
    let mut session = Session::open(&path).unwrap();
    session.issuer = issuer;
    session
        .save(&Credentials {
            access_token: "access-old".into(),
            refresh_token: "refresh-old".into(),
            account_id: "account".into(),
            expires_at: 0,
        })
        .unwrap();
    let session = Arc::new(session);
    let auth = Arc::new(Auth::Subscription(session.clone()));
    let a = {
        let auth = auth.clone();
        tokio::spawn(async move { auth.access(None).await })
    };
    let b = {
        let auth = auth.clone();
        tokio::spawn(async move { auth.access(None).await })
    };
    tokio::time::timeout(Duration::from_secs(2), async {
        while calls.load(Ordering::SeqCst) == 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    a.abort();
    assert_eq!(b.await.unwrap().unwrap().0, "access-new");
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    let (first, second) = tokio::join!(
        auth.access(Some("access-new")),
        auth.access(Some("access-new"))
    );
    assert_eq!(first.unwrap().0, "access-newer");
    assert_eq!(second.unwrap().0, "access-newer");
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert_eq!(
        session.load().unwrap().unwrap().refresh_token,
        "refresh-new"
    );
    let error = auth
        .access(Some("access-newer"))
        .await
        .err()
        .unwrap()
        .to_string();
    assert!(error.contains("login"));
    assert!(!error.contains("refresh-new"));
    assert_eq!(
        session.load().unwrap().unwrap().access_token,
        "access-newer"
    );
    let mut changed = token_response("access-invalid");
    changed["id_token"] = json!(format!(
        "h.{}.s",
        URL_SAFE_NO_PAD.encode(br#"{"chatgpt_account_id":"other"}"#)
    ));
    assert!(
        Credentials::from_response(
            serde_json::from_value(changed).unwrap(),
            session.load().unwrap().as_ref()
        )
        .is_err()
    );
    let mut missing = token_response("access");
    missing.as_object_mut().unwrap().remove("refresh_token");
    assert!(Credentials::from_response(serde_json::from_value(missing).unwrap(), None).is_err());
    task.abort();
    drop(auth);
    drop(session);
    let reopened = Session::open(&path).unwrap();
    assert_eq!(
        reopened.load().unwrap().unwrap().access_token,
        "access-newer"
    );
    drop(reopened);
    std::fs::remove_dir_all(path).unwrap();
}
