use crate::{
    config::Config,
    endpoints::{
        Endpoint, anthropic::AnthropicEndpoint, endpoint::error_response, openai::OpenAiEndpoint,
    },
    error::Error,
    models::ApiFormat,
    providers::registry::Registry,
};
use axum::{
    Router,
    extract::{DefaultBodyLimit, Request, State},
    http::StatusCode,
    middleware::{self, Next},
    response::Response,
};
use std::{sync::Arc, time::Instant};
use subtle::ConstantTimeEq;
use tokio::sync::Semaphore;
use tracing::Instrument;

pub struct AppState {
    pub config: Config,
    pub providers: Registry,
    pub permits: Option<Arc<Semaphore>>,
}

pub async fn router(config: Config) -> eyre::Result<Router> {
    config.validate()?;
    if config.server.auth_token.is_none() {
        tracing::warn!(
            bind = %config.server.bind,
            "gateway authentication is disabled; set server.auth_token to restrict access"
        );
    }
    let providers = Registry::new(&config).await?;
    let limit = config.server.max_request_bytes;
    let permits = config
        .server
        .max_concurrent_requests
        .map(|limit| Arc::new(Semaphore::new(limit)));
    let app = Arc::new(AppState {
        config,
        providers,
        permits,
    });
    Ok(Router::new()
        .merge(AnthropicEndpoint.router())
        .merge(OpenAiEndpoint.router())
        .fallback(|req: Request| async move {
            error_response(
                Error {
                    status: StatusCode::NOT_FOUND,
                    kind: "not_found_error",
                    message: "endpoint not implemented".into(),
                    headers: Box::default(),
                },
                format(&req),
            )
        })
        .layer(DefaultBodyLimit::max(limit))
        .layer(middleware::from_fn_with_state(app.clone(), authenticate))
        .layer(middleware::from_fn(log_request))
        .with_state(app))
}

fn format(request: &Request) -> ApiFormat {
    if request.uri().path().starts_with("/anthropic/") {
        ApiFormat::Anthropic
    } else {
        ApiFormat::Responses
    }
}

async fn authenticate(State(app): State<Arc<AppState>>, request: Request, next: Next) -> Response {
    let Some(expected) = app.config.server.auth_token.as_deref() else {
        return next.run(request).await;
    };
    let token = request
        .headers()
        .get("authorization")
        .and_then(|h| h.to_str().ok())
        .and_then(|h| h.strip_prefix("Bearer "))
        .or_else(|| {
            request
                .headers()
                .get("x-api-key")
                .and_then(|h| h.to_str().ok())
        });
    if !token.is_some_and(|token| bool::from(token.as_bytes().ct_eq(expected.as_bytes()))) {
        return error_response(
            Error {
                status: StatusCode::UNAUTHORIZED,
                kind: "authentication_error",
                message: "invalid local gateway token".into(),
                headers: Box::default(),
            },
            format(&request),
        );
    }
    next.run(request).await
}
async fn log_request(mut request: Request, next: Next) -> Response {
    let id = uuid::Uuid::new_v4();
    request.extensions_mut().insert(id);
    let span = tracing::info_span!("request", request_id = %id, method = %request.method(), path = request.uri().path());
    async move {
        let start = Instant::now();
        let response = next.run(request).await;
        tracing::info!(
            status = response.status().as_u16(),
            elapsed_ms = start.elapsed().as_millis() as u64,
            "response headers ready"
        );
        response
    }
    .instrument(span)
    .await
}
