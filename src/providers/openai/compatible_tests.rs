use super::*;
use crate::{models::ReasoningEffort, providers::openai::models::ServiceTier};
use serde_json::json;

fn model() -> super::super::models::Model {
    super::super::models::Model {
        id: "gpt-native".into(),
        reasoning_effort: Some(ReasoningEffort::High),
    }
}

#[test]
fn native_options_and_reasoning_are_preserved() {
    let input = json!({"model":"openai/gpt-native","stream":false,"input":[{"type":"reasoning","id":"rs_1","encrypted_content":"opaque","summary":[]}],"reasoning":{"effort":"low","summary":"auto"},"include":["message.output_text.logprobs"],"tools":[{"type":"web_search"}],"text":{"verbosity":"low"}});
    let body = request::native(input.clone(), &model(), false).unwrap();
    for field in ["input", "reasoning", "include", "tools", "text"] {
        assert_eq!(body[field], input[field]);
    }
    assert_eq!(body["model"], "gpt-native");
    assert!(request::native(json!({"input":"hello","store":true}), &model(), false).is_err());
    assert!(request::native(json!({"input":"hello","background":true}), &model(), false).is_err());
    assert!(
        request::native(
            json!({"input":"hello","max_output_tokens":20}),
            &model(),
            true
        )
        .is_err()
    );
    let subscription = request::native(json!({"input":"hello"}), &model(), true).unwrap();
    assert_eq!(subscription["stream"], true);
    assert_eq!(subscription["instructions"], "");
    assert_eq!(
        subscription["include"],
        json!(["reasoning.encrypted_content"])
    );
}

#[test]
fn chat_converts_roles_images_tools_and_rejects_unsupported_controls() {
    let original = json!({"model":"openai/gpt-native","messages":[{"role":"system","content":"system"},{"role":"developer","content":"developer"},{"role":"user","content":[{"type":"text","text":"describe"},{"type":"image_url","image_url":{"url":"https://example.com/image.png","detail":"low"}}]}],"tools":[{"type":"function","function":{"name":"lookup","description":"lookup a value","parameters":{"type":"object"},"strict":false}}],"tool_choice":{"type":"function","function":{"name":"lookup"}},"response_format":{"type":"json_schema","json_schema":{"name":"reply","schema":{"type":"object"},"strict":true}},"reasoning_effort":"low","max_completion_tokens":20});
    let converted = request::chat(&original, &model(), false, &Default::default()).unwrap();
    assert_eq!(converted["input"][1]["role"], "developer");
    assert_eq!(converted["input"][2]["content"][1]["detail"], "low");
    assert_eq!(converted["tools"][0]["name"], "lookup");
    assert_eq!(
        converted["tool_choice"],
        json!({"type":"function","name":"lookup"})
    );
    assert_eq!(converted["text"]["format"]["name"], "reply");
    assert_eq!(converted["reasoning"]["effort"], "low");
    assert_eq!(converted["max_output_tokens"], 20);
    for (key, value) in [
        ("n", json!(2)),
        ("logprobs", json!(true)),
        ("audio", json!({"voice":"alloy"})),
        ("stop", json!(["end"])),
    ] {
        let mut invalid = original.clone();
        invalid[key] = value;
        assert!(
            request::chat(&invalid, &model(), false, &Default::default()).is_err(),
            "{key}"
        );
    }
}

fn completed(output: Value) -> Value {
    json!({"id":"resp_1","object":"response","created_at":123,"status":"completed","output":output,"usage":{"input_tokens":10,"output_tokens":7,"total_tokens":17,"input_tokens_details":{"cached_tokens":3},"output_tokens_details":{"reasoning_tokens":2}}})
}

#[tokio::test]
async fn chat_reference_roundtrip_restores_reasoning_and_two_tools() {
    let directory = std::env::temp_dir().join(format!("tinyllm-chat-{}", uuid::Uuid::new_v4()));
    let store = super::super::state::Store::open(directory.clone(), 1_000_000, 100_000)
        .await
        .unwrap();
    let native = completed(
        json!([{"id":"rs_1","type":"reasoning","encrypted_content":"opaque","summary":[]},{"id":"fc_1","type":"function_call","call_id":"call_a","name":"lookup","arguments":"{\"x\":1}"},{"id":"fc_2","type":"function_call","call_id":"call_b","name":"lookup","arguments":"{\"x\":2}"}]),
    );
    let reference = super::super::state::Store::reference();
    let response = chat_response(&native, "openai/gpt-native", &reference).unwrap();
    let message = response["choices"][0]["message"].clone();
    store
        .save(
            &reference,
            "gpt-native",
            &native,
            request::continuation(&message).unwrap(),
        )
        .await
        .unwrap();
    let mut streamed_message = message.clone();
    streamed_message["content"] = json!("");
    assert_eq!(
        request::continuation(&streamed_message).unwrap(),
        request::continuation(&message).unwrap()
    );
    let mut history = json!({"model":"openai/gpt-native","messages":[{"role":"user","content":"lookup both"},message,{"role":"tool","tool_call_id":"call_a","content":"first"},{"role":"tool","tool_call_id":"call_b","content":"second"}]});
    let restored = store
        .restore(&request::history(&history).unwrap(), "gpt-native")
        .await
        .unwrap();
    let body = request::chat(&history, &model(), false, &restored).unwrap();
    assert_eq!(body["input"][1], native["output"][0]);
    assert_eq!(body["input"][4]["call_id"], "call_a");
    assert_eq!(body["input"][5]["call_id"], "call_b");
    assert_eq!(
        response["usage"]["prompt_tokens_details"]["cached_tokens"],
        3
    );
    history["messages"][1]["tool_calls"][0]["function"]["arguments"] = json!("{}");
    assert!(
        store
            .restore(&request::history(&history).unwrap(), "gpt-native")
            .await
            .is_err()
    );
    history["messages"][1]
        .as_object_mut()
        .unwrap()
        .remove("reasoning_details");
    assert!(request::history(&history).is_err());
    drop(store);
    tokio::fs::remove_dir_all(directory).await.unwrap();
}

#[test]
fn native_sse_preserves_hosted_items_and_repairs_sparse_completion() {
    let mut tracker = stream::Native::new(true);
    tracker.accept(json!({"type":"response.created","response":{"id":"resp_1","created_at":123,"model":"gpt-native","metadata":{"a":"b"}}})).unwrap();
    tracker.accept(json!({"type":"response.output_item.added","output_index":0,"item":{"id":"ws_1","type":"web_search_call","status":"in_progress"}})).unwrap();
    let custom =
        json!({"type":"response.web_search_call.searching","output_index":0,"item_id":"ws_1"});
    assert_eq!(tracker.accept(custom.clone()).unwrap(), custom);
    let item = json!({"id":"ws_1","type":"web_search_call","status":"completed","action":{"type":"search","query":"Rust"}});
    tracker
        .accept(json!({"type":"response.output_item.done","output_index":0,"item":item}))
        .unwrap();
    let event = tracker.accept(json!({"type":"response.completed","response":{"id":"resp_1","output":[],"usage":{"input_tokens":1,"output_tokens":2}}})).unwrap();
    assert_eq!(event["response"]["output"][0], item);
    assert_eq!(event["response"]["metadata"], json!({"a":"b"}));
    assert_eq!(event["response"]["status"], "completed");
    assert!(stream::Native::new(false).finish().is_err());
    assert!(
        stream::Native::new(false)
            .accept(json!({"type":"response.failed","response":{"error":{"message":"failed"}}}))
            .is_err()
    );
}

#[test]
fn chat_stream_keeps_fragmented_tool_arguments_separate() {
    let mut translator = stream::Chat::new("openai/gpt-native".into());
    translator
        .accept(&json!({"type":"response.created","response":{"id":"resp_1","created_at":123}}))
        .unwrap();
    let mut chunks = Vec::new();
    for (index, id) in [(0, "call_a"), (1, "call_b")] {
        chunks.extend(translator.accept(&json!({"type":"response.output_item.added","output_index":index,"item":{"id":format!("fc_{index}"),"type":"function_call","call_id":id,"name":"lookup","arguments":""}})).unwrap());
    }
    for (index, delta) in [(0, "{\"x\":"), (1, "{\"x\":2"), (0, "1}"), (1, "}")] {
        chunks.extend(translator.accept(&json!({"type":"response.function_call_arguments.delta","output_index":index,"delta":delta})).unwrap());
    }
    let arguments = |index| {
        chunks
            .iter()
            .filter_map(|c| c["choices"][0]["delta"]["tool_calls"].as_array())
            .flatten()
            .filter(|c| c["index"] == index)
            .map(|c| c["function"]["arguments"].as_str().unwrap_or(""))
            .collect::<String>()
    };
    assert_eq!(arguments(0), "{\"x\":1}");
    assert_eq!(arguments(1), "{\"x\":2}");
    assert_eq!(
        chunks[0]["choices"][0]["delta"]["tool_calls"][0]["id"],
        "call_a"
    );
    assert_eq!(
        chunks[1]["choices"][0]["delta"]["tool_calls"][0]["id"],
        "call_b"
    );
}

#[test]
fn native_sse_rejects_removed_streamed_content() {
    let mut tracker = stream::Native::new(false);
    tracker
        .accept(json!({"type":"response.created","response":{"id":"resp_1"}}))
        .unwrap();
    tracker.accept(json!({"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"msg_1","role":"assistant","content":[]}})).unwrap();
    tracker.accept(json!({"type":"response.content_part.added","output_index":0,"content_index":0,"part":{"type":"output_text","text":""}})).unwrap();
    tracker.accept(json!({"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":"visible text"})).unwrap();
    assert!(tracker.accept(json!({"type":"response.output_item.done","output_index":0,"item":{"type":"message","id":"msg_1","role":"assistant","content":[]}})).is_err());
}

#[tokio::test]
async fn subscription_json_and_sse_accept_missing_content_type_and_keep_reasoning() {
    use super::super::models::{Config, ModelOptions, OpenAiAuth, SubscriptionOptions};
    use axum::{Json, Router, response::IntoResponse, routing::post};
    use std::sync::Arc;
    let reasoning =
        json!({"id":"rs_1","type":"reasoning","encrypted_content":"opaque","summary":[]});
    let events = [
        json!({"type":"response.created","response":{"id":"resp_1","created_at":123,"model":"gpt-native"}}),
        json!({"type":"response.output_item.added","output_index":0,"item":{"id":"rs_1","type":"reasoning","summary":[]}}),
        json!({"type":"response.output_item.done","output_index":0,"item":reasoning}),
        json!({"type":"response.completed","response":{"id":"resp_1","output":[],"usage":{"input_tokens":3,"output_tokens":2}}}),
    ];
    let raw: String = events
        .iter()
        .map(|event| {
            format!(
                "event: {}\ndata: {event}\n\n",
                event["type"].as_str().unwrap()
            )
        })
        .collect();
    let app = Router::new().route(
        "/responses",
        post(move |headers: HeaderMap, Json(body): Json<Value>| {
            let raw = raw.clone();
            async move {
                assert_eq!(headers["authorization"], "Bearer fixture-key");
                assert_eq!(headers["chatgpt-account-id"], "account");
                assert_eq!(body["stream"], true);
                assert_eq!(body["service_tier"], "priority");
                assert_eq!(body["instructions"], "");
                assert_eq!(body["include"], json!(["reasoning.encrypted_content"]));
                assert!(body["input"].is_array());
                let mut response = raw.into_response();
                if body["input"][0]["content"][0]["text"] != "wrong-content-type" {
                    response.headers_mut().remove("content-type");
                }
                response
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let directory =
        std::env::temp_dir().join(format!("tinyllm-chat-http-{}", uuid::Uuid::new_v4()));
    let auth_dir = directory.join("auth");
    std::fs::create_dir_all(&auth_dir).unwrap();
    let path = auth_dir.join("openai.json");
    std::fs::write(&path,json!({"access_token":"fixture-key","refresh_token":"refresh","account_id":"account","expires_at":u64::MAX}).to_string()).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&auth_dir, std::fs::Permissions::from_mode(0o700)).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
    }
    let server = crate::config::Server::default();
    let store = Arc::new(
        Store::open(directory.join("state"), 1_000_000, 100_000)
            .await
            .unwrap(),
    );
    let provider = OpenAiProvider::new(
        Config {
            base_url: base,
            auth: OpenAiAuth::Subscription(SubscriptionOptions {
                credentials_dir: auth_dir,
            }),
            organization: None,
            project: None,
            models: [(
                "gpt-native".into(),
                ModelOptions {
                    service_tier: Some(ServiceTier::Fast),
                    ..Default::default()
                },
            )]
            .into(),
        },
        http::client(&server).unwrap(),
        server,
        store.clone(),
    )
    .unwrap();
    for format in [ApiFormat::Responses, ApiFormat::ChatCompletions] {
        for streaming in [false, true] {
            let mut body = json!({"model":"codex/gpt-native","stream":streaming});
            if format == ApiFormat::Responses {
                body["input"] = json!("hello");
            } else {
                body["messages"] = json!([{"role":"user","content":"hello"}]);
            }
            let result = execute(
                &provider,
                ApiRequest::parse(format, body).unwrap(),
                context(),
            )
            .await
            .unwrap();
            match result.body {
                ResponseBody::Json(response) => {
                    assert_eq!(response["model"], "codex/gpt-native");
                    if format == ApiFormat::Responses {
                        assert_eq!(response["output"][0], reasoning);
                    } else {
                        assert_eq!(response["usage"]["prompt_tokens"], 3);
                        let message = response["choices"][0]["message"].clone();
                        let history = request::history(&json!({"messages":[message]})).unwrap();
                        let restored = store
                            .restore_scoped(&history, "codex", "gpt-native")
                            .await
                            .unwrap();
                        assert_eq!(restored[&0][0], reasoning);
                    }
                }
                ResponseBody::Stream(mut events) => {
                    let mut terminal = false;
                    while let Some(event) = events.next().await {
                        match event.unwrap() {
                            ApiEvent::Responses(value) if value["type"] == "response.completed" => {
                                assert_eq!(value["response"]["output"][0], reasoning);
                                terminal = true;
                            }
                            ApiEvent::ChatCompletions(value)
                                if !value["choices"][0]["finish_reason"].is_null() =>
                            {
                                assert!(
                                    value["choices"][0]["delta"]["reasoning_details"].is_array()
                                );
                                terminal = true;
                            }
                            _ => {}
                        }
                    }
                    assert!(terminal);
                }
            }
        }
    }
    let invalid = ApiRequest::parse(
        ApiFormat::Responses,
        json!({"model":"codex/gpt-native","input":"wrong-content-type"}),
    )
    .unwrap();
    assert!(execute(&provider, invalid, context()).await.is_err());
    drop(provider);
    drop(store);
    task.abort();
    tokio::fs::remove_dir_all(directory).await.unwrap();
}

fn context() -> RequestContext {
    RequestContext {
        request_id: "test".into(),
        provider: "codex".into(),
        model: "gpt-native".into(),
        public_model: "codex/gpt-native".into(),
        headers: HeaderMap::new(),
        query: None,
    }
}

#[test]
fn subscription_string_input_becomes_one_user_message_without_changing_api_keys() {
    for input in ["Reply with hello.", "Keep\nall Unicode: Việt Nam 🦀", ""] {
        let original = json!({"input":input,"instructions":"keep these instructions"});
        let subscription = request::native(original.clone(), &model(), true).unwrap();
        assert_eq!(
            subscription["input"],
            json!([{"role":"user","content":[{"type":"input_text","text":input}]}])
        );
        assert_eq!(subscription["instructions"], original["instructions"]);
        assert_eq!(
            request::native(original, &model(), false).unwrap()["input"],
            input
        );
    }
    let input = json!([{"type":"reasoning","id":"rs_1","encrypted_content":"opaque","summary":[]},{"role":"system","content":"system instruction"},{"role":"developer","content":"developer instruction"},{"type":"function_call","call_id":"call_1","name":"lookup","arguments":"{}"}]);
    assert_eq!(
        request::native(json!({"input":input}), &model(), true).unwrap()["input"],
        input
    );
}

#[test]
fn chat_nullable_defaults_keep_client_reasoning_and_nonstrict_tools() {
    let body = json!({"messages":[{"role":"user","content":"hello"}],"reasoning_effort":null,"tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object"},"strict":null}}]});
    let converted = request::chat(&body, &model(), false, &Default::default()).unwrap();
    assert_eq!(converted["reasoning"]["effort"], Value::Null);
    assert_eq!(converted["tools"][0]["strict"], false);
}

#[test]
fn chat_usage_rejects_impossible_cache_or_reasoning_details() {
    let mut response = completed(json!([]));
    response["usage"]["input_tokens_details"]["cached_tokens"] = json!(11);
    assert!(chat_response(&response, "codex/gpt-native", "tinyllm:v1:reference").is_err());
    response["usage"]["input_tokens_details"]["cached_tokens"] = json!(0);
    response["usage"]["output_tokens_details"]["reasoning_tokens"] = json!(8);
    assert!(chat_response(&response, "codex/gpt-native", "tinyllm:v1:reference").is_err());
}

#[tokio::test]
async fn api_key_chat_json_and_fragmented_sse_roundtrip_two_tools() {
    use super::super::models::{Config, OpenAiAuth};
    use axum::{Json, Router, body::Body, response::IntoResponse, routing::post};
    use std::sync::Arc;
    let native = completed(json!([
        {"id":"rs_1","type":"reasoning","encrypted_content":"opaque","summary":[]},
        {"id":"fc_1","type":"function_call","call_id":"call_a","name":"lookup","arguments":"{\"x\":1}"},
        {"id":"fc_2","type":"function_call","call_id":"call_b","name":"lookup","arguments":"{\"x\":2}"}
    ]));
    let mut events = vec![
        json!({"type":"response.created","response":{"id":"resp_1","created_at":123,"model":"gpt-native"}}),
    ];
    for (index, item) in native["output"].as_array().unwrap().iter().enumerate() {
        let mut initial = item.clone();
        if item["type"] == "function_call" {
            initial["arguments"] = json!("");
        }
        events
            .push(json!({"type":"response.output_item.added","output_index":index,"item":initial}));
    }
    for (index, delta) in [(1, "{\"x\":"), (2, "{\"x\":2"), (1, "1}"), (2, "}")] {
        events.push(json!({"type":"response.function_call_arguments.delta","output_index":index,"delta":delta}));
    }
    for (index, item) in native["output"].as_array().unwrap().iter().enumerate() {
        events.push(json!({"type":"response.output_item.done","output_index":index,"item":item}));
    }
    events.push(json!({"type":"response.completed","response":native}));
    let raw: String = events
        .iter()
        .map(|event| format!("data: {event}\n\n"))
        .collect();
    let expected = native.clone();
    let app = Router::new().route(
        "/responses",
        post(move |headers: HeaderMap, Json(body): Json<Value>| {
            let raw = raw.clone();
            let native = native.clone();
            async move {
                assert_eq!(headers["authorization"], "Bearer fixture-key");
                assert_eq!(body["model"], "gpt-native");
                assert_eq!(body["max_output_tokens"], 50);
                if body["stream"] == true {
                    let chunks: Vec<_> = raw
                        .as_bytes()
                        .chunks(7)
                        .map(|bytes| Ok::<_, std::io::Error>(bytes::Bytes::copy_from_slice(bytes)))
                        .collect();
                    (
                        [("content-type", "text/event-stream")],
                        Body::from_stream(futures::stream::iter(chunks)),
                    )
                        .into_response()
                } else {
                    Json(native).into_response()
                }
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let directory = std::env::temp_dir().join(format!("tinyllm-api-http-{}", uuid::Uuid::new_v4()));
    let server = crate::config::Server::default();
    let store = Arc::new(
        Store::open(directory.clone(), 1_000_000, 100_000)
            .await
            .unwrap(),
    );
    let provider = OpenAiProvider::new(
        Config {
            base_url: base,
            auth: OpenAiAuth::ApiKey("fixture-key".into()),
            organization: None,
            project: None,
            models: Default::default(),
        },
        http::client(&server).unwrap(),
        server,
        store.clone(),
    )
    .unwrap();
    for streaming in [false, true] {
        let body = json!({"model":"codex/gpt-native","stream":streaming,"stream_options":{"include_usage":true},"max_tokens":50,"messages":[{"role":"user","content":"look up both"}]});
        let result = execute(
            &provider,
            ApiRequest::parse(ApiFormat::ChatCompletions, body).unwrap(),
            context(),
        )
        .await
        .unwrap();
        let message = match result.body {
            ResponseBody::Json(response) => {
                assert_eq!(response["choices"][0]["finish_reason"], "tool_calls");
                assert_eq!(response["usage"]["total_tokens"], 17);
                response["choices"][0]["message"].clone()
            }
            ResponseBody::Stream(mut events) => {
                let mut message = json!({"role":"assistant","content":"","tool_calls":[]});
                let mut done = false;
                let mut usage = false;
                while let Some(event) = events.next().await {
                    match event.unwrap() {
                        ApiEvent::ChatCompletions(chunk) => {
                            if chunk["choices"] == json!([]) {
                                assert_eq!(chunk["usage"]["total_tokens"], 17);
                                usage = true;
                                continue;
                            }
                            let delta = &chunk["choices"][0]["delta"];
                            if let Some(details) = delta.get("reasoning_details") {
                                message["reasoning_details"] = details.clone();
                            }
                            if let Some(calls) = delta["tool_calls"].as_array() {
                                for call in calls {
                                    let index = call["index"].as_u64().unwrap() as usize;
                                    if call.get("id").is_some() {
                                        let mut call = call.clone();
                                        call.as_object_mut().unwrap().remove("index");
                                        message["tool_calls"].as_array_mut().unwrap().push(call);
                                    } else {
                                        let arguments =
                                            message["tool_calls"][index]["function"]["arguments"]
                                                .as_str()
                                                .unwrap()
                                                .to_owned()
                                                + call["function"]["arguments"].as_str().unwrap();
                                        message["tool_calls"][index]["function"]["arguments"] =
                                            json!(arguments);
                                    }
                                }
                            }
                        }
                        ApiEvent::Done => done = true,
                        _ => panic!("wrong API event"),
                    }
                }
                assert!(done && usage);
                message
            }
        };
        assert_eq!(
            message["tool_calls"][0]["function"]["arguments"],
            "{\"x\":1}"
        );
        assert_eq!(message["tool_calls"][1]["id"], "call_b");
        let history = request::history(&json!({"messages":[message]})).unwrap();
        let restored = store
            .restore_scoped(&history, "codex", "gpt-native")
            .await
            .unwrap();
        assert_eq!(restored[&0], expected["output"].as_array().unwrap().clone());
    }
    drop(provider);
    drop(store);
    task.abort();
    tokio::fs::remove_dir_all(directory).await.unwrap();
}

#[test]
fn chat_created_timestamp_is_an_integer_when_upstream_omits_it() {
    let mut response = completed(json!([]));
    assert_eq!(
        chat_response(&response, "codex/gpt-native", "reference").unwrap()["created"],
        123
    );
    response.as_object_mut().unwrap().remove("created_at");
    assert!(chat_response(&response, "codex/gpt-native", "reference").unwrap()["created"].is_u64());
    let mut chat = stream::Chat::new("codex/gpt-native".into());
    assert!(
        chat.accept(&json!({"type":"response.created","response":{"id":"resp_1"}}))
            .unwrap()[0]["created"]
            .is_u64()
    );
    response["created_at"] = json!("bad timestamp");
    assert!(chat_response(&response, "codex/gpt-native", "reference").is_err());
}
