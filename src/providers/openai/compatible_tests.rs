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
fn native_responses_reject_all_gateway_reasoning_carriers() {
    for subscription in [false, true] {
        for item in [
            json!({"role":"assistant","content":[{"type":"redacted_thinking","data":"tinyllm:v1:Zm9yZWlnbg:opaque"}]}),
            json!({"type":"tinyllm_continuation","data":"tinyllm:v1:Zm9yZWlnbg:opaque"}),
        ] {
            assert!(request::native(json!({"input":[item]}), &model(), subscription).is_err());
        }
        assert!(
            request::native(
                json!({"input":"Discuss tinyllm_continuation as text", "text":{"format":{"type":"json_schema","name":"reply","schema":{"type":"object","properties":{"tinyllm_continuation":{"type":"string"}}}}}}),
                &model(),
                subscription
            )
            .is_ok()
        );
    }
}

#[test]
fn chat_converts_roles_images_tools_and_rejects_unsupported_controls() {
    let original = json!({"model":"openai/gpt-native","messages":[{"role":"system","content":"system"},{"role":"developer","content":"developer"},{"role":"user","content":[{"type":"text","text":"describe"},{"type":"image_url","image_url":{"url":"https://example.com/image.png","detail":"low"}}]}],"tools":[{"type":"function","function":{"name":"lookup","description":"lookup a value","parameters":{"type":"object"},"strict":false}}],"tool_choice":{"type":"function","function":{"name":"lookup"}},"response_format":{"type":"json_schema","json_schema":{"name":"reply","schema":{"type":"object"},"strict":true}},"reasoning_effort":"low","max_completion_tokens":20});
    let converted = request::chat(&original, &model(), false).unwrap();
    assert_eq!(converted["input"][0]["content"][0]["type"], "input_text");
    assert_eq!(converted["input"][1]["role"], "developer");
    assert_eq!(converted["input"][1]["content"][0]["type"], "input_text");
    assert_eq!(converted["input"][2]["content"][0]["type"], "input_text");
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
        assert!(request::chat(&invalid, &model(), false).is_err(), "{key}");
    }
}

fn completed(output: Value) -> Value {
    json!({"id":"resp_1","object":"response","created_at":123,"status":"completed","output":output,"usage":{"input_tokens":10,"output_tokens":7,"total_tokens":17,"input_tokens_details":{"cached_tokens":3},"output_tokens_details":{"reasoning_tokens":2}}})
}

#[test]
fn chat_carrier_roundtrip_replays_reasoning_and_two_tools() {
    let native = completed(
        json!([{"id":"rs_1","type":"reasoning","encrypted_content":"opaque","summary":[]},{"id":"fc_1","type":"function_call","call_id":"call_a","name":"lookup","arguments":"{\"x\":1}"},{"id":"fc_2","type":"function_call","call_id":"call_b","name":"lookup","arguments":"{\"x\":2}"}]),
    );
    let response = chat_response(&native, "openai/gpt-native").unwrap();
    let message = response["choices"][0]["message"].clone();
    assert_eq!(
        message["reasoning_details"][0]["type"],
        "tinyllm_continuation"
    );

    let mut history = json!({"model":"openai/gpt-native","messages":[{"role":"user","content":"lookup both"},message,{"role":"tool","tool_call_id":"call_a","content":"first"},{"role":"tool","tool_call_id":"call_b","content":"second"}]});
    let body = request::chat(&history, &model(), false).unwrap();
    assert_eq!(body["input"][1], native["output"][0]);
    assert_eq!(body["input"][4]["call_id"], "call_a");
    assert_eq!(body["input"][5]["call_id"], "call_b");
    assert_eq!(
        response["usage"]["prompt_tokens_details"]["cached_tokens"],
        3
    );

    // A client that rewrites the assistant turn keeps replaying its reasoning.
    history["messages"][1]["tool_calls"][0]["function"]["arguments"] = json!("{}");
    let body = request::chat(&history, &model(), false).unwrap();
    assert_eq!(body["input"][1], native["output"][0]);
    assert_eq!(body["input"][2]["arguments"], "{}");

    // A client that drops the carrier degrades to portable history instead of failing.
    history["messages"][1]
        .as_object_mut()
        .unwrap()
        .remove("reasoning_details");
    let body = request::chat(&history, &model(), false).unwrap();
    assert_eq!(body["input"][1]["type"], "function_call");
    assert_eq!(body["input"][1]["call_id"], "call_a");
    assert_eq!(body["input"][3]["type"], "function_call_output");
}

#[test]
fn chat_portable_history_preserves_visible_assistant_content_and_tools() {
    let assistant = json!({
        "role":"assistant",
        "content":[{"type":"text","text":"visible"}],
        "refusal":"declined",
        "annotations":[{"type":"url_citation","url_citation":{"start_index":0,"end_index":7,"title":"source","url":"https://example.com"}}],
        "tool_calls":[{"id":"call/raw","type":"function","function":{"name":"lookup","arguments":"{\"x\":1}"}}]
    });
    for details in [None, Some(Value::Null), Some(json!([]))] {
        let mut assistant = assistant.clone();
        if let Some(details) = details {
            assistant["reasoning_details"] = details;
        }
        let body = request::chat(
            &json!({"messages":[assistant,{"role":"tool","tool_call_id":"call/raw","content":[{"type":"text","text":"result"}]}]}),
            &model(),
            false,
        )
        .unwrap();
        assert_eq!(
            body["input"][0],
            json!({"role":"assistant","content":[{"type":"output_text","text":"visible"},{"type":"refusal","refusal":"declined"}]})
        );
        assert_eq!(
            body["input"][1],
            json!({"type":"function_call","call_id":"call/raw","name":"lookup","arguments":"{\"x\":1}"})
        );
        assert_eq!(
            body["input"][2],
            json!({"type":"function_call_output","call_id":"call/raw","output":[{"type":"input_text","text":"result"}]})
        );
    }
}

#[test]
fn chat_portable_history_rejects_malformed_assistant_fields_and_details() {
    let rejected = |message: Value| {
        assert!(request::chat(&json!({"messages":[message]}), &model(), false,).is_err());
    };
    for message in [
        json!({"role":"assistant","content":"text","unknown":true}),
        json!({"role":"assistant","content":1}),
        json!({"role":"assistant","refusal":[]}),
        json!({"role":"assistant","annotations":{}}),
        json!({"role":"assistant","annotations":[null]}),
        json!({"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"name":"lookup","arguments":{}}}]}),
        json!({"role":"assistant","reasoning_details":{}}),
        json!({"role":"assistant","reasoning_details":[{"type":"foreign","data":"opaque"}]}),
    ] {
        rejected(message);
    }
}

#[test]
fn chat_portable_history_rejects_invalid_tool_pairing() {
    let call = |id| json!({"role":"assistant","tool_calls":[{"id":id,"type":"function","function":{"name":"lookup","arguments":"{}"}}]});
    assert!(
        request::chat(
            &json!({"messages":[call(""),{"role":"tool","tool_call_id":"","content":"result"}]}),
            &model(),
            false,
        )
        .is_err()
    );
    assert!(
        request::chat(
            &json!({"messages":[{"role":"assistant","tool_calls":[
                {"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{}"}},
                {"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{}"}}
            ]},{"role":"tool","tool_call_id":"call_1","content":"result"}]}),
            &model(),
            false,
        )
        .is_err()
    );
    assert!(
        request::chat(
            &json!({"messages":[{"role":"tool","tool_call_id":"call_1","content":"result"},call("call_1")]}),
            &model(),
            false,
        )
        .is_err()
    );
    assert!(
        request::chat(
            &json!({"messages":[call("call_1"),{"role":"tool","tool_call_id":"call_1","content":"result"},{"role":"tool","tool_call_id":"call_1","content":"duplicate"}]}),
            &model(),
            false,
        )
        .is_err()
    );
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

/// Replays `reasoning_details` carriers back into upstream Responses input items.
fn replayed(details: &Value) -> Value {
    let history =
        json!({"messages":[{"role":"assistant","content":"x","reasoning_details":details}]});
    request::chat(&history, &model(), false).unwrap()["input"][0].clone()
}

#[test]
fn fast_suffix_resolves_only_configured_gpt_models() {
    use super::super::models::{Config, ModelOptions, OpenAiAuth};
    use crate::providers::Provider;
    let server = crate::config::Server::default();
    let provider = OpenAiProvider::new(
        Config {
            base_url: "https://api.openai.com/v1".into(),
            auth: OpenAiAuth::ApiKey("fixture-key".into()),
            organization: None,
            project: None,
            models: ["gpt-5.4", "gpt-5.4-fast", "gpt-6-astra", "o4-mini"]
                .map(|id| (id.to_owned(), ModelOptions::default()))
                .into(),
        },
        http::client(&server).unwrap(),
        server,
    )
    .unwrap();
    assert_eq!(
        provider.synthetic_fast_base("gpt-6-astra-fast"),
        Some("gpt-6-astra")
    );
    for native in [
        "gpt-5.4-fast",      // configured upstream ID stays literal
        "gpt-5.4-fast-fast", // the suffix never stacks
        "o4-mini-fast",      // not a gpt- model
        "gpt-9-fast",        // base is not configured
        "gpt-6-astra",
    ] {
        assert_eq!(provider.synthetic_fast_base(native), None, "{native}");
    }
    assert_eq!(
        provider
            .models()
            .into_iter()
            .map(|model| model.id)
            .collect::<Vec<_>>(),
        [
            "gpt-5.4",
            "gpt-5.4-fast",
            "gpt-6-astra",
            "gpt-6-astra-fast",
            "o4-mini"
        ]
    );
    assert!(
        provider
            .convert_reasoning_effort("gpt-6-astra-fast", "none")
            .is_err()
    );
}

#[tokio::test]
async fn subscription_json_and_sse_accept_missing_content_type_and_keep_reasoning() {
    use super::super::models::{Config, ModelOptions, OpenAiAuth, SubscriptionOptions};
    use axum::{Json, Router, response::IntoResponse, routing::post};
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
                "data: {{\"type\":\"ping\"}}\n\nevent: {}\ndata: {event}\n\nevent: keepalive\ndata: {{\"type\":\"keepalive\"}}\n\n",
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
                assert_eq!(
                    headers["x-codex-routing-hint"],
                    "model=gpt-native;tier=priority"
                );
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
                        assert_eq!(replayed(&message["reasoning_details"]), reasoning);
                    }
                }
                ResponseBody::Stream(mut events) => {
                    let mut terminal = false;
                    while let Some(event) = events.next().await {
                        match event.unwrap() {
                            ApiEvent::Responses(value) => {
                                assert!(!matches!(
                                    value["type"].as_str(),
                                    Some("ping" | "keepalive")
                                ));
                                if value["type"] == "response.completed" {
                                    assert_eq!(value["response"]["output"][0], reasoning);
                                    terminal = true;
                                }
                            }
                            ApiEvent::ChatCompletions(value)
                                if !value["choices"][0]["finish_reason"].is_null() =>
                            {
                                let details = &value["choices"][0]["delta"]["reasoning_details"];
                                assert!(details.is_array());
                                assert_eq!(replayed(details), reasoning);
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
    task.abort();
    let _ = tokio::fs::remove_dir_all(directory).await;
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
    let converted = request::chat(&body, &model(), false).unwrap();
    assert_eq!(converted["reasoning"]["effort"], Value::Null);
    assert_eq!(converted["tools"][0]["strict"], false);
}

#[test]
fn chat_usage_rejects_impossible_cache_or_reasoning_details() {
    let mut response = completed(json!([]));
    response["usage"]["input_tokens_details"]["cached_tokens"] = json!(11);
    assert!(chat_response(&response, "codex/gpt-native").is_err());
    response["usage"]["input_tokens_details"]["cached_tokens"] = json!(0);
    response["usage"]["output_tokens_details"]["reasoning_tokens"] = json!(8);
    assert!(chat_response(&response, "codex/gpt-native").is_err());
}

#[tokio::test]
async fn api_key_chat_json_and_fragmented_sse_roundtrip_two_tools() {
    use super::super::models::{Config, OpenAiAuth};
    use axum::{Json, Router, body::Body, response::IntoResponse, routing::post};
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
        assert_eq!(
            replayed(&message["reasoning_details"]),
            expected["output"][0]
        );
    }
    drop(provider);
    task.abort();
    let _ = tokio::fs::remove_dir_all(directory).await;
}

#[test]
fn chat_created_timestamp_is_an_integer_when_upstream_omits_it() {
    let mut response = completed(json!([]));
    assert_eq!(
        chat_response(&response, "codex/gpt-native").unwrap()["created"],
        123
    );
    response.as_object_mut().unwrap().remove("created_at");
    assert!(chat_response(&response, "codex/gpt-native").unwrap()["created"].is_u64());
    let mut chat = stream::Chat::new("codex/gpt-native".into());
    assert!(
        chat.accept(&json!({"type":"response.created","response":{"id":"resp_1"}}))
            .unwrap()[0]["created"]
            .is_u64()
    );
    response["created_at"] = json!("bad timestamp");
    assert!(chat_response(&response, "codex/gpt-native").is_err());
}
