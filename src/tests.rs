use crate::{
    config::ProviderConfig,
    models::ReasoningEffort,
    providers::openai::{
        models::{Model, ModelOptions, OpenAiAuth, ServiceTier, SubscriptionOptions},
        protocol, stream,
    },
};
use serde_json::{Value, json};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

#[test]
fn provider_config_uses_native_model_names() {
    let path =
        std::env::temp_dir().join(format!("tinyllm-providers-{}.toml", uuid::Uuid::new_v4()));
    std::fs::write(
        &path,
        r#"
[providers.openai]
type = "openai"
auth = { type = "ApiKey", options = "fixture-key" }
[providers.openai.models."gpt-6-astra"]
reasoning_effort = "medium"
[providers.openrouter]
type = "openrouter"
api_key = "fixture-key"
[providers.openrouter.models."deepseek/deepseek-4-pro"]
"#,
    )
    .unwrap();
    let loaded = crate::config::Config::load(&path);
    std::fs::remove_file(path).unwrap();
    assert!(
        loaded.is_ok(),
        "provider configuration should accept native IDs"
    );
}

#[test]
fn openai_service_tier_config_is_validated() {
    for tier in [
        json!(null),
        json!("auto"),
        json!("default"),
        json!("flex"),
        json!("priority"),
        json!("fast"),
        json!("ultrafast"),
        json!("invalid"),
        json!(""),
        json!(true),
        json!(42),
    ] {
        let valid =
            !matches!(tier, Value::Bool(_) | Value::Number(_)) && tier != "invalid" && tier != "";
        let config = serde_json::from_value::<crate::config::Config>(json!({
            "providers":{"openai":{"type":"openai","auth":{"type":"ApiKey","options":"fixture-key"},
                "models":{"gpt-test":{"service_tier":tier}}}}
        }));
        assert_eq!(config.is_ok(), valid, "deserialize service tier {tier}");
        if let Ok(config) = &config {
            let ProviderConfig::OpenAi(provider) = &config.providers["openai"] else {
                panic!("wrong provider variant")
            };
            assert_eq!(json!(provider.models["gpt-test"].service_tier), tier);
        }
        assert_eq!(
            config.is_ok_and(|config| config.validate().is_ok()),
            valid,
            "{tier}"
        );
    }
}

fn model() -> Model {
    Model {
        id: "gpt-test".into(),
        reasoning_effort: Some(ReasoningEffort::Medium),
        max_reasoning_effort: None,
    }
}

fn request() -> Value {
    json!({"model":"openai/gpt-test","max_tokens":1024,"system":[{"type":"text","text":"Keep ALL instructions."}],
        "messages":[{"role":"user","content":"hello"}],
        "tools":[{"name":"lookup","description":"Find a record","input_schema":{"type":"object","properties":{"url":{"type":"string","format":"uri"}},"required":[]}}]})
}

#[test]
fn web_search_request_maps_parameters_and_subscription_best_effort_cap() {
    let mut req = request();
    req["tools"].as_array_mut().unwrap().push(json!({
        "type":"web_search_20250305","name":"web_search","max_uses":8,
        "allowed_domains":["example.org","docs.example.org"],
        "user_location":{"type":"approximate","country":"US","city":"Boston","region":"Massachusetts","timezone":"America/New_York"}
    }));
    req["tool_choice"] =
        json!({"type":"tool","name":"web_search","disable_parallel_tool_use":true});
    let out = protocol::request(&req, &model()).unwrap();
    assert_eq!(out["tools"][0]["type"], "function");
    assert_eq!(
        out["tools"][0]["parameters"],
        req["tools"][0]["input_schema"]
    );
    assert_eq!(
        out["tools"][1],
        json!({
            "type":"web_search","filters":{"allowed_domains":["example.org","docs.example.org"]},
            "user_location":req["tools"][1]["user_location"]
        })
    );
    assert_eq!(out["max_tool_calls"], 8);
    assert_eq!(out["tool_choice"], json!({"type":"web_search"}));
    assert_eq!(out["parallel_tool_calls"], false);
    let subscription = protocol::subscription_request(&req, out.clone()).unwrap();
    assert_eq!(subscription["tools"], out["tools"]);
    assert_eq!(subscription["tool_choice"], out["tool_choice"]);
    assert!(subscription.get("max_tool_calls").is_none());
    assert!(subscription.get("max_output_tokens").is_none());
    assert_eq!(subscription["stream"], true);
    let instructions = subscription["instructions"].as_str().unwrap();
    assert!(instructions.starts_with("Keep ALL instructions."));
    assert!(instructions.contains("at most 8"), "{instructions}");
    assert_eq!(subscription["input"][0]["content"][0]["text"], "hello");
    for (choice, expected) in [
        (json!({"type":"auto"}), json!("auto")),
        (json!({"type":"any"}), json!("required")),
        (json!({"type":"none"}), json!("none")),
        (
            json!({"type":"tool","name":"lookup"}),
            json!({"type":"function","name":"lookup"}),
        ),
    ] {
        req["tool_choice"] = choice;
        assert_eq!(
            protocol::request(&req, &model()).unwrap()["tool_choice"],
            expected
        );
    }
    req.as_object_mut().unwrap().remove("tool_choice");
    req["tools"] = json!([{"type":"web_search_20250305","name":"web_search"}]);
    let out = protocol::request(&req, &model()).unwrap();
    assert_eq!(out["tools"], json!([{"type":"web_search"}]));
    assert!(out.get("max_tool_calls").is_none());
    assert_eq!(
        protocol::subscription_request(&req, out).unwrap()["instructions"],
        "Keep ALL instructions."
    );
    req["tool_choice"] = json!({"type":"tool","name":"web_search"});
    assert_eq!(
        protocol::request(&req, &model()).unwrap()["tool_choice"],
        "required"
    );
    req["tools"]
        .as_array_mut()
        .unwrap()
        .push(json!({"name":"unloaded","input_schema":{"type":"object"},"defer_loading":true}));
    assert_eq!(
        protocol::request(&req, &model()).unwrap()["tool_choice"],
        "required"
    );
    req["tools"][0]["blocked_domains"] = json!(["example.org"]);
    assert_eq!(
        protocol::request(&req, &model()).unwrap()["tools"][0]["filters"],
        json!({"blocked_domains":["example.org"]})
    );
    req["tools"][0]["blocked_domains"] = json!([]);
    req["tools"][0]["allowed_domains"] = json!([]);
    req["stop_sequences"] = json!([]);
    assert!(protocol::request(&req, &model()).is_ok());
    req["tools"][0]["allowed_domains"] = json!(
        (0..100)
            .map(|i| format!("site{i}.example.org"))
            .collect::<Vec<_>>()
    );
    assert!(protocol::request(&req, &model()).is_ok());
}

#[test]
fn web_search_request_rejects_invalid_or_unsupported_options() {
    let search = json!({"type":"web_search_20250305","name":"web_search"});
    for change in [
        json!({"type":"web_search_20260209"}),
        json!({"type":"web_fetch_20250910"}),
        json!({"type":42}),
        json!({"name":"search"}),
        json!({"max_uses":0}),
        json!({"max_uses":-1}),
        json!({"max_uses":1.5}),
        json!({"max_uses":"8"}),
        json!({"max_uses":true}),
        json!({"allowed_domains":false}),
        json!({"allowed_domains":[42]}),
        json!({"allowed_domains":["https://example.org"]}),
        json!({"allowed_domains":["example.org/path"]}),
        json!({"allowed_domains":["*.example.org"]}),
        json!({"allowed_domains":["example.org:443"]}),
        json!({"blocked_domains":["example.org?query"]}),
        json!({"blocked_domains":["example..org"]}),
        json!({"blocked_domains":[""]}),
        json!({"allowed_domains":["example.org"],"blocked_domains":["example.net"]}),
        json!({"allowed_domains":(0..101).map(|i| format!("site{i}.example.org")).collect::<Vec<_>>()}),
        json!({"user_location":{"type":"precise","country":"US"}}),
        json!({"user_location":{"type":"approximate","country":"USA"}}),
        json!({"user_location":{"type":"approximate","country":"12"}}),
        json!({"user_location":{"type":"approximate","city":42}}),
        json!({"user_location":{"type":"approximate","region":false}}),
        json!({"user_location":{"type":"approximate","timezone":[]}}),
        json!({"user_location":{"type":"approximate","latitude":42}}),
        json!({"input_schema":{"type":"object"}}),
        json!({"defer_loading":true}),
        json!({"strict":true}),
        json!({"unknown":true}),
    ] {
        let mut req = request();
        req["tools"] = json!([search]);
        req["tools"][0]
            .as_object_mut()
            .unwrap()
            .extend(change.as_object().unwrap().clone());
        assert!(protocol::request(&req, &model()).is_err(), "{change}");
    }
    for kind in ["web_search_20260209", "web_search_20260318"] {
        let mut req = request();
        req["tools"] = json!([{"type":kind,"name":"web_search","max_uses":8}]);
        assert!(
            protocol::request(&req, &model())
                .unwrap_err()
                .message
                .contains("unsupported hosted tool type")
        );
    }
    let custom = json!({"name":"web_search","input_schema":{"type":"object"}});
    for tools in [
        json!([search, search]),
        json!([search, custom]),
        json!([custom, search]),
    ] {
        let mut req = request();
        req["tools"] = tools;
        assert!(
            protocol::request(&req, &model()).is_err(),
            "{}",
            req["tools"]
        );
    }
    for control in [
        json!({"stop_sequences":["END"]}),
        json!({"output_config":{"format":{"type":"json_schema","schema":{"type":"object","properties":{},"additionalProperties":false}}}}),
    ] {
        let mut req = request();
        req["tools"] = json!([search]);
        req.as_object_mut()
            .unwrap()
            .extend(control.as_object().unwrap().clone());
        assert!(protocol::request(&req, &model()).is_err(), "{control}");
    }
}

#[test]
fn ordered_history_images_tool_errors_and_choices() {
    let mut req = request();
    req["tool_choice"] = json!({"type":"tool","name":"lookup","disable_parallel_tool_use":true});
    req["messages"] = json!([
        {"role":"user","content":[{"type":"image","source":{"type":"url","url":"https://example.org/image.png"}},{"type":"text","text":"inspect"}]},
        {"role":"assistant","content":[{"type":"text","text":"Checking"},{"type":"tool_use","id":"a","name":"lookup","input":{"url":"a"}},{"type":"tool_use","id":"b","name":"lookup","input":{"url":"b"}}]},
        {"role":"user","content":[{"type":"tool_result","tool_use_id":"b","is_error":true,"content":[{"type":"text","text":"denied"},{"type":"image","source":{"type":"base64","media_type":"image/png","data":"YWJj"}}]},{"type":"tool_result","tool_use_id":"a","content":{"answer":42}},{"type":"text","text":"Continue"}]}
    ]);
    let out = protocol::request(&req, &model()).unwrap();
    assert_eq!(out["model"], "gpt-test");
    assert_eq!(
        out["input"][0]["content"][0]["text"],
        "Keep ALL instructions."
    );
    assert_eq!(out["input"][1]["content"][0]["type"], "input_image");
    assert_eq!(out["input"][2]["content"][0]["type"], "output_text");
    assert_eq!(out["input"][3]["call_id"], "a");
    assert_eq!(out["input"][4]["call_id"], "b");
    assert_eq!(out["input"][5]["call_id"], "b");
    assert!(
        out["input"][5]["output"][0]["text"]
            .as_str()
            .unwrap()
            .contains("is_error")
    );
    assert_eq!(out["input"][5]["output"][2]["type"], "input_image");
    assert_eq!(out["input"][6]["output"], "{\"answer\":42}");
    assert_eq!(
        out["tools"][0]["parameters"],
        req["tools"][0]["input_schema"]
    );
    assert_eq!(out["tools"][0]["strict"], false);
    assert_eq!(
        out["tool_choice"],
        json!({"type":"function","name":"lookup"})
    );
    assert_eq!(out["parallel_tool_calls"], false);
    for (source, target) in [("auto", "auto"), ("any", "required"), ("none", "none")] {
        req["tool_choice"] = json!({"type":source});
        assert_eq!(
            protocol::request(&req, &model()).unwrap()["tool_choice"],
            target
        );
    }
}

#[test]
fn rejects_meaningful_unsupported_content_and_orphan_results() {
    for change in [
        json!({"top_k":3}),
        json!({"stop_sequences":[""]}),
        json!({"context_management":{"edits":[{}]}}),
    ] {
        let mut req = request();
        req.as_object_mut()
            .unwrap()
            .extend(change.as_object().unwrap().clone());
        assert!(protocol::request(&req, &model()).is_err());
    }
    let mut req = request();
    req["messages"][0]["content"] =
        json!([{"type":"tool_result","tool_use_id":"missing","content":"never drop me"}]);
    assert!(protocol::request(&req, &model()).is_err());
}

#[test]
fn context_management_preserves_history_and_rejects_unimplemented_edits() {
    let mut req = request();
    let expected = protocol::request(&req, &model()).unwrap();
    for context in [
        json!({}),
        json!({"edits":[]}),
        json!({"edits":[{"type":"clear_thinking_20251015","keep":"all"}]}),
    ] {
        req["context_management"] = context;
        assert_eq!(protocol::request(&req, &model()).unwrap(), expected);
    }
    for context in [
        json!({"edits":false}),
        json!({"edits":[{"type":"clear_thinking_20251015"}]}),
        json!({"edits":[{"type":"clear_thinking_20251015","keep":{"type":"thinking_turns","value":1}}]}),
        json!({"edits":[{"type":"clear_tool_uses_20250919"}]}),
        json!({"edits":[{"type":"compact_20260112"}]}),
        json!({"edits":[{"type":"clear_thinking_20251015","keep":"all","unknown":true}]}),
    ] {
        req["context_management"] = context;
        assert!(protocol::request(&req, &model()).is_err());
    }
}

#[test]
fn deferred_tools_load_from_references_without_losing_result_content() {
    let mut req = request();
    req["tools"] = json!([
        {"name":"ToolSearch","input_schema":{"type":"object"}},
        {"name":"lookup","description":"Find a record","input_schema":{"type":"object"},"defer_loading":true},
        {"name":"unused","input_schema":{"type":"object"},"defer_loading":true}
    ]);
    let initial = protocol::request(&req, &model()).unwrap();
    assert_eq!(initial["tools"].as_array().unwrap().len(), 1);
    req["messages"].as_array_mut().unwrap().extend([
        json!({"role":"assistant","content":[{"type":"tool_use","id":"search","name":"ToolSearch","input":{"query":"lookup"}}]}),
        json!({"role":"user","content":[{"type":"tool_result","tool_use_id":"search","content":[{"type":"text","text":"Found a tool"},{"type":"tool_reference","tool_name":"lookup"},{"type":"image","source":{"type":"url","url":"https://example.org/result.png"}}]}]})
    ]);
    let out = protocol::request(&req, &model()).unwrap();
    assert_eq!(out["tools"].as_array().unwrap().len(), 2);
    assert_eq!(out["tools"][1]["name"], "lookup");
    assert_eq!(
        out["tools"][1]["parameters"],
        req["tools"][1]["input_schema"]
    );
    assert_eq!(out["input"][3]["output"][0]["text"], "Found a tool");
    let reference: Value =
        serde_json::from_str(out["input"][3]["output"][1]["text"].as_str().unwrap()).unwrap();
    assert_eq!(
        reference,
        json!({"type":"tool_reference","tool_name":"lookup"})
    );
    assert_eq!(out["input"][3]["output"][2]["type"], "input_image");
    req["messages"][2]["content"][0]["content"][1]["tool_name"] = json!("missing");
    assert!(protocol::request(&req, &model()).is_err());
    req["messages"].as_array_mut().unwrap().truncate(1);
    assert_eq!(protocol::request(&req, &model()).unwrap(), initial);
    req["tools"][1]["defer_loading"] = json!("true");
    assert!(protocol::request(&req, &model()).is_err());
}

fn upstream_response(output: Value) -> Value {
    json!({"id":"resp_test","status":"completed","output":output,"usage":{"input_tokens":120,"output_tokens":25,"input_tokens_details":{"cached_tokens":20},"output_tokens_details":{"reasoning_tokens":15}}})
}

#[tokio::test]
async fn web_search_max_uses_eight_round_trip_restores_native_output_after_restart() {
    use axum::{Json, Router, routing::post};
    let mut native = web_search_stream_fixture().last().unwrap()["response"].clone();
    native["output"].as_array_mut().unwrap().truncate(2);
    native["output"].as_array_mut().unwrap().insert(0,
        json!({"type":"web_search_call","id":"ws_failed","status":"failed","action":{"type":"search","query":"failed query"}}));
    let (upstream, up_task) = serve(Router::new().route(
        "/responses",
        post({
            let native = native.clone();
            move |Json(req): Json<Value>| {
                let native = native.clone();
                async move {
                    assert_eq!(req["model"], "gpt-test");
                    assert_eq!(req["tools"], json!([{"type":"web_search"}]));
                    assert_eq!(req["tool_choice"], "required");
                    assert_eq!(req["max_tool_calls"], 8);
                    assert_eq!(req["max_output_tokens"], 1024);
                    assert_eq!(
                        req["input"][0]["content"][0]["text"],
                        "Keep ALL instructions."
                    );
                    assert_eq!(req["input"][1]["content"][0]["text"], "hello");
                    Json(native)
                }
            }
        }),
    ))
    .await;
    let directory =
        std::env::temp_dir().join(format!("tinyllm-web-search-{}", uuid::Uuid::new_v4()));
    let (gateway, task) = serve(
        crate::server::router(config(upstream, directory.clone()))
            .await
            .unwrap(),
    )
    .await;
    let mut req = request();
    req["tools"] = json!([{"type":"web_search_20250305","name":"web_search","max_uses":8}]);
    req["tool_choice"] = json!({"type":"tool","name":"web_search"});
    let response = reqwest::Client::new()
        .post(format!("{gateway}/anthropic/v1/messages"))
        .bearer_auth("local-secret")
        .json(&req)
        .send()
        .await
        .unwrap();
    let status = response.status();
    assert_eq!(
        response.headers()["x-tinyllm-web-search"],
        "native; citations=markdown; max-uses=upstream"
    );
    let response: Value = response.json().await.unwrap();
    assert_eq!(status, 200, "{response}");
    assert_eq!(response["stop_reason"], "end_turn");
    assert_eq!(response["content"].as_array().unwrap().len(), 2);
    assert_eq!(
        response["content"][0],
        json!({"type":"text","text":"héllo"})
    );
    assert_eq!(
        response["content"][1]["text"],
        "\n\nSources:\n- <https://example.org/report>\n- <https://example.net/a%20b>"
    );
    task.abort();
    let _ = task.await;
    up_task.abort();
    let _ = up_task.await;
    req["messages"].as_array_mut().unwrap().extend([
        json!({"role":"assistant","content":response["content"]}),
        json!({"role":"user","content":"Continue"}),
    ]);
    // Hosted search produces no reasoning carrier, so the turn replays as the
    // visible text plus the Sources block the response already carries.
    let replay = protocol::request(&req, &model()).unwrap();
    assert!(
        !replay["input"]
            .as_array()
            .unwrap()
            .iter()
            .any(|item| item["type"] == "reasoning")
    );
    assert_eq!(replay["input"][2]["content"][0]["text"], "héllo");
    assert_eq!(replay["input"][3]["content"][0]["text"], "Continue");
    let subscription = protocol::subscription_request(&req, replay).unwrap();
    assert!(subscription.get("max_tool_calls").is_none());

    // A rewritten visible turn still translates.
    let mut changed = req.clone();
    changed["messages"][1]["content"][0]["text"] = json!("changed visible content");
    let replay = protocol::request(&changed, &model()).unwrap();
    assert_eq!(
        replay["input"][2]["content"][0]["text"],
        "changed visible content"
    );
    let _ = tokio::fs::remove_dir_all(directory).await;
}

#[tokio::test]
async fn web_search_unsolicited_citations_with_stops_or_structured_output_fail_json_and_sse() {
    use axum::{Json, Router, response::IntoResponse, routing::post};
    let mut events = web_search_stream_fixture();
    events.retain(|event| event["output_index"] != 0);
    for event in &mut events {
        if let Some(index) = event["output_index"].as_u64() {
            event["output_index"] = json!(index - 1);
        }
    }
    events.last_mut().unwrap()["response"]["output"]
        .as_array_mut()
        .unwrap()
        .remove(0);
    for event in &mut events {
        if event["type"] == "response.output_text.delta" {
            event["delta"] = json!(if event["delta"] == "hé" {
                "{\"answer\":\"hé"
            } else {
                "llo\"}"
            });
        }
        for pointer in [
            "/text",
            "/part/text",
            "/item/content/0/text",
            "/response/output/0/content/0/text",
        ] {
            if let Some(text) = event.pointer_mut(pointer)
                && *text == "héllo"
            {
                *text = json!("{\"answer\":\"héllo\"}");
            }
        }
    }
    let native = events.last().unwrap()["response"].clone();
    let raw = events
        .iter()
        .map(|event| format!("data: {event}\n\n"))
        .collect::<String>();
    let (upstream, up_task) = serve(Router::new().route(
        "/responses",
        post(move |Json(req): Json<Value>| {
            let native = native.clone();
            let raw = raw.clone();
            async move {
                assert_eq!(req["tools"][0]["type"], "function");
                if req["stream"] == true {
                    ([("content-type", "text/event-stream")], raw).into_response()
                } else {
                    Json(native).into_response()
                }
            }
        }),
    ))
    .await;
    let directory =
        std::env::temp_dir().join(format!("tinyllm-web-search-stop-{}", uuid::Uuid::new_v4()));
    let (gateway, task) = serve(
        crate::server::router(config(upstream, directory.clone()))
            .await
            .unwrap(),
    )
    .await;
    for control in [
        json!({"stop_sequences":["éll"]}),
        json!({"output_config":{"format":{"type":"json_schema","schema":{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"],"additionalProperties":false}}}}),
    ] {
        for streaming in [false, true] {
            let mut req = request();
            req.as_object_mut()
                .unwrap()
                .extend(control.as_object().unwrap().clone());
            req["stream"] = json!(streaming);
            let response = reqwest::Client::new()
                .post(format!("{gateway}/anthropic/v1/messages"))
                .bearer_auth("local-secret")
                .json(&req)
                .send()
                .await
                .unwrap();
            if streaming {
                assert_eq!(response.status(), 200);
                let response = response.text().await.unwrap();
                assert!(
                    response.contains(
                        "cited output with stop_sequences or structured output is unsupported"
                    ),
                    "{control}: {response}"
                );
                assert!(
                    !response.contains("event: message_stop"),
                    "{control}: {response}"
                );
                assert!(!response.contains("Sources:"), "{control}: {response}");
            } else {
                assert_eq!(response.status(), 502, "{control}");
                assert_eq!(
                    response.json::<Value>().await.unwrap()["error"]["message"],
                    "cited output with stop_sequences or structured output is unsupported"
                );
            }
        }
    }
    task.abort();
    let _ = task.await;
    up_task.abort();
    let _ = up_task.await;
    let _ = tokio::fs::remove_dir_all(directory).await;
}

#[tokio::test]
async fn rewritten_assistant_turns_still_replay_their_reasoning() {
    use axum::{Json, Router, routing::post};
    let native = upstream_response(json!([
        {"type":"reasoning","id":"rs","summary":[],"encrypted_content":"opaque"},
        {"type":"function_call","id":"fc","call_id":"edit_1","name":"Edit","arguments":"{\"file_path\":\"/tmp/example\",\"old_string\":\"OLD\",\"new_string\":\"NEW\"}"}
    ]));
    let response = native.clone();
    let (upstream, upstream_task) = serve(Router::new().route(
        "/responses",
        post(move || {
            let response = response.clone();
            async move { Json(response) }
        }),
    ))
    .await;
    let directory = std::env::temp_dir().join(format!("tinyllm-defaults-{}", uuid::Uuid::new_v4()));
    let (gateway, task) = serve(
        crate::server::router(config(upstream, directory.clone()))
            .await
            .unwrap(),
    )
    .await;
    let mut req = request();
    req["tools"] = json!([{"name":"Edit","input_schema":{"type":"object","properties":{"file_path":{"type":"string"},"old_string":{"type":"string"},"new_string":{"type":"string"},"replace_all":{"type":"boolean","default":false}},"required":["file_path","old_string","new_string"]}}]);
    let response: Value = reqwest::Client::new()
        .post(format!("{gateway}/anthropic/v1/messages"))
        .bearer_auth("local-secret")
        .json(&req)
        .send()
        .await
        .unwrap()
        .error_for_status()
        .unwrap()
        .json()
        .await
        .unwrap();
    task.abort();
    let _ = task.await;
    upstream_task.abort();

    // Claude Code materializes omitted defaults, renames IDs and rewrites tool
    // input between turns. None of that may cost the turn its reasoning.
    let original = response["content"].clone();
    type Edit = (&'static str, fn(&mut Value));
    let edits: [Edit; 6] = [
        ("materialized default", |c| {
            c[1]["input"]["replace_all"] = json!(false)
        }),
        ("changed argument", |c| {
            c[1]["input"]["file_path"] = json!("/tmp/changed")
        }),
        ("unknown argument", |c| {
            c[1]["input"]["unknown"] = json!(false)
        }),
        ("renamed call", |c| c[1]["id"] = json!("changed")),
        ("renamed tool", |c| c[1]["name"] = json!("Other")),
        ("foreign carrier", |c| {
            c[0]["data"] = json!("anthropic-signature")
        }),
    ];
    for (label, edit) in edits {
        let mut content = original.clone();
        edit(&mut content);
        let call_id = content[1]["id"].clone();
        req["messages"] = json!([
            {"role":"assistant","content":content},
            {"role":"user","content":[{"type":"tool_result","tool_use_id":call_id,"content":"ok"}]}
        ]);
        req["tools"] = json!([]);
        let replay = protocol::request(&req, &model()).unwrap();
        if label == "foreign carrier" {
            // A carrier tinyllm did not write is ignored, not rejected.
            assert_eq!(replay["input"][1]["type"], "function_call", "{label}");
        } else {
            assert_eq!(replay["input"][1], native["output"][0], "{label}");
            assert_eq!(replay["input"][2]["call_id"], call_id, "{label}");
        }
    }
    let _ = tokio::fs::remove_dir_all(directory).await;
}

#[tokio::test]
async fn classifier_stop_sequence_returns_only_visible_output() {
    use axum::{Json, Router, routing::post};
    let (upstream, up_task) = serve(Router::new().route("/responses", post(|Json(req): Json<Value>| async move {
        assert!(req.get("stop_sequences").is_none());
        assert!(req.get("stop").is_none());
        Json(upstream_response(json!([
            {"type":"reasoning","id":"rs_before","summary":[],"encrypted_content":"opaque-before"},
            {"type":"message","id":"m","role":"assistant","content":[{"type":"output_text","text":"<block>false</block>hidden","annotations":[]}]},
            {"type":"reasoning","id":"rs_after","summary":[],"encrypted_content":"opaque-after"},
            {"type":"function_call","id":"f","call_id":"hidden_call","name":"lookup","arguments":"{}"}
        ])))
    }))).await;
    let directory = std::env::temp_dir().join(format!("tinyllm-stop-{}", uuid::Uuid::new_v4()));
    let (gateway, task) = serve(
        crate::server::router(config(upstream, directory.clone()))
            .await
            .unwrap(),
    )
    .await;
    let req = json!({"model":"openai/gpt-test","messages":[{"role":"user","content":"Classify this action"}],"max_tokens":2112,"stop_sequences":["</block>"]});
    let response = reqwest::Client::new()
        .post(format!("{gateway}/anthropic/v1/messages?beta=true"))
        .bearer_auth("local-secret")
        .json(&req)
        .send()
        .await
        .unwrap();
    let status = response.status();
    let response: Value = response.json().await.unwrap();
    assert_eq!(status, 200, "{response}");
    assert_eq!(response["stop_reason"], "stop_sequence");
    assert_eq!(response["stop_sequence"], "</block>");
    assert_eq!(response["content"].as_array().unwrap().len(), 2);
    assert_eq!(response["content"][0]["type"], "redacted_thinking");
    assert_eq!(response["content"][1]["text"], "<block>false");
    assert_eq!(response["usage"]["output_tokens"], 25);
    task.abort();
    let _ = task.await;
    up_task.abort();
    let req = json!({"model":"openai/gpt-test","max_tokens":1024,"messages":[{"role":"assistant","content":response["content"]}]});
    let replay = protocol::request(&req, &model()).unwrap();
    assert_eq!(replay["input"][0]["encrypted_content"], "opaque-before");
    assert_eq!(replay["input"][1]["content"][0]["text"], "<block>false");
    let _ = tokio::fs::remove_dir_all(directory).await;
}

#[test]
fn model_switch_preserves_reasoning_carriers() {
    let native = upstream_response(json!([
        {"type":"reasoning","id":"rs_1","summary":[],"encrypted_content":"opaque"},
        {"type":"message","id":"msg_1","role":"assistant","phase":"final_answer","status":"completed","content":[{"type":"output_text","text":"Ready","annotations":[]}]}
    ]));
    let content = serde_json::to_value(
        protocol::response(&native, "openai/gpt-5.6-sol")
            .unwrap()
            .content,
    )
    .unwrap();
    let req = json!({"model":"openai/gpt-5.6-terra","max_tokens":1024,"messages":[{"role":"assistant","content":content}]});
    // The carrier is self-contained, so any model on this provider replays it.
    for model_id in ["gpt-5.6-terra", "gpt-5.6-sol", "gpt-6-astra"] {
        let model = crate::providers::openai::models::Model {
            id: model_id.into(),
            ..model()
        };
        let replay = protocol::request(&req, &model).unwrap();
        assert_eq!(replay["input"][0], native["output"][0], "{model_id}");
        assert_eq!(replay["input"][1]["content"][0]["text"], "Ready");
    }
}

#[test]
fn model_switch_carriers_replay_text_and_tool_only_history() {
    let text = json!({"type":"message","id":"msg_1","role":"assistant","phase":"commentary","status":"completed","content":[{"type":"output_text","text":"Checking","annotations":[]}]});
    let call = json!({"type":"function_call","id":"fc_1","call_id":"call_1","name":"lookup","arguments":"{}","status":"completed"});
    let other = json!({"type":"function_call","id":"fc_2","call_id":"call_2","name":"lookup","arguments":"{}","status":"completed"});
    let mut long = call.clone();
    long["call_id"] = json!("n".repeat(1024));
    for output in [
        vec![text.clone()],
        vec![call.clone()],
        vec![text, call, other],
        vec![long],
    ] {
        let mut items =
            vec![json!({"type":"reasoning","id":"rs_1","summary":[],"encrypted_content":"opaque"})];
        items.extend(output);
        let native = upstream_response(json!(items));
        let content = serde_json::to_value(
            protocol::response(&native, "openai/gpt-5.6-sol")
                .unwrap()
                .content,
        )
        .unwrap();
        for keep_thinking in [false, true] {
            let mut wire = content.clone();
            wire.as_array_mut()
                .unwrap()
                .retain(|block| keep_thinking || block["type"] != "redacted_thinking");
            let mut req = request();
            req["model"] = json!("openai/gpt-5.6-terra");
            let messages = req["messages"].as_array_mut().unwrap();
            for block in wire.as_array().unwrap() {
                messages.push(json!({"role":"assistant","content":[block]}));
            }
            let results: Vec<_> = wire
                .as_array()
                .unwrap()
                .iter()
                .filter(|block| block["type"] == "tool_use")
                .map(|block| json!({"type":"tool_result","tool_use_id":block["id"],"content":"ok"}))
                .collect();
            messages.push(json!({"role":"user","content":if results.is_empty() {json!("Continue")} else {json!(results)}}));
            let translated = protocol::request(
                &req,
                &Model {
                    id: "gpt-5.6-terra".into(),
                    reasoning_effort: None,
                    max_reasoning_effort: None,
                },
            )
            .unwrap();
            assert_eq!(translated["model"], "gpt-5.6-terra");
            let input = translated["input"].as_array().unwrap();
            assert_eq!(
                input.iter().any(|item| *item == items[0]),
                keep_thinking,
                "reasoning replays only when its carrier survives"
            );
            for call in items.iter().filter(|item| item["type"] == "function_call") {
                assert!(
                    input.iter().any(|item| item["type"] == "function_call"
                        && item["call_id"] == call["call_id"])
                );
            }
            for item in input
                .iter()
                .filter(|item| item["type"] == "function_call_output")
            {
                assert!(items.iter().any(|native| native["type"] == "function_call"
                    && native["call_id"] == item["call_id"]));
            }
            if !results.is_empty() {
                let last = req["messages"].as_array().unwrap().len() - 1;
                let mut changed = req.clone();
                changed["messages"][last]["content"][0]["tool_use_id"] = json!("unknown_call");
                assert!(protocol::request(&changed, &model()).is_err());
                let mut duplicate = req.clone();
                duplicate["messages"][last]["content"]
                    .as_array_mut()
                    .unwrap()
                    .push(results[0].clone());
                assert!(protocol::request(&duplicate, &model()).is_err());
                let mut premature = req.clone();
                let messages = premature["messages"].as_array_mut().unwrap();
                let result = messages.remove(last);
                messages.insert(1, result);
                assert!(
                    protocol::request(&premature, &model())
                        .unwrap_err()
                        .message
                        .contains("unresolved client tool call")
                );
            }
        }
    }
}

#[test]
fn history_from_a_stateful_build_still_replays() {
    // A session that started before carriers carries a store reference on its
    // assistant text and a wrapped tool call ID. Neither may fail the request.
    let mut req = request();
    req["messages"].as_array_mut().unwrap().extend([
        json!({"role":"assistant","content":[
            {"type":"redacted_thinking","data":"tinyllm:v1:2b6f0cc904d137be2e1730235f5664094b83"},
            {"type":"text","text":"old answer","tinyllm_continuation":"tinyllm:v1:2b6f0cc904d137be2e1730235f5664094b83"},
            {"type":"tool_use","id":"toolu_tinyllm_2b6f0cc904d137be2e1730235f5664094b83_Y2FsbF9h","name":"lookup","input":{}}
        ]}),
        json!({"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_tinyllm_2b6f0cc904d137be2e1730235f5664094b83_Y2FsbF9h","content":"ok"}]}),
    ]);
    let out = protocol::request(&req, &model()).unwrap();
    let input = out["input"].as_array().unwrap();
    // The legacy reference is not a carrier, so the turn replays portably.
    assert!(!input.iter().any(|item| item["type"] == "reasoning"));
    assert_eq!(input[2]["content"][0]["text"], "old answer");
    assert_eq!(input[3]["type"], "function_call");
    assert_eq!(input[4]["type"], "function_call_output");
    assert_eq!(input[3]["call_id"], input[4]["call_id"]);
    assert!(!out.to_string().contains("tinyllm_continuation"));
}

#[test]
fn carrier_stripping_leaves_tool_arguments_and_schemas_alone() {
    use crate::providers::http::strip_carriers;
    // Tool data may legitimately describe these shapes; only protocol slots
    // holding a real carrier may be rewritten.
    let mut value = json!({
        "messages":[{"role":"assistant","content":[
            {"type":"redacted_thinking","data":"tinyllm:v1:cnNfMQ:opaque"},
            {"type":"text","text":"kept"},
            {"type":"tool_use","id":"call_1","name":"write","input":{
                "blocks":[{"type":"tinyllm_continuation","data":"user content"},
                          {"type":"redacted_thinking","data":"tinyllm:v1:also user content"}]
            }}
        ],"reasoning_details":[{"type":"tinyllm_continuation","data":"tinyllm:v1:cnNfMQ:opaque"}]}],
        "tools":[{"name":"write","input_schema":{"properties":{"type":{"enum":["tinyllm_continuation","redacted_thinking"]}}}}]
    });
    let tools = value["tools"].clone();
    let tool_input = value["messages"][0]["content"][2]["input"].clone();
    strip_carriers(&mut value);
    assert_eq!(value["tools"], tools, "tool schemas must survive");
    let blocks = value["messages"][0]["content"].as_array().unwrap();
    assert_eq!(blocks.len(), 2, "only the carrier block is removed");
    assert_eq!(blocks[0]["text"], "kept");
    assert_eq!(
        blocks[1]["input"], tool_input,
        "tool arguments must survive"
    );
    assert!(
        value["messages"][0]["reasoning_details"]
            .as_array()
            .unwrap()
            .is_empty()
    );
}

#[test]
fn portable_anthropic_history_preserves_text_and_tools_without_reasoning() {
    let native_id = "call_legacy";
    let mut wire = json!([
        {"type":"text","text":"before"},
        {"type":"tool_use","id":native_id,"name":"lookup","input":{"url":"https://example.test"}},
        {"type":"text","text":"after"}
    ]);
    // Foreign thinking carriers replay as nothing rather than failing the turn.
    wire.as_array_mut().unwrap().splice(
        0..0,
        [
            json!({"type":"thinking","thinking":"private foreign thinking","signature":"foreign-signature"}),
            json!({"type":"redacted_thinking","data":"gAAAAforeign"}),
        ],
    );
    let mut req = request();
    req["messages"].as_array_mut().unwrap().extend([
        json!({"role":"assistant","content":wire}),
        json!({"role":"user","content":[{"type":"tool_result","tool_use_id":native_id,"content":"ok"}]}),
    ]);
    let body = protocol::request(&req, &model()).unwrap();
    let input = body["input"].as_array().unwrap();
    let texts: Vec<_> = input
        .iter()
        .filter(|item| item["role"] == "assistant")
        .flat_map(|item| item["content"].as_array().unwrap())
        .map(|part| {
            assert_eq!(part["type"], "output_text");
            part["text"].as_str().unwrap()
        })
        .collect();
    assert_eq!(texts, ["before", "after"]);
    for item in input.iter().filter(|item| {
        matches!(
            item["type"].as_str(),
            Some("function_call" | "function_call_output")
        )
    }) {
        assert_eq!(item["call_id"], native_id);
    }
    assert!(!input.iter().any(|item| item["type"] == "reasoning"));
    assert!(!body.to_string().contains("private foreign thinking"));
    assert!(!body.to_string().contains("gAAAAforeign"));
    for extra in [
        json!({"type":"future_content","text":"unsupported"}),
        json!({"type":"thinking","thinking":"bad signature shape","signature":42}),
    ] {
        let mut invalid = req.clone();
        invalid["messages"][1]["content"]
            .as_array_mut()
            .unwrap()
            .push(extra);
        assert!(protocol::request(&invalid, &model()).is_err());
    }
    // A carrier from another tinyllm version replays as portable history.
    let mut malformed = req.clone();
    malformed["messages"][1]["content"][1]["data"] = json!("tinyllm:v2:invalid");
    assert!(protocol::request(&malformed, &model()).is_ok());
}

#[test]
fn carriers_survive_restart_and_explicit_compaction_boundary() {
    let native = upstream_response(json!([
        {"type":"reasoning","id":"rs_1","summary":[],"encrypted_content":"opaque-openai-data"},
        {"type":"message","id":"msg_1","role":"assistant","phase":"commentary","status":"completed","content":[{"type":"output_text","text":"Checking","annotations":[]}]},
        {"type":"function_call","id":"fc_1","call_id":"call_1","name":"lookup","arguments":"{}","status":"completed"}
    ]));
    let converted = protocol::response(&native, "openai/gpt-test").unwrap();
    assert_eq!(converted.usage.input_tokens, 100);
    assert_eq!(converted.usage.cache_read_input_tokens, 20);
    assert_eq!(converted.usage.output_tokens, 25);
    let content = serde_json::to_value(converted.content).unwrap();
    let mut req = request();
    req["context_management"] = json!({"edits":[{"type":"clear_thinking_20251015","keep":"all"}]});
    req["thinking"] = json!({"type":"adaptive","display":"omitted"});
    req["messages"].as_array_mut().unwrap().extend([
        json!({"role":"assistant","content":content}),
        json!({"role":"user","content":[{"type":"tool_result","tool_use_id":"call_1","content":"ok"}]})
    ]);
    // The carrier travels with the client, so no gateway state has to survive.
    let out = protocol::request(&req, &model()).unwrap();
    assert_eq!(out["input"][2]["encrypted_content"], "opaque-openai-data");
    assert_eq!(out["input"][3]["content"][0]["text"], "Checking");

    // A rewritten visible turn keeps its reasoning.
    req["messages"][1]["content"][1]["text"] = json!("modified");
    let out = protocol::request(&req, &model()).unwrap();
    assert_eq!(out["input"][2]["encrypted_content"], "opaque-openai-data");
    assert_eq!(out["input"][3]["content"][0]["text"], "modified");

    // Compaction drops the carriers; the summary replays as plain history.
    req["messages"] = json!([{"role":"user","content":"Compacted summary; new reasoning context"}]);
    let out = protocol::request(&req, &model()).unwrap();
    assert!(
        !out["input"]
            .as_array()
            .unwrap()
            .iter()
            .any(|item| item["type"] == "reasoning")
    );
}

fn claude_fork_request() -> Value {
    let mut req = request();
    req["messages"] = json!([
        {"role":"user","content":"Parent task"},
        {"role":"assistant","content":[{"type":"tool_use","id":"fork_1","name":"Agent","input":{"subagent_type":"fork","description":"Check transport","prompt":"Check the transport."}}]},
        {"role":"user","content":[
            {"type":"tool_result","tool_use_id":"fork_1","content":[{"type":"text","text":"Fork started — processing in background"}]},
            {"type":"text","text":"<fork-boilerplate>\nYou are a worker fork. The transcript above is the parent's history — inherited reference, not your situation. You are NOT a continuation of that agent. Execute ONE directive, then stop.\n</fork-boilerplate>\n\nYour directive: Check the transport."}
        ]}
    ]);
    req
}

#[tokio::test]
async fn claude_fork_bootstrap_preserves_inherited_and_worker_reasoning() {
    let mut req = claude_fork_request();
    let native = upstream_response(json!([
        {"type":"reasoning","id":"rs_parent","summary":[],"encrypted_content":"parent-reasoning"},
        {"type":"message","id":"msg_parent","role":"assistant","phase":"commentary","content":[{"type":"output_text","text":"Parent context","annotations":[]}]}
    ]));
    let content = serde_json::to_value(
        protocol::response(&native, "openai/gpt-test")
            .unwrap()
            .content,
    )
    .unwrap();
    req["messages"].as_array_mut().unwrap().splice(
        1..1,
        [
            json!({"role":"assistant","content":content}),
            json!({"role":"user","content":"Start the worker"}),
        ],
    );
    let out = protocol::request(&req, &model()).unwrap();
    assert_eq!(out["input"][2]["encrypted_content"], "parent-reasoning");
    assert_eq!(out["input"][3]["content"][0]["text"], "Parent context");
    assert_eq!(out["input"][5]["type"], "function_call");
    assert_eq!(out["input"][5]["call_id"], "fork_1");
    assert_eq!(out["input"][6]["type"], "function_call_output");
    assert_eq!(out["input"][6]["call_id"], "fork_1");
    assert_eq!(
        out["input"][6]["output"][0]["text"],
        "Fork started — processing in background"
    );
    assert_eq!(
        out["input"][7]["content"][0]["text"],
        req["messages"][4]["content"][1]["text"]
    );

    let native = upstream_response(json!([
        {"type":"reasoning","id":"rs_worker","summary":[],"encrypted_content":"worker-reasoning"},
        {"type":"function_call","call_id":"work_1","name":"lookup","arguments":"{}"}
    ]));
    let content = serde_json::to_value(
        protocol::response(&native, "openai/gpt-test")
            .unwrap()
            .content,
    )
    .unwrap();
    req["messages"].as_array_mut().unwrap().extend([
        json!({"role":"assistant","content":content}),
        json!({"role":"user","content":[{"type":"tool_result","tool_use_id":"work_1","content":"ok"}]}),
    ]);
    let out = protocol::request(&req, &model()).unwrap();
    assert_eq!(out["input"][8]["encrypted_content"], "worker-reasoning");
    assert_eq!(out["input"][9]["call_id"], "work_1");
    assert_eq!(out["input"][10]["call_id"], "work_1");
    // Editing the inherited parent turn must not cost the fork its reasoning.
    let mut edited = req.clone();
    edited["messages"][1]["content"][1]["text"] = json!("changed");
    let out = protocol::request(&edited, &model()).unwrap();
    assert_eq!(out["input"][2]["encrypted_content"], "parent-reasoning");
    assert_eq!(out["input"][8]["encrypted_content"], "worker-reasoning");

    // So must renaming the worker's own tool call.
    let mut renamed = req;
    renamed["messages"][5]["content"][1]["name"] = json!("Other");
    let out = protocol::request(&renamed, &model()).unwrap();
    assert_eq!(out["input"][8]["encrypted_content"], "worker-reasoning");
}

#[test]
fn claude_fork_history_replays_no_reasoning_it_did_not_inherit() {
    let native = upstream_response(json!([
        {"type":"reasoning","id":"rs_spawn","summary":[],"encrypted_content":"spawning-reasoning"},
        {"type":"function_call","id":"fc_spawn_1","call_id":"fork_1","name":"Agent","arguments":"{\"subagent_type\":\"fork\"}"},
        {"type":"function_call","id":"fc_spawn_2","call_id":"fork_2","name":"Agent","arguments":"{\"subagent_type\":\"fork\"}"}
    ]));
    let wire =
        serde_json::to_value(protocol::response(&native, "openai/old").unwrap().content).unwrap();
    // Claude Code hands a worker the spawning tool call without its carrier.
    let mut req = claude_fork_request();
    req["messages"][1]["content"] = json!([wire[1]]);
    req["messages"][2]["content"][0]["tool_use_id"] = wire[1]["id"].clone();
    let output = protocol::request(&req, &model()).unwrap();
    assert_eq!(output["input"][2]["call_id"], "fork_1");
    assert_eq!(output["input"][3]["call_id"], "fork_1");
    assert!(!output.to_string().contains("spawning-reasoning"));
    assert!(!output.to_string().contains("fork_2"));
}

#[tokio::test]
async fn claude_fork_bootstrap_requires_the_complete_start_exchange() {
    let req = claude_fork_request();
    let mut parallel = req.clone();
    let mut call = parallel["messages"][1]["content"][0].clone();
    call["id"] = json!("fork_2");
    parallel["messages"][1]["content"]
        .as_array_mut()
        .unwrap()
        .push(call);
    let mut result = parallel["messages"][2]["content"][0].clone();
    result["tool_use_id"] = json!("fork_2");
    parallel["messages"][2]["content"]
        .as_array_mut()
        .unwrap()
        .insert(1, result);
    let out = protocol::request(&parallel, &model()).unwrap();
    assert_eq!(out["input"][3]["call_id"], "fork_2");
    assert_eq!(out["input"][5]["call_id"], "fork_2");
    parallel["messages"][2]["content"][1]["tool_use_id"] = json!("fork_1");
    assert!(protocol::request(&parallel, &model()).is_err());
    for (pointer, value) in [
        ("/messages/1/content", json!([])),
        ("/messages/1/content/0/name", json!("lookup")),
        ("/messages/1/content/0/id", json!("")),
        (
            "/messages/1/content/0/input/subagent_type",
            json!("general-purpose"),
        ),
        ("/messages/2/role", json!("system")),
        ("/messages/2/content", json!([])),
        ("/messages/2/content/0/tool_use_id", json!("other_call")),
        (
            "/messages/2/content/0/content/0/text",
            json!("Actual tool output"),
        ),
        ("/messages/2/content/1/type", json!("image")),
        (
            "/messages/2/content/1/text",
            json!("Continue the parent task"),
        ),
    ] {
        let mut invalid = req.clone();
        *invalid.pointer_mut(pointer).unwrap() = value;
        let valid = matches!(
            pointer,
            "/messages/1/content/0/name"
                | "/messages/1/content/0/input/subagent_type"
                | "/messages/2/content/0/content/0/text"
                | "/messages/2/content/1/text"
        );
        assert_eq!(
            protocol::request(&invalid, &model()).is_ok(),
            valid,
            "{pointer}"
        );
    }
    let mut error_result = req.clone();
    error_result["messages"][2]["content"][0]["is_error"] = json!(true);
    let mut empty_directive = req.clone();
    empty_directive["messages"][2]["content"][1]["text"] = json!(
        req["messages"][2]["content"][1]["text"]
            .as_str()
            .unwrap()
            .replace("Your directive: Check the transport.", "Your directive: ")
    );
    let mut extra_text = req.clone();
    extra_text["messages"][1]["content"]
        .as_array_mut()
        .unwrap()
        .push(json!({"type":"text","text":"Unreferenced model output"}));
    for portable in [error_result, empty_directive, extra_text] {
        assert!(protocol::request(&portable, &model()).is_ok());
    }
}

#[test]
fn invalid_tool_json_and_upstream_failures_never_become_success() {
    let mut r = upstream_response(
        json!([{"type":"function_call","call_id":"a","name":"lookup","arguments":"{broken"}]),
    );
    assert!(protocol::response(&r, "openai/gpt-test").is_err());
    r["output"] = json!([]);
    r["status"] = json!("failed");
    assert!(protocol::response(&r, "openai/gpt-test").is_err());
    r["status"] = json!("incomplete");
    r["incomplete_details"] = json!({"reason":"max_output_tokens"});
    assert_eq!(
        protocol::response(&r, "openai/gpt-test")
            .unwrap()
            .stop_reason
            .as_deref(),
        Some("max_tokens")
    );
}

#[tokio::test]
async fn fragmented_stream_preserves_utf8_parallel_arguments_and_completion() {
    use futures::StreamExt;
    let events = stream_fixture();
    let heartbeats = concat!(
        ": keep-alive\r\n\r\n",
        "event: keepalive\r\ndata:\r\n\r\n",
        "event: keepalive\r\ndata: {\"type\":\"keepalive\"}\r\n\r\n",
        "event: ping\r\ndata: {\"type\":\"ping\"}\r\n\r\n",
        "data: {\"type\":\"keepalive\"}\r\n\r\n",
        "event: message\r\ndata: {\"type\":\"ping\"}\r\n\r\n",
    );
    let mut raw: String = events
        .iter()
        .map(|e| {
            format!(
                "{heartbeats}event: {}\r\ndata: {}\r\n\r\n",
                e["type"].as_str().unwrap(),
                e
            )
        })
        .collect();
    raw.push_str(heartbeats);
    let upstream = futures::stream::iter(
        raw.bytes()
            .map(|b| Ok::<_, std::io::Error>(bytes::Bytes::from(vec![b])))
            .collect::<Vec<_>>(),
    );
    let mut decoder = Box::pin(stream::decode(upstream, 1_000_000));
    let mut translator = stream::Translator::new("openai/gpt-test".into());
    let mut output = Vec::new();
    while let Some(event) = decoder.next().await {
        output.extend(
            translator
                .accept(&event.unwrap())
                .unwrap()
                .into_iter()
                .map(|e| serde_json::to_value(e).unwrap()),
        );
    }
    assert!(translator.completed.is_some());
    let starts: Vec<_> = output
        .iter()
        .filter(|e| e["type"] == "content_block_start")
        .collect();
    assert_eq!(starts.len(), 3);
    assert_eq!(starts[1]["content_block"]["id"], "call_a");
    assert_eq!(starts[2]["content_block"]["id"], "call_b");
    let deltas = |index: u64, key: &str| {
        output
            .iter()
            .filter(|e| e["type"] == "content_block_delta" && e["index"] == index)
            .map(|e| e["delta"][key].as_str().unwrap())
            .collect::<String>()
    };
    assert_eq!(deltas(0, "text"), "héllo");
    assert_eq!(deltas(1, "partial_json"), "{\"a\":1}");
    assert_eq!(deltas(2, "partial_json"), "{\"b\":2}");
    let mut open = None;
    for e in &output {
        if e["type"] == "content_block_start" {
            assert!(open.replace(e["index"].clone()).is_none());
        }
        if e["type"] == "content_block_stop" {
            assert_eq!(open.take(), Some(e["index"].clone()));
        }
    }
    assert!(open.is_none());
}

#[tokio::test]
async fn sse_heartbeats_preserve_validation_and_byte_limits() {
    use futures::StreamExt;
    for raw in [
        "event: keepalive\ndata: {broken}\n\n",
        "event: keepalive\ndata: {\"type\":\"response.completed\"}\n\n",
        "event: response.completed\ndata: {\"type\":\"keepalive\"}\n\n",
        "event: ping\ndata: {\"type\":\"keepalive\"}\n\n",
        "data: [DONE]\n\n",
    ]
    .map(str::to_owned)
    .into_iter()
    .chain(["data: {\"type\":\"keepalive\"}\n\n".repeat(100)])
    {
        let source = futures::stream::iter([Ok::<_, std::io::Error>(
            bytes::Bytes::copy_from_slice(raw.as_bytes()),
        )]);
        let mut decoded = Box::pin(stream::decode(source, 1024));
        assert!(decoded.next().await.unwrap().is_err(), "{raw}");
        assert!(decoded.next().await.is_none());
    }
    for event in [
        json!({"type":"error","message":"upstream failed"}),
        json!({"type":"response.unknown_semantic_event"}),
    ] {
        let source = futures::stream::iter([Ok::<_, std::io::Error>(bytes::Bytes::from(format!(
            "data: {{\"type\":\"keepalive\"}}\n\ndata: {event}\n\n"
        )))]);
        let mut decoded = Box::pin(stream::decode(source, 1024));
        assert_eq!(decoded.next().await.unwrap().unwrap(), event);
        assert!(decoded.next().await.is_none());
    }
}

fn stream_fixture() -> Vec<Value> {
    let text = json!({"type":"message","id":"m","role":"assistant","status":"completed","content":[{"type":"output_text","text":"héllo","annotations":[]}]});
    let a = json!({"type":"function_call","id":"f_a","call_id":"call_a","name":"lookup","arguments":"{\"a\":1}","status":"completed"});
    let b = json!({"type":"function_call","id":"f_b","call_id":"call_b","name":"lookup","arguments":"{\"b\":2}","status":"completed"});
    vec![
        json!({"type":"response.created","response":{"id":"resp_test"}}),
        json!({"type":"response.output_item.added","output_index":0,"item":{"id":"m","type":"message","role":"assistant","content":[]}}),
        json!({"type":"response.content_part.added","output_index":0,"item_id":"m","content_index":0,"part":{"type":"output_text","text":"","annotations":[]}}),
        json!({"type":"response.output_text.delta","output_index":0,"item_id":"m","content_index":0,"delta":"hé"}),
        json!({"type":"response.output_text.delta","output_index":0,"item_id":"m","content_index":0,"delta":"llo"}),
        json!({"type":"response.output_text.done","output_index":0,"item_id":"m","content_index":0,"text":"héllo"}),
        json!({"type":"response.content_part.done","output_index":0,"item_id":"m","content_index":0,"part":text["content"][0]}),
        json!({"type":"response.output_item.done","output_index":0,"item":text}),
        json!({"type":"response.output_item.added","output_index":1,"item":{"id":"f_a","type":"function_call","call_id":"call_a","name":"lookup","arguments":""}}),
        json!({"type":"response.function_call_arguments.delta","output_index":1,"item_id":"f_a","delta":"{\"a\":"}),
        json!({"type":"response.output_item.added","output_index":2,"item":{"id":"f_b","type":"function_call","call_id":"call_b","name":"lookup","arguments":""}}),
        json!({"type":"response.function_call_arguments.delta","output_index":2,"item_id":"f_b","delta":"{\"b\":"}),
        json!({"type":"response.function_call_arguments.delta","output_index":1,"item_id":"f_a","delta":"1}"}),
        json!({"type":"response.function_call_arguments.delta","output_index":2,"item_id":"f_b","delta":"2}"}),
        json!({"type":"response.function_call_arguments.done","output_index":2,"item_id":"f_b","arguments":b["arguments"]}),
        json!({"type":"response.output_item.done","output_index":2,"item":b}),
        json!({"type":"response.function_call_arguments.done","output_index":1,"item_id":"f_a","arguments":a["arguments"]}),
        json!({"type":"response.output_item.done","output_index":1,"item":a}),
        json!({"type":"response.completed","response":upstream_response(json!([text,a,b]))}),
    ]
}

fn web_search_stream_fixture() -> Vec<Value> {
    let search = json!({"type":"web_search_call","id":"ws","status":"completed","action":{"type":"search","query":"fixture","sources":[{"type":"url","url":"https://example.org/report"}]}});
    let annotations = json!([
        {"type":"url_citation","url":"https://example.org/report","title":"\n- [Injected](javascript:alert(1))","start_index":0,"end_index":5},
        {"type":"url_citation","url":"https://EXAMPLE.org:443/report","title":"Duplicate","start_index":0,"end_index":2},
        {"type":"url_citation","url":"https://example.net/a%20b","title":"Second source","start_index":2,"end_index":5}
    ]);
    let mut events = stream_fixture();
    for event in &mut events {
        if let Some(index) = event["output_index"].as_u64() {
            event["output_index"] = json!(index + 1);
        }
    }
    events[6]["part"]["annotations"] = annotations.clone();
    events[7]["item"]["content"][0]["annotations"] = annotations.clone();
    events.last_mut().unwrap()["response"]["output"][0]["content"][0]["annotations"] =
        annotations.clone();
    events.last_mut().unwrap()["response"]["output"]
        .as_array_mut()
        .unwrap()
        .insert(0, search.clone());
    events.splice(5..5, annotations.as_array().unwrap().iter().enumerate().map(|(index, annotation)| {
        json!({"type":"response.output_text.annotation.added","output_index":1,"item_id":"m","content_index":0,"annotation_index":index,"annotation":annotation})
    }));
    events.splice(1..1, [
        json!({"type":"response.output_item.added","output_index":0,"item":{"type":"web_search_call","id":"ws","status":"in_progress"}}),
        json!({"type":"response.web_search_call.in_progress","output_index":0,"item_id":"ws"}),
        json!({"type":"response.web_search_call.searching","output_index":0,"item_id":"ws"}),
        json!({"type":"response.web_search_call.completed","output_index":0,"item_id":"ws"}),
        json!({"type":"response.output_item.done","output_index":0,"item":search}),
    ]);
    events
}

#[test]
fn web_search_terminal_items_preserve_the_answer_and_opaque_actions() {
    use crate::models::anthropic::{Delta, StreamEvent};
    let baseline = web_search_stream_fixture();
    let expected =
        protocol::response(&baseline.last().unwrap()["response"], "openai/gpt-test").unwrap();
    for status in ["failed", "incomplete", "completed"] {
        for sparse in [false, true] {
            let mut events = baseline.clone();
            events.retain(|event| {
                status == "completed" || event["type"] != "response.web_search_call.completed"
            });
            let action = if status == "completed" {
                json!({"type":"provider_extension","opaque":[1,2,3]})
            } else {
                baseline.last().unwrap()["response"]["output"][0]["action"].clone()
            };
            let item = json!({"type":"web_search_call","id":"ws","status":status,"action":action});
            events
                .iter_mut()
                .find(|event| {
                    event["type"] == "response.output_item.done" && event["output_index"] == 0
                })
                .unwrap()["item"] = item.clone();
            events.last_mut().unwrap()["response"]["output"][0] = item;
            let native = events.last().unwrap()["response"].clone();
            let response = protocol::response(&native, "openai/gpt-test").unwrap();
            assert_eq!(json!(response.content), json!(expected.content));
            if sparse {
                events.last_mut().unwrap()["response"]["output"] = json!([]);
            }
            let mut translator = stream::Translator::new("openai/gpt-test".into());
            translator.sparse_completion = sparse;
            let mut text = String::new();
            for event in &events {
                for event in translator.accept(event).unwrap() {
                    if let StreamEvent::ContentBlockDelta {
                        delta: Delta::Text { text: delta },
                        ..
                    } = event
                    {
                        text.push_str(&delta);
                    }
                }
            }
            assert_eq!(translator.completed.as_ref(), Some(&native));
            assert_eq!(
                text,
                "héllo\n\nSources:\n- <https://example.org/report>\n- <https://example.net/a%20b>"
            );
            let mut failed = native;
            failed["status"] = json!("failed");
            assert!(protocol::response(&failed, "openai/gpt-test").is_err());
        }
    }
}

#[test]
fn web_search_json_preserves_text_and_validates_citations() {
    let events = web_search_stream_fixture();
    let native = events.last().unwrap()["response"].clone();
    for action in [
        native["output"][0]["action"].clone(),
        json!({"type":"open_page","url":"https://example.org/report"}),
        json!({"type":"find_in_page","url":"https://example.org/report","pattern":"fixture"}),
    ] {
        let mut response = native.clone();
        response["output"][0]["action"] = action;
        let response = protocol::response(&response, "openai/gpt-test").unwrap();
        assert_eq!(
            json!(response.content),
            json!([
                {"type":"text","text":"héllo"},
                {"type":"tool_use","id":"call_a","name":"lookup","input":{"a":1}},
                {"type":"tool_use","id":"call_b","name":"lookup","input":{"b":2}},
                {"type":"text","text":"\n\nSources:\n- <https://example.org/report>\n- <https://example.net/a%20b>"}
            ])
        );
        assert_eq!(response.stop_reason.as_deref(), Some("tool_use"));
    }
    for (pointer, value) in [
        ("/output/0/id", json!(null)),
        ("/output/0/status", json!("searching")),
        ("/output/0/status", json!("in_progress")),
        ("/output/1/content/0/annotations", json!({})),
        (
            "/output/1/content/0/annotations/0/type",
            json!("file_citation"),
        ),
        ("/output/1/content/0/annotations/0/title", json!(42)),
        ("/output/1/content/0/annotations/0/start_index", json!(-1)),
        ("/output/1/content/0/annotations/0/start_index", json!(6)),
        ("/output/1/content/0/annotations/0/end_index", json!(1.5)),
        (
            "/output/1/content/0/annotations/0/url",
            json!("javascript:alert(1)"),
        ),
        ("/output/1/content/0/annotations/0/url", json!("/relative")),
        (
            "/output/1/content/0/annotations/0/url",
            json!("https://user:password@example.org/"),
        ),
        (
            "/output/1/content/0/annotations/0/url",
            json!("https://example.org/\nInjected"),
        ),
    ] {
        let mut changed = native.clone();
        *changed.pointer_mut(pointer).unwrap() = value;
        assert!(
            protocol::response(&changed, "openai/gpt-test").is_err(),
            "{pointer}: {}",
            changed.pointer(pointer).unwrap()
        );
    }
}

#[tokio::test]
async fn web_search_fragmented_sse_matches_json_and_sparse_subscription_output() {
    use futures::StreamExt;
    for sparse in [false, true] {
        let mut events = web_search_stream_fixture();
        let native = events.last().unwrap()["response"].clone();
        if sparse {
            events.last_mut().unwrap()["response"]["output"] = json!([]);
            events.last_mut().unwrap()["response"]
                .as_object_mut()
                .unwrap()
                .remove("status");
        }
        let raw = events.iter().map(|event| format!(
            ": keep-alive\r\n\r\nevent: ping\r\ndata: {{\"type\":\"ping\"}}\r\n\r\nevent: {}\r\ndata: {event}\r\n\r\n",
            event["type"].as_str().unwrap()
        )).collect::<String>();
        let upstream = futures::stream::iter(
            raw.bytes()
                .map(|b| Ok::<_, std::io::Error>(bytes::Bytes::from(vec![b])))
                .collect::<Vec<_>>(),
        );
        let mut decoded = Box::pin(stream::decode(upstream, 1_000_000));
        let mut translator = stream::Translator::new("openai/gpt-test".into());
        translator.sparse_completion = sparse;
        let mut output = Vec::new();
        while let Some(event) = decoded.next().await {
            output.extend(
                translator
                    .accept(&event.unwrap())
                    .unwrap()
                    .into_iter()
                    .map(|e| json!(e)),
            );
        }
        assert_eq!(translator.completed.as_ref(), Some(&native));
        let response = protocol::response(&native, "openai/gpt-test").unwrap();
        let mut content = Vec::<Value>::new();
        let mut open = None;
        let mut arguments = String::new();
        for event in output {
            let index = event["index"].as_u64().map(|index| index as usize);
            match event["type"].as_str().unwrap() {
                "content_block_start" => {
                    assert!(open.replace(index.unwrap()).is_none());
                    assert_eq!(index, Some(content.len()));
                    content.push(event["content_block"].clone());
                    arguments.clear();
                }
                "content_block_delta" => {
                    assert_eq!(index, open);
                    match event["delta"]["type"].as_str().unwrap() {
                        "text_delta" => {
                            let block = &mut content[index.unwrap()];
                            block["text"] = json!(format!(
                                "{}{}",
                                block["text"].as_str().unwrap(),
                                event["delta"]["text"].as_str().unwrap()
                            ));
                        }
                        "input_json_delta" => {
                            arguments.push_str(event["delta"]["partial_json"].as_str().unwrap())
                        }
                        other => panic!("unexpected delta: {other}"),
                    }
                }
                "content_block_stop" => {
                    assert_eq!(open.take(), index);
                    let block = &mut content[index.unwrap()];
                    if block["type"] == "tool_use" {
                        block["input"] = serde_json::from_str(&arguments).unwrap();
                    }
                }
                "message_start" => {}
                other => panic!("unexpected event: {other}"),
            }
        }
        assert!(open.is_none());
        assert_eq!(json!(content), json!(response.content));
        assert_eq!(json!(stream::finish(&response))[1]["type"], "message_stop");
    }
}

#[test]
fn web_search_sse_rejects_invalid_progress_and_changed_annotations() {
    for (kind, pointer, value) in [
        (
            "response.web_search_call.searching",
            "/item_id",
            json!("wrong"),
        ),
        (
            "response.web_search_call.searching",
            "/output_index",
            json!(1),
        ),
        (
            "response.web_search_call.searching",
            "/type",
            json!("response.web_search_call.failed"),
        ),
        (
            "response.web_search_call.searching",
            "/type",
            json!("response.web_search_call.unknown"),
        ),
        (
            "response.output_item.done",
            "/item/status",
            json!("in_progress"),
        ),
        ("response.output_item.done", "/item/id", json!("wrong")),
        (
            "response.output_text.annotation.added",
            "/item_id",
            json!("wrong"),
        ),
        (
            "response.output_text.annotation.added",
            "/annotation_index",
            json!(1),
        ),
        (
            "response.output_text.annotation.added",
            "/annotation/type",
            json!("file_citation"),
        ),
        (
            "response.output_text.annotation.added",
            "/annotation/url",
            json!("javascript:alert(1)"),
        ),
        (
            "response.content_part.done",
            "/part/annotations/0/url",
            json!("https://changed.example.org/"),
        ),
        ("response.content_part.done", "/part/annotations", json!([])),
        (
            "response.completed",
            "/response/output/1/content/0/annotations/0/url",
            json!("https://changed.example.org/"),
        ),
    ] {
        let mut events = web_search_stream_fixture();
        let event = events
            .iter_mut()
            .find(|event| event["type"] == kind)
            .unwrap();
        *event.pointer_mut(pointer).unwrap() = value;
        let mut translator = stream::Translator::new("openai/gpt-test".into());
        let result = events
            .iter()
            .try_for_each(|event| translator.accept(event).map(|_| ()));
        assert!(result.is_err(), "{kind} {pointer}");
        assert!(translator.completed.is_none(), "{kind} {pointer}");
    }
    let mut events = web_search_stream_fixture();
    events.last_mut().unwrap()["type"] = json!("response.failed");
    events.last_mut().unwrap()["response"]["status"] = json!("failed");
    let mut translator = stream::Translator::new("openai/gpt-test".into());
    assert!(
        events
            .iter()
            .try_for_each(|event| translator.accept(event).map(|_| ()))
            .is_err()
    );
    assert!(translator.completed.is_none());
}

#[test]
fn subscription_sparse_completions_preserve_output_and_ignore_metadata() {
    for sparse_output in [None, Some(Value::Null), Some(json!([]))] {
        let mut events = stream_fixture();
        let expected = events.last().unwrap()["response"]["output"].clone();
        let terminal = events.last_mut().unwrap()["response"]
            .as_object_mut()
            .unwrap();
        terminal.remove("status");
        terminal.remove("output");
        if let Some(output) = sparse_output {
            terminal.insert("output".into(), output);
        }
        events.insert(
            0,
            json!({"type":"response.metadata","metadata":{"safety_buffering":true}}),
        );
        events.insert(2, json!({"type":"codex.response.metadata","metadata":{}}));
        let mut translator = stream::Translator::new("openai/gpt-test".into());
        translator.sparse_completion = true;
        for event in events {
            translator.accept(&event).unwrap();
        }
        let response = translator.completed.unwrap();
        assert_eq!(response["output"], expected);
        assert_eq!(response["status"], "completed");
    }
    let mut events = stream_fixture();
    events.retain(|e| !(e["type"] == "response.output_item.done" && e["output_index"] == 2));
    events.last_mut().unwrap()["response"]["output"] = json!([]);
    let mut translator = stream::Translator::new("openai/gpt-test".into());
    translator.sparse_completion = true;
    let terminal = events.pop().unwrap();
    for event in events {
        translator.accept(&event).unwrap();
    }
    assert!(translator.accept(&terminal).is_err());
}

async fn serve(router: axum::Router) -> (String, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let task = tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    (url, task)
}

fn config(base_url: String, state_dir: std::path::PathBuf) -> crate::config::Config {
    crate::config::Config {
        server: crate::config::Server {
            state_dir,
            auth_token: Some("local-secret".into()),
            keep_alive_seconds: 1,
            ..Default::default()
        },
        providers: [(
            "openai".into(),
            ProviderConfig::OpenAi(crate::providers::openai::models::Config {
                base_url,
                auth: OpenAiAuth::ApiKey("upstream-secret".into()),
                organization: None,
                project: None,
                models: [(
                    "gpt-test".into(),
                    ModelOptions {
                        reasoning_effort: Some(ReasoningEffort::Medium),
                        ..Default::default()
                    },
                )]
                .into(),
            }),
        )]
        .into(),
        logging: Default::default(),
    }
}

struct HttpFixture {
    gateway: String,
    url: String,
    client: reqwest::Client,
    directory: std::path::PathBuf,
    tasks: [tokio::task::JoinHandle<()>; 2],
    maximum: Arc<AtomicUsize>,
}

impl HttpFixture {
    async fn close(self) {
        for task in self.tasks {
            task.abort();
            let _ = task.await;
        }
        let _ = tokio::fs::remove_dir_all(self.directory).await;
    }
}

async fn http_fixture(prefix: &str, auth_token: Option<&str>) -> HttpFixture {
    use axum::{Json, Router, response::IntoResponse, routing::post};
    let active = Arc::new(AtomicUsize::new(0));
    let maximum = Arc::new(AtomicUsize::new(0));
    let barrier = Arc::new(tokio::sync::Barrier::new(33));
    let (upstream, up_task) = serve(Router::new().route("/responses",post({
        let active = active.clone(); let maximum = maximum.clone();
        move |headers: axum::http::HeaderMap, Json(req): Json<Value>| {
            let active = active.clone(); let maximum = maximum.clone();
            let barrier = barrier.clone();
            async move {
                assert_eq!(headers["authorization"],"Bearer upstream-secret");
                assert!(matches!(req["model"].as_str(), Some("gpt-test" | "Claude.Native/Version")));
                maximum.fetch_max(active.fetch_add(1,Ordering::SeqCst)+1,Ordering::SeqCst);
                if req["max_output_tokens"] == 33 { barrier.wait().await; }
                tokio::time::sleep(std::time::Duration::from_millis(80)).await;
                active.fetch_sub(1,Ordering::SeqCst);
                if req["max_output_tokens"] == 13 {return (axum::http::StatusCode::TOO_MANY_REQUESTS,[("retry-after","2")],Json(json!({"error":{"message":"quota exhausted","type":"rate_limit_error","code":"rate_limit_exceeded"}}))).into_response();}
                if req["stream"] == true {
                    let events = stream_fixture();
                    let data = events.iter().map(|e| format!("event: {}\ndata: {}\n\n",e["type"].as_str().unwrap(),e)).collect::<String>();
                    ([("content-type","text/event-stream")],data).into_response()
                } else {Json(upstream_response(json!([{"type":"message","id":"m","role":"assistant","content":[{"type":"output_text","text":"hello","annotations":[]}]}]))).into_response()}
            }
        }
    }))).await;
    let directory = std::env::temp_dir().join(format!("tinyllm-http-{}", uuid::Uuid::new_v4()));
    let mut cfg = config(upstream, directory.clone());
    cfg.server.auth_token = auth_token.map(str::to_owned);
    let provider = cfg.providers.remove("openai").unwrap();
    cfg.providers.insert(prefix.into(), provider);
    let (gateway, gw_task) = serve(crate::server::router(cfg).await.unwrap()).await;
    HttpFixture {
        url: format!("{gateway}/anthropic/v1/messages?beta=true"),
        gateway,
        client: reqwest::Client::new(),
        directory,
        tasks: [gw_task, up_task],
        maximum,
    }
}

#[tokio::test]
async fn continuation_headers_report_portable_and_mixed_histories() {
    let fixture = http_fixture("openai", Some("local-secret")).await;
    for chat in [false, true] {
        let url = format!(
            "{}{}",
            fixture.gateway,
            if chat {
                "/v1/chat/completions"
            } else {
                "/anthropic/v1/messages"
            }
        );
        let base = json!({"model":"openai/gpt-test","max_tokens":32,"messages":[{"role":"user","content":"hello"}]});
        let response = fixture
            .client
            .post(&url)
            .bearer_auth("local-secret")
            .json(&base)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        assert_eq!(response.headers()["x-tinyllm-continuation"], "fresh");
        let _: Value = response.json().await.unwrap();
        let carrier = "tinyllm:v1:cnNfMQ:opaque";
        let assistant = if chat {
            json!({"role":"assistant","content":"answer","reasoning_details":[{"type":"tinyllm_continuation","data":carrier}]})
        } else {
            json!({"role":"assistant","content":[{"type":"redacted_thinking","data":carrier},{"type":"text","text":"answer"}]})
        };
        let portable = json!({"role":"assistant","content":"legacy answer"});
        let foreign = "tinyllm:v0:unreadable";
        let missing = if chat {
            json!({"role":"assistant","content":"old answer","reasoning_details":[{"type":"tinyllm_continuation","data":foreign}]})
        } else {
            json!({"role":"assistant","content":[{"type":"redacted_thinking","data":foreign},{"type":"text","text":"old answer"}]})
        };
        let mut restored = base.clone();
        restored["messages"].as_array_mut().unwrap().extend([
            assistant.clone(),
            json!({"role":"user","content":"continue"}),
        ]);
        let mut plain = base.clone();
        plain["messages"]
            .as_array_mut()
            .unwrap()
            .extend([portable, json!({"role":"user","content":"continue"})]);
        let mut absent = base.clone();
        absent["messages"]
            .as_array_mut()
            .unwrap()
            .extend([missing, json!({"role":"user","content":"continue"})]);
        let mut mixed = plain.clone();
        mixed["messages"]
            .as_array_mut()
            .unwrap()
            .extend([assistant, json!({"role":"user","content":"continue again"})]);
        for streaming in [false, true] {
            for (source, status) in [
                (&restored, "restored"),
                (&plain, "fresh"),
                (&absent, "fresh"),
                (&mixed, "restored"),
            ] {
                let mut req = source.clone();
                req["stream"] = json!(streaming);
                let response = fixture
                    .client
                    .post(&url)
                    .bearer_auth("local-secret")
                    .json(&req)
                    .send()
                    .await
                    .unwrap();
                let http_status = response.status();
                let continuation = response
                    .headers()
                    .get("x-tinyllm-continuation")
                    .map(|value| value.to_str().unwrap().to_owned());
                let body = response.text().await.unwrap();
                assert_eq!(http_status, 200, "chat={chat} {status}: {body}");
                assert_eq!(continuation.as_deref(), Some(status));
                if streaming {
                    assert!(
                        body.contains(if chat {
                            "[DONE]"
                        } else {
                            "event: message_stop"
                        }),
                        "{body}"
                    );
                    assert!(!body.contains("event: error"), "{body}");
                }
            }
        }
    }
    fixture.close().await;
}

#[tokio::test]
async fn http_auth_rejects_missing_and_invalid_client_credentials() {
    let fixture = http_fixture("openai", Some("local-secret")).await;
    for header in [None, Some("authorization"), Some("x-api-key")] {
        let mut call = fixture.client.post(&fixture.url).json(&request());
        if let Some(header) = header {
            call = call.header(header, "Bearer wrong-token");
        }
        assert_eq!(call.send().await.unwrap().status(), 401);
    }
    // The preconnect probe is the one route that answers without credentials.
    let hello = format!("{}/anthropic/api/hello", fixture.gateway);
    for response in [
        fixture.client.get(&hello).send().await.unwrap(),
        fixture.client.head(&hello).send().await.unwrap(),
        fixture
            .client
            .get(&hello)
            .header("x-api-key", "wrong-token")
            .send()
            .await
            .unwrap(),
    ] {
        assert_eq!(response.status(), 200);
    }
    // It must not reveal anything, and must not open any other route.
    let body = fixture
        .client
        .get(&hello)
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert_eq!(body, r#"{"ok":true}"#);
    for path in [
        "/anthropic/v1/models",
        "/anthropic/v1/messages/count_tokens",
    ] {
        let response = fixture
            .client
            .get(format!("{}{path}", fixture.gateway))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 401, "{path} must stay authenticated");
    }
    fixture.close().await;
}

#[tokio::test]
async fn http_protocol_preserves_json_streams_and_concurrent_requests() {
    let fixture = http_fixture("openai", Some("local-secret")).await;
    let mut namespaced = request();
    namespaced["model"] = json!("openai/gpt-test");
    let (a, b) = tokio::join!(
        fixture
            .client
            .post(&fixture.url)
            .bearer_auth("local-secret")
            .json(&namespaced)
            .send(),
        fixture
            .client
            .post(&fixture.url)
            .header("x-api-key", "local-secret")
            .json(&request())
            .send()
    );
    let a: Value = a.unwrap().json().await.unwrap();
    let b: Value = b.unwrap().json().await.unwrap();
    assert_eq!(a["model"], "openai/gpt-test");
    assert_eq!(b["model"], "openai/gpt-test");
    assert_eq!(a["content"][0]["text"], "hello");
    assert_eq!(b["content"][0]["text"], "hello");
    assert!(fixture.maximum.load(Ordering::SeqCst) >= 2);
    let mut req = namespaced;
    req["stream"] = json!(true);
    let response = fixture
        .client
        .post(&fixture.url)
        .bearer_auth("local-secret")
        .json(&req)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let text = response.text().await.unwrap();
    assert!(text.contains("event: message_stop"), "{text}");
    assert!(!text.contains("event: error"), "{text}");
    fixture.close().await;
}

#[tokio::test]
async fn streamed_stop_sequence_suppresses_later_tool_calls() {
    let fixture = http_fixture("openai", Some("local-secret")).await;
    let mut req = request();
    req["stream"] = json!(true);
    req["stop_sequences"] = json!(["éll"]);
    let response = fixture
        .client
        .post(&fixture.url)
        .bearer_auth("local-secret")
        .json(&req)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    assert_eq!(
        response.headers()["x-tinyllm-stop-sequences"],
        "local; usage-includes-discarded-output"
    );
    let body = response.text().await.unwrap();
    let events: Vec<Value> = body
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    let text: String = events
        .iter()
        .filter_map(|e| e["delta"]["text"].as_str())
        .collect();
    assert_eq!(text, "h");
    assert!(!body.contains("tool_use"), "{body}");
    assert!(!body.contains("event: error"), "{body}");
    assert_eq!(events.last().unwrap()["type"], "message_stop");
    let delta = events
        .iter()
        .find(|e| e["type"] == "message_delta")
        .unwrap();
    assert_eq!(delta["delta"]["stop_sequence"], "éll");
    assert_eq!(delta["delta"]["stop_reason"], "stop_sequence");
    assert_eq!(delta["usage"]["output_tokens"], 25);
    fixture.close().await;
}

#[tokio::test]
async fn http_concurrency_is_uncapped_by_default() {
    let fixture = http_fixture("openai", Some("local-secret")).await;
    let mut concurrent = request();
    concurrent["max_tokens"] = json!(33);
    let responses = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        futures::future::join_all((0..33).map(|_| {
            fixture
                .client
                .post(&fixture.url)
                .bearer_auth("local-secret")
                .json(&concurrent)
                .send()
        })),
    )
    .await
    .expect("all 33 requests must reach upstream concurrently without a local cap");
    for response in responses {
        assert_eq!(response.unwrap().status(), 200);
    }
    assert_eq!(fixture.maximum.load(Ordering::SeqCst), 33);
    fixture.close().await;
}

#[tokio::test]
async fn http_errors_preserve_upstream_status_retry_and_type() {
    let fixture = http_fixture("openai", Some("local-secret")).await;
    let mut req = request();
    req["max_tokens"] = json!(13);
    let error = fixture
        .client
        .post(&fixture.url)
        .bearer_auth("local-secret")
        .json(&req)
        .send()
        .await
        .unwrap();
    assert_eq!(error.status(), 429);
    assert_eq!(error.headers()["retry-after"], "2");
    assert_eq!(
        error.json::<Value>().await.unwrap()["error"]["type"],
        "rate_limit_error"
    );
    fixture.close().await;
}

#[tokio::test]
async fn http_discovery_and_model_names_use_provider_prefixes() {
    let fixture = http_fixture("openai", Some("local-secret")).await;
    for path in ["/anthropic/v1/models", "/v1/models"] {
        let models = fixture
            .client
            .get(format!("{}{path}", fixture.gateway))
            .bearer_auth("local-secret")
            .send()
            .await
            .unwrap()
            .json::<Value>()
            .await
            .unwrap();
        assert_eq!(
            models["data"]
                .as_array()
                .unwrap()
                .iter()
                .map(|model| model["id"].as_str().unwrap())
                .collect::<Vec<_>>(),
            ["openai/gpt-test", "openai/gpt-test-fast"]
        );
    }
    for model in [
        "gpt-test",
        "main",
        "openrouter/gpt-test",
        "openai/",
        "openai//gpt-test",
    ] {
        let mut invalid = request();
        invalid["model"] = json!(model);
        assert_eq!(
            fixture
                .client
                .post(&fixture.url)
                .bearer_auth("local-secret")
                .json(&invalid)
                .send()
                .await
                .unwrap()
                .status(),
            400,
            "invalid public model: {model}"
        );
    }
    let mut native = request();
    native["model"] = json!("openai/Claude.Native/Version");
    let response = fixture
        .client
        .post(&fixture.url)
        .bearer_auth("local-secret")
        .json(&native)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    assert_eq!(
        response.json::<Value>().await.unwrap()["model"],
        "openai/Claude.Native/Version"
    );
    fixture.close().await;
}

#[tokio::test]
async fn http_unauthenticated_mode_accepts_requests_with_any_client_credentials() {
    let fixture = http_fixture("codex", None).await;
    let mut namespaced = request();
    namespaced["model"] = json!("codex/gpt-test");
    for header in [None, Some("authorization"), Some("x-api-key")] {
        let mut call = fixture.client.post(&fixture.url).json(&namespaced);
        if let Some(header) = header {
            call = call.header(header, "ignored-token");
        }
        let response = call.send().await.unwrap();
        assert_eq!(response.status(), 200);
        assert_eq!(
            response.json::<Value>().await.unwrap()["content"][0]["text"],
            "hello"
        );
    }
    assert_eq!(
        fixture
            .client
            .get(format!("{}/anthropic/v1/models", fixture.gateway))
            .send()
            .await
            .unwrap()
            .status(),
        200
    );
    fixture.close().await;
}

#[tokio::test]
async fn disconnect_cancels_upstream_for_stream_and_json() {
    use axum::{Router, body::Body, routing::post};
    use futures::StreamExt;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    struct Active(Arc<AtomicUsize>);
    impl Drop for Active {
        fn drop(&mut self) {
            self.0.fetch_sub(1, Ordering::SeqCst);
        }
    }
    for streaming in [true, false] {
        let active = Arc::new(AtomicUsize::new(0));
        let (upstream, up_task) = serve(Router::new().route("/responses",post({let active=active.clone(); move || {
            active.fetch_add(1,Ordering::SeqCst);
            let guard=Active(active.clone());
            async move {
                let stream = async_stream::stream! {
                    let _guard=guard;
                    let first = if streaming {stream_fixture()[..5].iter().map(|event|format!("event: {}\ndata: {event}\n\n",event["type"].as_str().unwrap())).collect::<String>()} else {"{\"id\":".into()};
                    yield Ok::<_,std::io::Error>(bytes::Bytes::from(first));
                    loop {tokio::time::sleep(std::time::Duration::from_millis(50)).await; yield Ok(bytes::Bytes::from_static(b" "));}
                };
                ([("content-type",if streaming {"text/event-stream"} else {"application/json"})],Body::from_stream(stream))
            }
        }}))).await;
        let dir = std::env::temp_dir().join(format!("tinyllm-cancel-{}", uuid::Uuid::new_v4()));
        let mut cfg = config(upstream, dir.clone());
        cfg.server.max_concurrent_requests = Some(1);
        let (gateway, gw_task) = serve(crate::server::router(cfg).await.unwrap()).await;
        let mut req = request();
        req["stream"] = json!(streaming);
        req["stop_sequences"] = json!(["éll"]);
        let url = format!("{gateway}/anthropic/v1/messages");
        let call = tokio::spawn(async move {
            let response = reqwest::Client::new()
                .post(format!("{gateway}/anthropic/v1/messages"))
                .bearer_auth("local-secret")
                .json(&req)
                .send()
                .await
                .unwrap();
            if streaming {
                let events = stream::decode(response.bytes_stream(), 10_000);
                futures::pin_mut!(events);
                loop {
                    let event = events.next().await.unwrap().unwrap();
                    if event["type"] == "content_block_stop" && event["index"] == 0 {
                        break;
                    }
                }
            } else {
                response.text().await.unwrap();
            }
        });
        tokio::time::timeout(std::time::Duration::from_secs(2), async {
            while active.load(Ordering::SeqCst) == 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        if streaming {
            call.await.unwrap();
        } else {
            call.abort();
            let _ = call.await;
        }
        let stopped = tokio::time::timeout(std::time::Duration::from_secs(2), async {
            while active.load(Ordering::SeqCst) != 0 {
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
        })
        .await;
        assert_eq!(
            reqwest::Client::new()
                .post(&url)
                .bearer_auth("local-secret")
                .header("content-type", "application/json")
                .body("invalid JSON")
                .send()
                .await
                .unwrap()
                .status(),
            400,
            "disconnect must release the local concurrency permit"
        );
        gw_task.abort();
        up_task.abort();
        let _ = tokio::fs::remove_dir_all(dir).await;
        assert!(
            stopped.is_ok(),
            "upstream continued after disconnect; streaming={streaming}"
        );
    }
}

#[tokio::test]
async fn stream_failure_truncation_and_bounds_are_explicit() {
    use futures::StreamExt;
    let mut translator = stream::Translator::new("openai/gpt-test".into());
    translator
        .accept(&json!({"type":"response.created","response":{"id":"r"}}))
        .unwrap();
    assert!(translator.accept(&json!({"type":"response.failed","response":{"id":"r","error":{"message":"failure"}}})).is_err());
    let raw = futures::stream::iter(vec![Ok::<_, std::io::Error>(bytes::Bytes::from(vec![
        b'x';
        1025
    ]))]);
    let decoded = stream::decode(raw, 1024);
    futures::pin_mut!(decoded);
    assert!(decoded.next().await.unwrap().is_err());
    let dir = std::env::temp_dir().join(format!("tinyllm-limits-{}", uuid::Uuid::new_v4()));
    let mut cfg = config("http://127.0.0.1:1".into(), dir.clone());
    cfg.server.max_request_bytes = 32;
    let (gateway, task) = serve(crate::server::router(cfg).await.unwrap()).await;
    let response = reqwest::Client::new()
        .post(format!("{gateway}/anthropic/v1/messages"))
        .bearer_auth("local-secret")
        .json(&request())
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 413);
    assert_eq!(response.json::<Value>().await.unwrap()["type"], "error");
    task.abort();
    let _ = tokio::fs::remove_dir_all(dir).await;
}

#[tokio::test]
async fn missing_continuations_degrade_but_duplicates_and_limits_still_fail() {
    let native = upstream_response(
        json!([{"type":"reasoning","id":"r","summary":[],"encrypted_content":"x".repeat(4400)}, {"type":"message","id":"m","role":"assistant","content":[{"type":"output_text","text":"done","annotations":[]}]}]),
    );
    let content = serde_json::to_value(
        protocol::response(&native, "openai/gpt-test")
            .unwrap()
            .content,
    )
    .unwrap();
    let mut req = request();
    req["messages"] = json!([{"role":"assistant","content":[{"type":"text","text":"done"}]}]);
    let replay = protocol::request(&req, &model()).unwrap();
    assert_eq!(replay["input"][1]["content"][0]["text"], "done");
    // The same carrier replayed twice is history, not a conflict.
    req["messages"] = json!([{"role":"assistant","content":content},{"role":"user","content":"next"},{"role":"assistant","content":content}]);
    let replay = protocol::request(&req, &model()).unwrap();
    let reasoning: Vec<_> = replay["input"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|item| item["type"] == "reasoning")
        .collect();
    assert_eq!(reasoning.len(), 2);
    assert_eq!(reasoning[0]["encrypted_content"], "x".repeat(4400));
}

#[tokio::test]
async fn subscription_json_stream_tools_reasoning_and_concurrency() {
    use axum::{Json, Router, response::IntoResponse, routing::post};
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    let active = Arc::new(AtomicUsize::new(0));
    let maximum = Arc::new(AtomicUsize::new(0));
    let replays = Arc::new(AtomicUsize::new(0));
    let reasoning =
        json!({"type":"reasoning","id":"rs_1","summary":[],"encrypted_content":"opaque-reasoning"});
    let mut events = stream_fixture();
    for event in &mut events {
        if let Some(index) = event["output_index"].as_u64() {
            event["output_index"] = json!(index + 1);
        }
    }
    events.splice(1..1, [
        json!({"type":"response.output_item.added","output_index":0,"item":{"type":"reasoning","id":"rs_1","summary":[]}}),
        json!({"type":"response.output_item.done","output_index":0,"item":reasoning}),
    ]);
    events.last_mut().unwrap()["response"]["output"] = json!([]);
    let raw = events
        .iter()
        .map(|e| format!("data: {{\"type\":\"ping\"}}\n\nevent: {}\ndata: {e}\n\nevent: keepalive\ndata: {{\"type\":\"keepalive\"}}\n\n", e["type"].as_str().unwrap()))
        .collect::<String>();
    let (upstream, up_task) = serve(Router::new().route(
        "/responses",
        post({
            let active = active.clone();
            let maximum = maximum.clone();
            let replays = replays.clone();
            move |headers: axum::http::HeaderMap, Json(req): Json<Value>| {
                let active = active.clone();
                let maximum = maximum.clone();
                let replays = replays.clone();
                let raw = raw.clone();
                let reasoning = reasoning.clone();
                async move {
                    assert_eq!(headers["authorization"], "Bearer fixture-key");
                    assert_eq!(headers["chatgpt-account-id"], "account");
                    assert_eq!(headers["originator"], "tinyllm");
                    assert_eq!(req["stream"], true);
                    assert_eq!(req["service_tier"], "priority");
                    assert_eq!(req["instructions"], "Keep ALL instructions.");
                    assert!(req.get("max_output_tokens").is_none());
                    assert!(req.get("truncation").is_none());
                    if req["input"][0]["content"][0]["text"] == "fail" {
                        return (
                            axum::http::StatusCode::TOO_MANY_REQUESTS,
                            Json(json!({"error":{"message":"limit fixture-key"}})),
                        )
                            .into_response();
                    }
                    if req["input"][0]["content"][0]["text"] == "unsupported-model" {
                        return (
                            axum::http::StatusCode::BAD_REQUEST,
                            Json(json!({"detail":"unsupported model fixture-key"})),
                        )
                            .into_response();
                    }
                    maximum.fetch_max(active.fetch_add(1, Ordering::SeqCst) + 1, Ordering::SeqCst);
                    tokio::time::sleep(std::time::Duration::from_millis(40)).await;
                    active.fetch_sub(1, Ordering::SeqCst);
                    if req["input"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .any(|i| i["type"] == "function_call_output")
                    {
                        assert!(req["input"].as_array().unwrap().contains(&reasoning));
                        assert!(
                            req["input"]
                                .as_array()
                                .unwrap()
                                .iter()
                                .any(|i| i["role"] == "developer"
                                    && i["content"][0]["text"] == "background notification")
                        );
                        replays.fetch_add(1, Ordering::SeqCst);
                    }
                    let mut response = raw.into_response();
                    if req["input"][0]["content"][0]["text"] != "wrong-content-type" {
                        response.headers_mut().remove("content-type");
                    }
                    response
                }
            }
        }),
    ))
    .await;
    let directory =
        std::env::temp_dir().join(format!("tinyllm-subscription-{}", uuid::Uuid::new_v4()));
    let auth_dir = directory.join("auth");
    let mut builder = std::fs::DirBuilder::new();
    builder.recursive(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    builder.create(&auth_dir).unwrap();
    let path = auth_dir.join("openai.json");
    std::fs::write(&path, json!({"access_token":"fixture-key","refresh_token":"refresh","account_id":"account","expires_at":u64::MAX}).to_string()).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
    }
    let mut cfg = config(upstream, directory.join("state"));
    let ProviderConfig::OpenAi(provider) = cfg.providers.get_mut("openai").unwrap() else {
        panic!("wrong provider variant")
    };
    provider.auth = OpenAiAuth::Subscription(SubscriptionOptions {
        credentials_dir: auth_dir,
    });
    provider.models.get_mut("gpt-test").unwrap().service_tier = Some(ServiceTier::Fast);
    let (gateway, gw_task) = serve(crate::server::router(cfg).await.unwrap()).await;
    let url = format!("{gateway}/anthropic/v1/messages");
    let client = reqwest::Client::new();
    let mut streamed = request();
    streamed["stream"] = json!(true);
    let (json, sse) = tokio::join!(
        client
            .post(&url)
            .bearer_auth("local-secret")
            .json(&request())
            .send(),
        client
            .post(&url)
            .bearer_auth("local-secret")
            .json(&streamed)
            .send()
    );
    let json = json.unwrap();
    assert_eq!(json.status(), 200);
    assert!(
        json.headers()["x-tinyllm-compatibility"]
            .to_str()
            .unwrap()
            .contains("max-tokens-unenforced")
    );
    let response: Value = json.json().await.unwrap();
    assert_eq!(response["stop_reason"], "tool_use");
    assert_eq!(
        response["content"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|b| b["type"] == "tool_use")
            .count(),
        2
    );
    let sse = sse.unwrap().text().await.unwrap();
    assert!(sse.contains("input_json_delta") && sse.contains("message_stop"));
    assert!(!sse.contains("event: error"));
    assert!(maximum.load(Ordering::SeqCst) >= 2);
    let calls: Vec<_> = response["content"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|block| block["type"] == "tool_use")
        .collect();
    let mut followup = request();
    followup["messages"].as_array_mut().unwrap().extend([
        json!({"role":"assistant","content":response["content"]}),
        json!({"role":"user","content":[{"type":"tool_result","tool_use_id":calls[0]["id"],"content":"ok"},{"type":"tool_result","tool_use_id":calls[1]["id"],"is_error":true,"content":"denied"}]}),
        json!({"role":"system","content":"background notification"}),
    ]);
    assert_eq!(
        client
            .post(&url)
            .bearer_auth("local-secret")
            .json(&followup)
            .send()
            .await
            .unwrap()
            .status(),
        200
    );
    assert_eq!(replays.load(Ordering::SeqCst), 1);
    let mut failed = request();
    failed["messages"][0]["content"] = json!("unsupported-model");
    let response = client
        .post(&url)
        .bearer_auth("local-secret")
        .json(&failed)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 400);
    assert_eq!(
        response.json::<Value>().await.unwrap()["error"]["message"],
        "unsupported model [redacted]"
    );
    failed["messages"][0]["content"] = json!("wrong-content-type");
    assert_eq!(
        client
            .post(&url)
            .bearer_auth("local-secret")
            .json(&failed)
            .send()
            .await
            .unwrap()
            .status(),
        502
    );
    failed["messages"][0]["content"] = json!("fail");
    let response = client
        .post(&url)
        .bearer_auth("local-secret")
        .json(&failed)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 429);
    assert!(!response.text().await.unwrap().contains("fixture-key"));
    failed["temperature"] = json!(0.5);
    assert_eq!(
        client
            .post(&url)
            .bearer_auth("local-secret")
            .json(&failed)
            .send()
            .await
            .unwrap()
            .status(),
        400
    );
    gw_task.abort();
    up_task.abort();
    let _ = gw_task.await;
    let _ = std::fs::remove_dir_all(directory);
}

#[tokio::test]
async fn auto_review_subrequests_reach_the_configured_reviewer() {
    use axum::{Json, Router, routing::post};
    let seen = Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
    let (upstream, up_task) = serve(Router::new().route(
        "/responses",
        post({
            let seen = seen.clone();
            move |Json(req): Json<Value>| {
                seen.lock()
                    .unwrap()
                    .push(req["model"].as_str().unwrap_or_default().to_owned());
                async move {
                    Json(upstream_response(
                        json!([{"type":"message","id":"m","role":"assistant","content":[{"type":"output_text","text":"allow","annotations":[]}]}]),
                    ))
                }
            }
        }),
    ))
    .await;
    let directory = std::env::temp_dir().join(format!("tinyllm-review-{}", uuid::Uuid::new_v4()));
    let mut cfg = config(upstream, directory.clone());
    cfg.server.auto_review_model = Some("openai/gpt-reviewer".into());
    let (gateway, task) = serve(crate::server::router(cfg).await.unwrap()).await;
    let url = format!("{gateway}/anthropic/v1/messages");
    let send = |body: Value| {
        let url = url.clone();
        async move {
            reqwest::Client::new()
                .post(&url)
                .bearer_auth("local-secret")
                .json(&body)
                .send()
                .await
                .unwrap()
                .status()
        }
    };

    let monitor = "You are a security monitor for autonomous AI coding agents.\n\n## Context";
    assert_eq!(
        send(
            json!({"model":"openai/gpt-test","max_tokens":512,"system":monitor,
                    "messages":[{"role":"user","content":"rm -rf /"}]})
        )
        .await,
        200
    );
    // An ordinary turn must stay on the session's model.
    assert_eq!(
        send(
            json!({"model":"openai/gpt-test","max_tokens":512,"system":"You are Claude Code.",
                    "messages":[{"role":"user","content":"hello"}]})
        )
        .await,
        200
    );
    task.abort();
    up_task.abort();
    assert_eq!(
        seen.lock().unwrap().as_slice(),
        ["gpt-reviewer", "gpt-test"],
        "only the classifier is rerouted"
    );
    let _ = tokio::fs::remove_dir_all(directory).await;
}

#[tokio::test]
async fn streamed_read_arguments_are_repaired_before_the_client_sees_them() {
    let call = json!({"type":"function_call","id":"f","call_id":"call_r","name":"Read",
                      "arguments":"{\"file_path\":\"/tmp/a\",\"offset\":1300000}","status":"completed"});
    let events = vec![
        json!({"type":"response.created","response":{"id":"resp_test"}}),
        json!({"type":"response.output_item.added","output_index":0,
               "item":{"id":"f","type":"function_call","call_id":"call_r","name":"Read","arguments":""}}),
        // The model streams the bad offset in fragments.
        json!({"type":"response.function_call_arguments.delta","output_index":0,"item_id":"f",
               "delta":"{\"file_path\":\"/tmp/a\","}),
        json!({"type":"response.function_call_arguments.delta","output_index":0,"item_id":"f",
               "delta":"\"offset\":1300000}"}),
        json!({"type":"response.function_call_arguments.done","output_index":0,"item_id":"f",
               "arguments":"{\"file_path\":\"/tmp/a\",\"offset\":1300000}"}),
        json!({"type":"response.output_item.done","output_index":0,"item":call}),
        json!({"type":"response.completed","response":upstream_response(json!([call]))}),
    ];
    let mut translator = stream::Translator::new("openai/gpt-test".into());
    let mut out = Vec::new();
    let mut deltas_before_done = 0;
    for event in &events {
        for emitted in translator.accept(event).unwrap() {
            let value = serde_json::to_value(&emitted).unwrap();
            if value["type"] == "content_block_delta"
                && event["type"] == "response.function_call_arguments.delta"
            {
                deltas_before_done += 1;
            }
            out.push(value);
        }
    }
    assert_eq!(
        deltas_before_done, 0,
        "tool arguments must not stream out before they can be repaired"
    );
    let json: String = out
        .iter()
        .filter(|e| e["type"] == "content_block_delta")
        .map(|e| e["delta"]["partial_json"].as_str().unwrap())
        .collect();
    let input: Value = serde_json::from_str(&json).unwrap();
    assert!(input.get("offset").is_none(), "{input}");
    assert_eq!(input["file_path"], "/tmp/a");

    // The accumulated response agrees with what was streamed.
    let native = translator.completed.take().unwrap();
    let response = protocol::response(&native, "openai/gpt-test").unwrap();
    let serde_json::Value::Object(_) = serde_json::to_value(&response.content).unwrap()[0].clone()
    else {
        panic!("expected a content block")
    };
    let content = serde_json::to_value(&response.content).unwrap();
    assert_eq!(content[0]["name"], "Read");
    assert!(content[0]["input"].get("offset").is_none(), "{content}");
}

#[test]
fn an_empty_completion_is_an_error_not_a_silent_empty_turn() {
    // Claude Code would render a success with no content as the model saying
    // nothing, and would not retry it.
    for output in [
        json!([]),
        json!([{"type":"reasoning","id":"rs","summary":[],"encrypted_content":"opaque"}]),
    ] {
        let native = upstream_response(output.clone());
        let error = protocol::response(&native, "openai/gpt-test").unwrap_err();
        assert_eq!(error.status, 502, "{output}");
        assert!(error.message.contains("without any text or tool call"));
    }
    // Truncation is a real outcome and keeps its own stop reason.
    let mut truncated = upstream_response(json!([]));
    truncated["status"] = json!("incomplete");
    truncated["incomplete_details"] = json!({"reason":"max_output_tokens"});
    let response = protocol::response(&truncated, "openai/gpt-test").unwrap();
    assert_eq!(response.stop_reason.as_deref(), Some("max_tokens"));
    // Anything usable still passes.
    let usable = upstream_response(
        json!([{"type":"message","id":"m","role":"assistant","content":[{"type":"output_text","text":"hi","annotations":[]}]}]),
    );
    assert!(protocol::response(&usable, "openai/gpt-test").is_ok());
}

#[test]
fn configured_ceiling_lowers_client_effort_but_never_raises_it() {
    let capped = |ceiling: Option<ReasoningEffort>, asked: Option<&str>| {
        let model = Model {
            id: "gpt-test".into(),
            reasoning_effort: None,
            max_reasoning_effort: ceiling,
        };
        let mut req = request();
        if let Some(asked) = asked {
            req["output_config"] = json!({"effort": asked});
        }
        protocol::request(&req, &model).unwrap()["reasoning"]["effort"].clone()
    };

    // Claude Code asks for high on every turn, so a ceiling has to win.
    assert_eq!(capped(Some(ReasoningEffort::Low), Some("high")), "low");
    assert_eq!(capped(Some(ReasoningEffort::Low), Some("max")), "low");
    // It is a ceiling, not a setting: a smaller ask survives untouched.
    assert_eq!(capped(Some(ReasoningEffort::High), Some("low")), "low");
    assert_eq!(capped(Some(ReasoningEffort::Low), Some("none")), "none");
    // With no ask at all the ceiling is stated, so an upstream default cannot
    // silently exceed it.
    assert_eq!(capped(Some(ReasoningEffort::Low), None), "low");
    // Unset ceiling changes nothing.
    assert_eq!(capped(None, Some("high")), "high");
    assert_eq!(capped(None, None), Value::Null);
}

#[tokio::test]
async fn model_aliases_route_hardcoded_client_names() {
    use axum::{Json, Router, routing::post};
    let seen = Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
    let (upstream, up_task) = serve(Router::new().route(
        "/responses",
        post({
            let seen = seen.clone();
            move |Json(req): Json<Value>| {
                seen.lock().unwrap().push(req["model"].as_str().unwrap_or_default().to_owned());
                async move {
                    Json(upstream_response(
                        json!([{"type":"message","id":"m","role":"assistant","content":[{"type":"output_text","text":"ok","annotations":[]}]}]),
                    ))
                }
            }
        }),
    ))
    .await;
    let directory = std::env::temp_dir().join(format!("tinyllm-alias-{}", uuid::Uuid::new_v4()));
    let mut cfg = config(upstream, directory.clone());
    cfg.server.model_aliases = [
        (
            "claude-sonnet-4-6".to_string(),
            "openai/gpt-test".to_string(),
        ),
        ("haiku".to_string(), "openai/gpt-test".to_string()),
    ]
    .into_iter()
    .collect();
    let (gateway, task) = serve(crate::server::router(cfg).await.unwrap()).await;
    let send = |model: &str| {
        let url = format!("{gateway}/anthropic/v1/messages");
        let mut body = request();
        body["model"] = json!(model);
        async move {
            reqwest::Client::new()
                .post(&url)
                .bearer_auth("local-secret")
                .json(&body)
                .send()
                .await
                .unwrap()
                .status()
        }
    };

    // Names that would otherwise be unknown-model errors now route.
    assert_eq!(send("claude-sonnet-4-6").await, 200);
    assert_eq!(send("haiku").await, 200);
    // An explicit provider/model is untouched, and an unaliased name still fails.
    assert_eq!(send("openai/gpt-test").await, 200);
    assert_eq!(send("claude-opus-4-7").await, 400);
    task.abort();
    up_task.abort();
    assert_eq!(
        seen.lock().unwrap().len(),
        3,
        "the unaliased name never dispatched"
    );
    let _ = tokio::fs::remove_dir_all(directory).await;
}

#[tokio::test]
async fn count_tokens_answers_locally_without_an_upstream() {
    // The upstream refuses every connection: a count must never need one.
    let directory = std::env::temp_dir().join(format!("tinyllm-count-{}", uuid::Uuid::new_v4()));
    let cfg = config("http://127.0.0.1:1".into(), directory.clone());
    let (gateway, task) = serve(crate::server::router(cfg).await.unwrap()).await;
    let url = format!("{gateway}/anthropic/v1/messages/count_tokens");
    let count = |body: Value| {
        let url = url.clone();
        async move {
            let response = reqwest::Client::new()
                .post(&url)
                .bearer_auth("local-secret")
                .json(&body)
                .send()
                .await
                .unwrap();
            let status = response.status();
            (status, response.json::<Value>().await.unwrap())
        }
    };

    let (status, small) = count(request()).await;
    assert_eq!(status, 200, "{small}");
    let small = small["input_tokens"].as_u64().unwrap();
    assert!(small > 0);

    let mut big = request();
    big["messages"] = json!([{"role":"user","content":"token ".repeat(2000)}]);
    let (_, big) = count(big).await;
    let big = big["input_tokens"].as_u64().unwrap();
    assert!(big > small * 20, "big={big} small={small}");

    // Malformed input is rejected rather than answered with a wrong number.
    let (status, body) = count(json!({"messages":"not a list"})).await;
    assert_eq!(status, 400);
    assert!(body.get("input_tokens").is_none());
    task.abort();
    let _ = tokio::fs::remove_dir_all(directory).await;
}

#[test]
fn auto_review_detection_needs_all_three_signals() {
    use crate::models::request::RequestBody;
    let monitor = "You are a security monitor for autonomous AI coding agents.\n\n## Context";
    let body = |value: Value| serde_json::from_value::<RequestBody>(value).unwrap();

    for system in [json!(monitor), json!([{"type":"text","text":monitor}])] {
        assert!(
            body(json!({"model":"openai/gpt-test","system":system})).is_auto_review(),
            "system as {system}"
        );
    }
    // Streaming, tools, or an ordinary system prompt each rule it out; a normal
    // Claude Code turn has all three and must never be rerouted.
    for value in [
        json!({"model":"openai/gpt-test","system":monitor,"stream":true}),
        json!({"model":"openai/gpt-test","system":monitor,"tools":[{"name":"Bash"}]}),
        json!({"model":"openai/gpt-test","system":"You are Claude Code."}),
        json!({"model":"openai/gpt-test"}),
        json!({"model":"openai/gpt-test","system":[{"type":"text","text":"You are Claude Code."}]}),
    ] {
        assert!(!body(value.clone()).is_auto_review(), "{value}");
    }
    // An empty tools array is still tool-free.
    assert!(body(json!({"model":"openai/gpt-test","system":monitor,"tools":[]})).is_auto_review());
}

#[test]
fn auto_review_model_must_name_a_configured_provider() {
    use crate::config::Config;
    let directory =
        std::env::temp_dir().join(format!("tinyllm-review-cfg-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir(&directory).unwrap();
    let path = directory.join("config.toml");
    let write = |reviewer: &str| {
        std::fs::write(
            &path,
            format!(
                r#"[server]
auto_review_model = "{reviewer}"
[providers.openai]
type = "openai"
[providers.openai.auth]
type = "ApiKey"
options = "fixture-key"
"#
            ),
        )
        .unwrap();
    };
    write("openai/gpt-5.6-luna");
    assert_eq!(
        Config::load(&path)
            .unwrap()
            .server
            .auto_review_model
            .unwrap(),
        "openai/gpt-5.6-luna"
    );
    // A typo must fail at startup, not on every classifier subrequest.
    for reviewer in ["gpt-5.6-luna", "codex/gpt-5.6-luna", "openai/"] {
        write(reviewer);
        assert!(Config::load(&path).is_err(), "{reviewer}");
    }
    let _ = std::fs::remove_dir_all(directory);
}

#[test]
fn config_from_a_stateful_build_still_starts() {
    use crate::config::Config;
    let directory =
        std::env::temp_dir().join(format!("tinyllm-legacy-cfg-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir(&directory).unwrap();
    let path = directory.join("config.toml");
    std::fs::write(
        &path,
        r#"[server]
bind = "127.0.0.1:8080"
max_response_bytes = 33554432
max_state_bytes = 268435456

[server.state_cleanup]
idle_days = 30
interval_seconds = 3600

[providers.openai]
type = "openai"
[providers.openai.auth]
type = "ApiKey"
options = "fixture-key"
"#,
    )
    .unwrap();
    // Removing the settings must not strand configs written by earlier builds.
    let cfg = Config::load(&path).unwrap();
    assert_eq!(cfg.server.max_response_bytes, 33_554_432);
    assert!(cfg.server.obsolete_max_state_bytes.is_some());
    assert!(cfg.server.obsolete_state_cleanup.is_some());
    let _ = std::fs::remove_dir_all(directory);
}

#[test]
fn config_renders_before_deserialization_and_rejects_invalid_auth() {
    use crate::config::Config;
    let directory = std::env::temp_dir().join(format!("tinyllm-config-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir(&directory).unwrap();
    let missing = format!("TINYLLM_TEST_{}", uuid::Uuid::new_v4().simple());
    assert!(std::env::var_os(&missing).is_none());
    let toml = format!(
        r#"[server]
auth_token = "${{{missing}:-local-secret}}"
state_dir = "${{HOME}}/tinyllm-test"
max_concurrent_requests = ${{{missing}:-2}}
request_body_timeout_seconds = ${{{missing}:-7}}
[providers.${{{missing}:-codex}}]
type = "openai"
[providers.codex.auth]
type = "ApiKey"
options = "${{{missing}:-fixture-key}}"
[providers.codex.models."Claude.Test[main]/Native"]
reasoning_effort = "low"
service_tier = "priority"
"#
    );
    let yaml = format!(
        r#"server:
  auth_token: "${{{missing}:-local-secret}}"
  state_dir: "${{HOME}}/tinyllm-test"
  max_concurrent_requests: ${{{missing}:-2}}
  request_body_timeout_seconds: ${{{missing}:-7}}
providers:
  "${{{missing}:-codex}}":
    type: openai
    auth:
      type: ApiKey
      options: "${{{missing}:-fixture-key}}"
    models:
      "Claude.Test[main]/Native":
        reasoning_effort: low
        service_tier: priority
"#
    );
    for (extension, source) in [("toml", &toml), ("yaml", &yaml), ("yml", &yaml)] {
        let path = directory.join(format!("config.{extension}"));
        std::fs::write(&path, source).unwrap();
        let cfg = Config::load(&path).unwrap();
        let ProviderConfig::OpenAi(provider) = &cfg.providers["codex"] else {
            panic!("wrong provider variant")
        };
        assert_eq!(provider.auth.api_key().unwrap(), "fixture-key");
        assert_eq!(cfg.server.auth_token.as_deref(), Some("local-secret"));
        assert_eq!(cfg.server.max_concurrent_requests, Some(2));
        assert_eq!(cfg.server.request_body_timeout_seconds, 7);
        assert_eq!(
            provider.models["Claude.Test[main]/Native"].service_tier,
            Some(ServiceTier::Priority)
        );
        assert_eq!(cfg.providers.len(), 1);
        assert_eq!(
            cfg.server.state_dir,
            std::path::PathBuf::from(std::env::var("HOME").unwrap()).join("tinyllm-test")
        );
        assert_eq!(
            provider.models["Claude.Test[main]/Native"].reasoning_effort,
            Some(ReasoningEffort::Low)
        );
        std::fs::write(&path, source.replace(&format!("${{{missing}:-7}}"), "0")).unwrap();
        assert!(
            Config::load(&path)
                .err()
                .unwrap()
                .to_string()
                .contains("timeouts must be positive")
        );
        std::fs::write(
            &path,
            source.replace(
                &format!("${{{missing}:-fixture-key}}"),
                &format!("${{{missing}}}"),
            ),
        )
        .unwrap();
        assert!(
            Config::load(&path)
                .err()
                .unwrap()
                .to_string()
                .contains("environment variable")
        );
    }
    let path = directory.join("invalid.toml");
    for source in [
        "[server]\nauth_token = ''\n[providers.openai]\ntype = 'openai'\n[providers.openai.auth]\ntype = 'Subscription'",
        "[server]\nauth_token = 'private-value '\n[providers.openai]\ntype = 'openai'\n[providers.openai.auth]\ntype = 'Subscription'",
        "[server]\nauth_token = \"private-value\\n\"\n[providers.openai]\ntype = 'openai'\n[providers.openai.auth]\ntype = 'Subscription'",
        "[providers.openai]\ntype = 'openai'\n[providers.openai.auth]\ntype = 'ApiKey'\noptions = ''",
        "[providers.openai]\ntype = 'openai'\n[providers.openai.auth]\ntype = 'ApiKey'",
        "[providers.openai]\ntype = 'openai'\n[providers.openai.auth]\ntype = 'ApiKey'\noptions = 'private-value '\n",
        "[providers.openai]\ntype = 'openai'\n[providers.openai.auth]\ntype = 'ApiKey'\noptions = \"private-value\\n\"",
        "[providers.openai]\ntype = 'openai'\n[providers.openai.auth]\ntype = 'ApiKey'\noptions = { key = 'private-value' }",
        "[providers.openai]\ntype = 'openai'\n[providers.openai.auth]\ntype = 'private-value'",
        "[providers.openai]\ntype = 'openai'\n[providers.openai.auth]\ntype = 'ApiKey'\noptions = 'private-value'\nunknown = true",
        "[providers.openai]\ntype = 'openai'\n[providers.openai.auth]\ntype = 'ApiKey'\noptions = \"private-value",
        "[providers.openai]\ntype = 'openai'\nauth = 'api_key'\napi_key_env = 'private-value'",
    ] {
        std::fs::write(&path, source).unwrap();
        let error = format!("{:#}", Config::load(&path).err().unwrap());
        assert!(!error.contains("private-value"));
    }
    std::fs::write(
        &path,
        "[providers.openai]\ntype = 'openai'\n[providers.openai.auth]\ntype = 'Subscription'",
    )
    .unwrap();
    let subscription = Config::load(&path).unwrap();
    assert!(subscription.server.auth_token.is_none());
    assert!(subscription.server.max_concurrent_requests.is_none());
    assert_eq!(subscription.server.request_body_timeout_seconds, 30);
    assert!(subscription.validate().is_ok());
    for limit in [0, tokio::sync::Semaphore::MAX_PERMITS + 1] {
        let mut invalid = subscription.clone();
        invalid.server.max_concurrent_requests = Some(limit);
        assert!(invalid.validate().is_err());
    }
    let ProviderConfig::OpenAi(provider) = &subscription.providers["openai"] else {
        panic!("wrong provider variant")
    };
    assert_eq!(provider.base_url(), "https://chatgpt.com/backend-api/codex");
    let OpenAiAuth::Subscription(options) = &provider.auth else {
        panic!("wrong auth variant")
    };
    assert_eq!(
        options.credentials_dir,
        subscription.server.state_dir.join("auth")
    );
    let example = include_str!("../tinyllm.example.toml");
    std::fs::write(&path, example).unwrap();
    let cfg = Config::load(&path).unwrap();
    assert!(cfg.server.auth_token.is_none());
    assert!(cfg.server.max_concurrent_requests.is_none());
    let ProviderConfig::OpenAi(provider) = &cfg.providers["openai"] else {
        panic!("wrong provider variant")
    };
    assert!(provider.auth.is_subscription());
    assert!(provider.models.contains_key("gpt-5.6-sol"));
    assert!(cfg.server.state_dir.is_absolute());
    let relative = example.replace(
        "# state_dir = \"/custom/state/tinyllm\"",
        "state_dir = \"relative-state\"",
    );
    std::fs::write(&path, relative).unwrap();
    assert_eq!(
        Config::load(&path).unwrap().server.state_dir,
        directory.join("relative-state")
    );
    let path = directory.join("config.yaml");
    std::fs::write(
        &path,
        "server:\n  auth_token: null\nproviders:\n  openai:\n    type: openai\n    auth:\n      type: Subscription\n",
    )
    .unwrap();
    assert!(Config::load(&path).unwrap().server.auth_token.is_none());
    std::fs::write(&path, "providers:\n  codex:\n    type: openai\n    auth:\n      type: Subscription\n    models:\n      gpt-test: {}\n").unwrap();
    let cfg = Config::load(&path).unwrap();
    let ProviderConfig::OpenAi(provider) = &cfg.providers["codex"] else {
        panic!("wrong provider variant")
    };
    assert!(provider.models.contains_key("gpt-test"));
    let OpenAiAuth::Subscription(options) = &provider.auth else {
        panic!("wrong auth variant")
    };
    assert_eq!(
        options.credentials_dir,
        cfg.server.state_dir.join("auth/codex")
    );
    for prefix in ["", ".", "..", "openai/", "two words"] {
        let mut invalid = cfg.clone();
        let provider = invalid.providers.remove("codex").unwrap();
        invalid.providers.insert(prefix.into(), provider);
        assert!(invalid.validate().is_err());
    }
    std::fs::write(&path, "providers:\n  codex:\n    type: openai\n    auth:\n      type: Subscription\n    models:\n      alias:\n        id: gpt-test\n").unwrap();
    assert!(Config::load(&path).is_err());
    let _ = std::fs::remove_dir_all(directory);
}

#[test]
fn claude_controls_effort_and_format_over_server_defaults() {
    let mut req = request();
    req["thinking"] = json!({"type":"adaptive"});
    req["output_config"] = json!({"effort":"high","format":{"type":"json_schema","schema":{"type":"object","properties":{},"additionalProperties":false}}});
    let out = protocol::request(&req, &model()).unwrap();
    assert_eq!(out["reasoning"]["effort"], "high");
    assert_eq!(
        out["text"]["format"]["schema"],
        req["output_config"]["format"]["schema"]
    );
    req["output_config"]["effort"] = json!("max");
    assert_eq!(
        protocol::request(&req, &model()).unwrap()["reasoning"]["effort"],
        "max"
    );
    req["thinking"] = json!({"type":"disabled"});
    assert_eq!(
        protocol::request(&req, &model()).unwrap()["reasoning"]["effort"],
        "none"
    );
}

#[tokio::test]
async fn openai_model_defaults_and_client_overrides_reach_each_endpoint() {
    use axum::{Json, Router, routing::post};
    let (sender, mut receiver) = tokio::sync::mpsc::channel(1);
    let (upstream, up_task) = serve(Router::new().route("/responses", post(move |Json(body): Json<Value>| {
        let sender = sender.clone();
        async move {
            sender.send(body).await.unwrap();
            let mut response = upstream_response(json!([{"type":"message","id":"m","role":"assistant","content":[{"type":"output_text","text":"ok","annotations":[]}]}]));
            response["service_tier"] = json!("default");
            Json(response)
        }
    }))).await;
    let directory =
        std::env::temp_dir().join(format!("tinyllm-effort-http-{}", uuid::Uuid::new_v4()));
    let mut cfg = config(upstream, directory.clone());
    let ProviderConfig::OpenAi(provider) = cfg.providers.get_mut("openai").unwrap() else {
        panic!()
    };
    provider.models.insert(
        "gpt-5.4".into(),
        ModelOptions {
            reasoning_effort: Some(ReasoningEffort::Max),
            service_tier: Some(ServiceTier::Priority),
            ..Default::default()
        },
    );
    let (gateway, task) = serve(crate::server::router(cfg).await.unwrap()).await;
    let client = reqwest::Client::new();
    for (path, container) in [
        ("/anthropic/v1/messages", "output_config"),
        ("/v1/chat/completions", ""),
        ("/v1/responses", "reasoning"),
    ] {
        for (model, native_model, effort, expected) in [
            ("gpt-5.6-sol", "gpt-5.6-sol", Some("max"), "max"),
            ("gpt-5.4", "gpt-5.4", Some("max"), "xhigh"),
            ("gpt-5.4", "gpt-5.4", Some("low"), "low"),
            ("gpt-5.4", "gpt-5.4", None, "xhigh"),
            ("gpt-5.4-fast", "gpt-5.4", None, "xhigh"),
            ("gpt-9-fast", "gpt-9-fast", Some("low"), "low"),
        ] {
            let mut body = if path.ends_with("/responses") {
                json!({"input":"hello"})
            } else {
                json!({"messages":[{"role":"user","content":"hello"}]})
            };
            body["model"] = json!(format!("openai/{model}"));
            if path.starts_with("/anthropic") {
                body["max_tokens"] = json!(32);
                body["thinking"] = json!({"type":"adaptive"});
            }
            if model.ends_with("-fast") {
                body["service_tier"] = if path.starts_with("/anthropic") {
                    json!("standard_only")
                } else {
                    json!("flex")
                };
            }
            if let Some(effort) = effort {
                if container.is_empty() {
                    body["reasoning_effort"] = json!(effort);
                } else {
                    body[container] = json!({"effort":effort});
                }
            }
            let response = client
                .post(format!("{gateway}{path}"))
                .bearer_auth("local-secret")
                .json(&body)
                .send()
                .await
                .unwrap();
            let status = response.status();
            let response = response.json::<Value>().await.unwrap();
            assert_eq!(status, 200, "{path} {model}: {response}");
            let upstream = receiver.recv().await.unwrap();
            assert_eq!(upstream["model"], native_model);
            assert_eq!(upstream["reasoning"]["effort"], expected);
            assert_eq!(response["model"], format!("openai/{model}"));
            if !path.starts_with("/anthropic") {
                assert_eq!(response["service_tier"], "default");
            }
            match native_model {
                "gpt-5.4" => assert_eq!(upstream["service_tier"], "priority"),
                // An unconfigured base forwards the name literally and keeps the client tier.
                "gpt-9-fast" => assert_eq!(
                    upstream["service_tier"],
                    if path.starts_with("/anthropic") {
                        "default"
                    } else {
                        "flex"
                    }
                ),
                _ => assert!(upstream.get("service_tier").is_none()),
            }
        }
        let tiers = if path.starts_with("/anthropic") {
            vec![
                (json!("auto"), json!("auto")),
                (json!("standard_only"), json!("default")),
            ]
        } else {
            vec![
                (json!("auto"), json!("auto")),
                (json!("default"), json!("default")),
                (json!("fast"), json!("priority")),
                (json!("flex"), json!("flex")),
                (Value::Null, Value::Null),
            ]
        };
        for (tier, expected) in tiers {
            let mut body = if path.ends_with("/responses") {
                json!({"input":"hello"})
            } else {
                json!({"messages":[{"role":"user","content":"hello"}]})
            };
            body["model"] = json!("openai/gpt-5.4");
            body["service_tier"] = tier;
            if path.starts_with("/anthropic") {
                body["max_tokens"] = json!(32);
            }
            let response = client
                .post(format!("{gateway}{path}"))
                .bearer_auth("local-secret")
                .json(&body)
                .send()
                .await
                .unwrap();
            assert_eq!(
                response.status(),
                200,
                "{path}: {}",
                response.text().await.unwrap()
            );
            let upstream = receiver.recv().await.unwrap();
            assert_eq!(upstream.get("service_tier"), Some(&expected));
            assert_eq!(upstream["reasoning"]["effort"], "xhigh");
        }
    }
    let response = client.post(format!("{gateway}/anthropic/v1/messages")).bearer_auth("local-secret")
        .json(&json!({"model":"openai/gpt-6-astra","messages":[{"role":"user","content":"hello"}],"max_tokens":32,"thinking":{"type":"disabled"}})).send().await.unwrap();
    assert_eq!(response.status(), 400);
    assert!(
        response.json::<Value>().await.unwrap()["error"]["message"]
            .as_str()
            .unwrap()
            .contains("requires reasoning")
    );
    assert!(receiver.try_recv().is_err());
    task.abort();
    up_task.abort();
    let _ = task.await;
    let _ = std::fs::remove_dir_all(directory);
}

#[tokio::test]
async fn local_command_acknowledgement_has_no_model_reasoning() {
    let mut req = request();
    req["messages"] = json!([
        {"role":"user","content":[{"type":"text","text":"<local-command-stdout>Compacted</local-command-stdout>"}]},
        {"role":"system","content":"Retain these tool reminders"},
        {"role":"assistant","content":[{"type":"text","text":"No response requested."}]},
        {"role":"user","content":"Continue"}
    ]);
    let out = protocol::request(&req, &model()).unwrap();
    assert_eq!(
        out["input"][3]["content"][0],
        json!({"type":"output_text","text":"No response requested."})
    );
    req["messages"].as_array_mut().unwrap().push(json!({
        "role":"assistant", "content":"No response requested."
    }));
    req["messages"][0]["content"] = json!("an ordinary user message");
    assert!(protocol::request(&req, &model()).is_ok());
}

#[tokio::test]
async fn live_sse_is_incremental_keeps_alive_and_never_finishes_failed_streams() {
    use axum::{Json, Router, body::Body, routing::post};
    use futures::StreamExt;
    let (upstream,up_task)=serve(Router::new().route("/responses",post(|Json(req):Json<Value>| async move {
        let output=async_stream::stream! {
            let first = stream_fixture()[..5].iter().map(|event|format!("event: {}\ndata: {event}\n\n",event["type"].as_str().unwrap())).collect::<String>();
            yield Ok::<_,std::io::Error>(bytes::Bytes::from(first));
            yield Ok(bytes::Bytes::from_static(b"event: keepalive\ndata: {\"type\":\"keepalive\"}\n\n"));
            tokio::time::sleep(std::time::Duration::from_millis(1150)).await;
            if req["max_output_tokens"] == 13 {
                yield Ok(bytes::Bytes::from_static(b"event: error\ndata: {\"type\":\"error\",\"code\":\"server_error\",\"message\":\"backend failed upstream-secret\"}\n\n"));
            }
        };
        ([("content-type","text/event-stream")],Body::from_stream(output))
    }))).await;
    let directory = std::env::temp_dir().join(format!("tinyllm-sse-{}", uuid::Uuid::new_v4()));
    let mut cfg = config(upstream, directory.clone());
    cfg.server.max_concurrent_requests = Some(1);
    let (gateway, gw_task) = serve(crate::server::router(cfg).await.unwrap()).await;
    let client = reqwest::Client::new();
    let url = format!("{gateway}/anthropic/v1/messages");
    for (max, stops) in [(13, false), (14, false), (13, true), (14, true)] {
        let mut req = request();
        req["stream"] = json!(true);
        req["max_tokens"] = json!(max);
        if stops {
            req["stop_sequences"] = json!(["éll"]);
        }
        let response = client
            .post(&url)
            .bearer_auth("local-secret")
            .json(&req)
            .send()
            .await
            .unwrap();
        let mut body = response.bytes_stream();
        let first = tokio::time::timeout(std::time::Duration::from_millis(500), body.next())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert!(String::from_utf8_lossy(&first).contains("message_start"));
        let overloaded = client
            .post(&url)
            .bearer_auth("local-secret")
            .header("content-type", "application/json")
            .body("invalid JSON")
            .send()
            .await
            .unwrap();
        assert_eq!(overloaded.status(), 503);
        let mut bytes = first.to_vec();
        while let Some(chunk) = body.next().await {
            bytes.extend_from_slice(&chunk.unwrap());
        }
        let text = String::from_utf8(bytes).unwrap();
        assert!(text.contains("event: ping"));
        assert!(text.contains("event: error"));
        assert!(!text.contains("message_stop"));
        assert!(!text.contains("upstream-secret"));
        if stops {
            let deltas: String = text
                .lines()
                .filter_map(|line| line.strip_prefix("data: "))
                .map(|line| serde_json::from_str::<Value>(line).unwrap())
                .filter_map(|event| event["delta"]["text"].as_str().map(str::to_owned))
                .collect();
            assert_eq!(deltas, "h");
        }
        assert!(text.contains(if max == 13 {
            "backend failed [redacted]"
        } else {
            "ended before completion"
        }));
    }
    // The gateway writes no conversation state, so nothing may be left behind.
    assert!(
        std::fs::read_dir(&directory)
            .is_err_and(|error| error.kind() == std::io::ErrorKind::NotFound)
    );
    gw_task.abort();
    up_task.abort();
    let _ = tokio::fs::remove_dir_all(directory).await;
}
