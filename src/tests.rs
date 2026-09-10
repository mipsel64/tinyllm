use crate::{
    config::ProviderConfig,
    models::ReasoningEffort,
    providers::openai::{
        models::{Model, ModelOptions, OpenAiAuth, ServiceTier, SubscriptionOptions},
        protocol, state, stream,
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
    }
}

fn request() -> Value {
    json!({"model":"openai/gpt-test","max_tokens":1024,"system":[{"type":"text","text":"Keep ALL instructions."}],
        "messages":[{"role":"user","content":"hello"}],
        "tools":[{"name":"lookup","description":"Find a record","input_schema":{"type":"object","properties":{"url":{"type":"string","format":"uri"}},"required":[]}}]})
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
    let out = protocol::request(&req, &model(), &Default::default()).unwrap();
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
            protocol::request(&req, &model(), &Default::default()).unwrap()["tool_choice"],
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
        assert!(protocol::request(&req, &model(), &Default::default()).is_err());
    }
    let mut req = request();
    req["messages"][0]["content"] =
        json!([{"type":"tool_result","tool_use_id":"missing","content":"never drop me"}]);
    assert!(protocol::request(&req, &model(), &Default::default()).is_err());
}

#[test]
fn context_management_preserves_history_and_rejects_unimplemented_edits() {
    let mut req = request();
    let expected = protocol::request(&req, &model(), &Default::default()).unwrap();
    for context in [
        json!({}),
        json!({"edits":[]}),
        json!({"edits":[{"type":"clear_thinking_20251015","keep":"all"}]}),
    ] {
        req["context_management"] = context;
        assert_eq!(
            protocol::request(&req, &model(), &Default::default()).unwrap(),
            expected
        );
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
        assert!(protocol::request(&req, &model(), &Default::default()).is_err());
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
    let initial = protocol::request(&req, &model(), &Default::default()).unwrap();
    assert_eq!(initial["tools"].as_array().unwrap().len(), 1);
    req["messages"].as_array_mut().unwrap().extend([
        json!({"role":"assistant","content":[{"type":"tool_use","id":"search","name":"ToolSearch","input":{"query":"lookup"}}]}),
        json!({"role":"user","content":[{"type":"tool_result","tool_use_id":"search","content":[{"type":"text","text":"Found a tool"},{"type":"tool_reference","tool_name":"lookup"},{"type":"image","source":{"type":"url","url":"https://example.org/result.png"}}]}]})
    ]);
    let out = protocol::request(&req, &model(), &Default::default()).unwrap();
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
    assert!(protocol::request(&req, &model(), &Default::default()).is_err());
    req["messages"].as_array_mut().unwrap().truncate(1);
    assert_eq!(
        protocol::request(&req, &model(), &Default::default()).unwrap(),
        initial
    );
    req["tools"][1]["defer_loading"] = json!("true");
    assert!(protocol::request(&req, &model(), &Default::default()).is_err());
}

fn upstream_response(output: Value) -> Value {
    json!({"id":"resp_test","status":"completed","output":output,"usage":{"input_tokens":120,"output_tokens":25,"input_tokens_details":{"cached_tokens":20},"output_tokens_details":{"reasoning_tokens":15}}})
}

#[tokio::test]
async fn tool_default_replay_uses_the_original_schema_after_restart() {
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
    let store = state::Store::open(directory.clone(), 100_000, 50_000)
        .await
        .unwrap();
    let original = response["content"].clone();
    for input in [original[1]["input"].clone(), {
        let mut input = original[1]["input"].clone();
        input["replace_all"] = json!(false);
        input
    }] {
        req["messages"] = json!([{"role":"assistant","content":original}]);
        req["messages"][0]["content"][1]["input"] = input;
        req["tools"] = json!([]);
        let restored = store
            .restore_scoped(&req, "openai", "gpt-test")
            .await
            .unwrap();
        assert_eq!(json!(restored[&0]), native["output"]);
    }
    let valid = req.clone();
    for (pointer, value) in [
        ("/messages/0/content/1/input/replace_all", json!(true)),
        (
            "/messages/0/content/1/input/file_path",
            json!("/tmp/changed"),
        ),
        ("/messages/0/content/1/id", json!("changed")),
        ("/messages/0/content/1/name", json!("Other")),
        ("/messages/0/content/1", json!(42)),
        ("/messages/0/content/1", json!([])),
        ("/messages/0/content/1", json!("invalid")),
    ] {
        let mut changed = valid.clone();
        *changed.pointer_mut(pointer).unwrap() = value;
        changed["tools"] =
            json!([{"name":"Edit","input_schema":{"properties":{"replace_all":{"default":true}}}}]);
        assert!(
            store
                .restore_scoped(&changed, "openai", "gpt-test")
                .await
                .is_err(),
            "{pointer}"
        );
    }
    let mut changed = valid;
    changed["messages"][0]["content"][1]["input"]["unknown"] = json!(false);
    assert!(
        store
            .restore_scoped(&changed, "openai", "gpt-test")
            .await
            .is_err()
    );
    drop(store);
    tokio::fs::remove_dir_all(directory).await.unwrap();
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
    assert_eq!(response["content"][1]["text"], "<block>false");
    assert_eq!(response["usage"]["output_tokens"], 25);
    task.abort();
    let _ = task.await;
    up_task.abort();
    let store = state::Store::open(directory.clone(), 100_000, 50_000)
        .await
        .unwrap();
    let restored = store
        .restore_scoped(
            &json!({"messages":[{"role":"assistant","content":response["content"]}]}),
            "openai",
            "gpt-test",
        )
        .await
        .unwrap();
    assert_eq!(restored[&0].len(), 2);
    assert_eq!(restored[&0][0]["encrypted_content"], "opaque-before");
    assert_eq!(restored[&0][1]["content"][0]["text"], "<block>false");
    assert!(restored[&0][1].get("id").is_none());
    drop(store);
    tokio::fs::remove_dir_all(directory).await.unwrap();
}

#[tokio::test]
async fn continuation_survives_restart_forks_and_explicit_compaction_boundary() {
    use state::Store;
    let directory =
        std::env::temp_dir().join(format!("tinyllm-state-test-{}", uuid::Uuid::new_v4()));
    let reference = Store::reference();
    let native = upstream_response(json!([
        {"type":"reasoning","id":"rs_1","summary":[],"encrypted_content":"opaque-openai-data"},
        {"type":"message","id":"msg_1","role":"assistant","phase":"commentary","status":"completed","content":[{"type":"output_text","text":"Checking","annotations":[]}]},
        {"type":"function_call","id":"fc_1","call_id":"call_1","name":"lookup","arguments":"{}","status":"completed"}
    ]));
    let converted = protocol::response(&native, "openai/gpt-test", &reference).unwrap();
    assert_eq!(converted.usage.input_tokens, 100);
    assert_eq!(converted.usage.cache_read_input_tokens, 20);
    assert_eq!(converted.usage.output_tokens, 25);
    let content = serde_json::to_value(converted.content).unwrap();
    let store = Store::open(directory.clone(), 100_000, 50_000)
        .await
        .unwrap();
    store
        .save(&reference, "gpt-test", &native, content.clone())
        .await
        .unwrap();
    drop(store);
    let store = Store::open(directory.clone(), 100_000, 50_000)
        .await
        .unwrap();
    let mut req = request();
    req["context_management"] = json!({"edits":[{"type":"clear_thinking_20251015","keep":"all"}]});
    req["thinking"] = json!({"type":"adaptive","display":"omitted"});
    req["messages"].as_array_mut().unwrap().extend([
        json!({"role":"assistant","content":content}),
        json!({"role":"user","content":[{"type":"tool_result","tool_use_id":"call_1","content":"ok"}]})
    ]);
    let (a, b) = tokio::join!(
        store.restore(&req, "gpt-test"),
        store.restore(&req, "gpt-test")
    );
    assert_eq!(a.as_ref().unwrap(), b.as_ref().unwrap());
    let out = protocol::request(&req, &model(), &a.unwrap()).unwrap();
    assert_eq!(out["input"][2]["encrypted_content"], "opaque-openai-data");
    assert_eq!(out["input"][3]["phase"], "commentary");
    req["messages"][1]["content"][1]["text"] = json!("modified");
    assert!(store.restore(&req, "gpt-test").await.is_err());
    req["messages"] = json!([{"role":"user","content":"Compacted summary; new reasoning context"}]);
    assert!(store.restore(&req, "gpt-test").await.unwrap().is_empty());
    req["messages"] = json!([{"role":"assistant","content":[{"type":"redacted_thinking","data":Store::reference()}]}]);
    assert!(store.restore(&req, "gpt-test").await.is_err());
    tokio::fs::remove_dir_all(directory).await.unwrap();
}

#[test]
fn invalid_tool_json_and_upstream_failures_never_become_success() {
    let mut r = upstream_response(
        json!([{"type":"function_call","call_id":"a","name":"lookup","arguments":"{broken"}]),
    );
    assert!(protocol::response(&r, "openai/gpt-test", "reference").is_err());
    r["output"] = json!([]);
    r["status"] = json!("failed");
    assert!(protocol::response(&r, "openai/gpt-test", "reference").is_err());
    r["status"] = json!("incomplete");
    r["incomplete_details"] = json!({"reason":"max_output_tokens"});
    assert_eq!(
        protocol::response(&r, "openai/gpt-test", "reference")
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
    let raw: String = events
        .iter()
        .map(|e| {
            format!(
                "event: {}\r\ndata: {}\r\n\r\n",
                e["type"].as_str().unwrap(),
                e
            )
        })
        .collect();
    let upstream = futures::stream::iter(
        raw.bytes()
            .map(|b| Ok::<_, std::io::Error>(bytes::Bytes::from(vec![b])))
            .collect::<Vec<_>>(),
    );
    let mut decoder = Box::pin(stream::decode(upstream, 1_000_000));
    let mut translator =
        stream::Translator::new("openai/gpt-test".into(), "tinyllm:v1:test".into());
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
    assert_eq!(starts.len(), 4);
    assert_eq!(starts[2]["content_block"]["id"], "call_a");
    assert_eq!(starts[3]["content_block"]["id"], "call_b");
    let deltas = |index: u64, key: &str| {
        output
            .iter()
            .filter(|e| e["type"] == "content_block_delta" && e["index"] == index)
            .map(|e| e["delta"][key].as_str().unwrap())
            .collect::<String>()
    };
    assert_eq!(deltas(1, "text"), "héllo");
    assert_eq!(deltas(2, "partial_json"), "{\"a\":1}");
    assert_eq!(deltas(3, "partial_json"), "{\"b\":2}");
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
        let mut translator = stream::Translator::new("openai/gpt-test".into(), "ref".into());
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
    let mut translator = stream::Translator::new("openai/gpt-test".into(), "ref".into());
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
        tokio::fs::remove_dir_all(self.directory).await.unwrap();
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
async fn http_auth_rejects_missing_and_invalid_client_credentials() {
    let fixture = http_fixture("openai", Some("local-secret")).await;
    for header in [None, Some("authorization"), Some("x-api-key")] {
        let mut call = fixture.client.post(&fixture.url).json(&request());
        if let Some(header) = header {
            call = call.header(header, "Bearer wrong-token");
        }
        assert_eq!(call.send().await.unwrap().status(), 401);
    }
    fixture.close().await;
}

#[tokio::test]
async fn http_protocol_preserves_json_streams_and_unique_continuations() {
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
    assert_eq!(a["content"][1]["text"], "hello");
    assert_ne!(a["content"][0]["data"], b["content"][0]["data"]);
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
    assert_eq!(
        fixture
            .client
            .get(format!("{}/anthropic/v1/models", fixture.gateway))
            .bearer_auth("local-secret")
            .send()
            .await
            .unwrap()
            .json::<Value>()
            .await
            .unwrap()["data"][0]["id"],
        "openai/gpt-test"
    );
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
            response.json::<Value>().await.unwrap()["content"][1]["text"],
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
                    if event["type"] == "content_block_stop" && event["index"] == 1 {
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
        tokio::fs::remove_dir_all(dir).await.unwrap();
        assert!(
            stopped.is_ok(),
            "upstream continued after disconnect; streaming={streaming}"
        );
    }
}

#[tokio::test]
async fn stream_failure_truncation_and_bounds_are_explicit() {
    use futures::StreamExt;
    let mut translator = stream::Translator::new("openai/gpt-test".into(), "ref".into());
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
    tokio::fs::remove_dir_all(dir).await.unwrap();
}

#[tokio::test]
async fn dropped_duplicate_and_oversized_continuations_fail_closed() {
    use state::Store;
    let directory = std::env::temp_dir().join(format!("tinyllm-replay-{}", uuid::Uuid::new_v4()));
    let store = Store::open(directory.clone(), 100_000, 8000).await.unwrap();
    let mut blocks = Vec::new();
    for _ in 0..2 {
        let reference = Store::reference();
        let native = upstream_response(
            json!([{"type":"reasoning","id":"r","summary":[],"encrypted_content":"x".repeat(4400)}, {"type":"message","id":"m","role":"assistant","content":[{"type":"output_text","text":"done","annotations":[]}]}]),
        );
        let response = protocol::response(&native, "openai/gpt-test", &reference).unwrap();
        let content = serde_json::to_value(response.content).unwrap();
        store
            .save(&reference, "gpt-test", &native, content.clone())
            .await
            .unwrap();
        blocks.push(content);
    }
    let mut req = request();
    req["messages"] = json!([{"role":"assistant","content":[{"type":"text","text":"done"}]}]);
    assert!(
        store
            .restore(&req, "gpt-test")
            .await
            .unwrap_err()
            .message
            .contains("missing")
    );
    req["messages"] = json!([{"role":"assistant","content":blocks[0]},{"role":"user","content":"next"},{"role":"assistant","content":blocks[0]}]);
    assert!(
        store
            .restore(&req, "gpt-test")
            .await
            .unwrap_err()
            .message
            .contains("duplicate")
    );
    req["messages"][2]["content"] = blocks[1].clone();
    assert!(
        store
            .restore(&req, "gpt-test")
            .await
            .unwrap_err()
            .message
            .contains("exceeds")
    );
    tokio::fs::remove_dir_all(directory).await.unwrap();
}

#[test]
fn state_directory_defaults_follow_xdg() {
    if let Some(expected) = std::env::var_os("TINYLLM_TEST_EXPECTED_STATE_DIR") {
        assert_eq!(
            crate::config::Server::default().state_dir,
            std::path::PathBuf::from(expected)
        );
        return;
    }
    let fallback =
        std::path::PathBuf::from(std::env::var_os("HOME").unwrap()).join(".local/state/tinyllm");
    let custom = std::env::temp_dir().join("tinyllm-xdg-test");
    for (xdg, expected) in [
        (None, fallback.clone()),
        (Some(std::path::PathBuf::from("relative")), fallback),
        (Some(custom.clone()), custom.join("tinyllm")),
    ] {
        let mut child = std::process::Command::new(std::env::current_exe().unwrap());
        child
            .args(["--exact", "tests::state_directory_defaults_follow_xdg"])
            .env("TINYLLM_TEST_EXPECTED_STATE_DIR", expected)
            .env_remove("XDG_STATE_HOME");
        if let Some(xdg) = xdg {
            child.env("XDG_STATE_HOME", xdg);
        }
        let output = child.output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
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
        .map(|e| format!("event: {}\ndata: {e}\n\n", e["type"].as_str().unwrap()))
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
    let mut followup = request();
    followup["messages"].as_array_mut().unwrap().extend([
        json!({"role":"assistant","content":response["content"]}),
        json!({"role":"user","content":[{"type":"tool_result","tool_use_id":"call_a","content":"ok"},{"type":"tool_result","tool_use_id":"call_b","is_error":true,"content":"denied"}]}),
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
    std::fs::remove_dir_all(directory).unwrap();
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
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn claude_controls_effort_and_format_over_server_defaults() {
    let mut req = request();
    req["thinking"] = json!({"type":"adaptive"});
    req["output_config"] = json!({"effort":"high","format":{"type":"json_schema","schema":{"type":"object","properties":{},"additionalProperties":false}}});
    let out = protocol::request(&req, &model(), &Default::default()).unwrap();
    assert_eq!(out["reasoning"]["effort"], "high");
    assert_eq!(
        out["text"]["format"]["schema"],
        req["output_config"]["format"]["schema"]
    );
    req["output_config"]["effort"] = json!("max");
    assert_eq!(
        protocol::request(&req, &model(), &Default::default()).unwrap()["reasoning"]["effort"],
        "max"
    );
    req["thinking"] = json!({"type":"disabled"});
    assert_eq!(
        protocol::request(&req, &model(), &Default::default()).unwrap()["reasoning"]["effort"],
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
        },
    );
    let (gateway, task) = serve(crate::server::router(cfg).await.unwrap()).await;
    let client = reqwest::Client::new();
    for (path, container) in [
        ("/anthropic/v1/messages", "output_config"),
        ("/v1/chat/completions", ""),
        ("/v1/responses", "reasoning"),
    ] {
        for (model, effort, expected) in [
            ("gpt-5.6-sol", Some("max"), "max"),
            ("gpt-5.4", Some("max"), "xhigh"),
            ("gpt-5.4", Some("low"), "low"),
            ("gpt-5.4", None, "xhigh"),
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
            assert_eq!(
                response.status(),
                200,
                "{path} {model}: {}",
                response.text().await.unwrap()
            );
            let upstream = receiver.recv().await.unwrap();
            assert_eq!(upstream["model"], model);
            assert_eq!(upstream["reasoning"]["effort"], expected);
            if !path.starts_with("/anthropic") {
                assert_eq!(
                    response.json::<Value>().await.unwrap()["service_tier"],
                    "default"
                );
            }
            if model == "gpt-5.4" {
                assert_eq!(upstream["service_tier"], "priority");
            } else {
                assert!(upstream.get("service_tier").is_none());
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
    std::fs::remove_dir_all(directory).unwrap();
}

#[tokio::test]
async fn local_command_acknowledgement_has_no_model_reasoning() {
    use state::Store;
    let directory = std::env::temp_dir().join(format!("tinyllm-ack-{}", uuid::Uuid::new_v4()));
    let store = Store::open(directory.clone(), 10_000, 8000).await.unwrap();
    assert!(Store::open(directory.clone(), 10_000, 8000).await.is_err());
    let mut req = request();
    req["messages"] = json!([
        {"role":"user","content":[{"type":"text","text":"<local-command-stdout>Compacted</local-command-stdout>"}]},
        {"role":"system","content":"Retain these tool reminders"},
        {"role":"assistant","content":[{"type":"text","text":"No response requested."}]},
        {"role":"user","content":"Continue"}
    ]);
    let restored = store.restore(&req, "gpt-test").await.unwrap();
    let out = protocol::request(&req, &model(), &restored).unwrap();
    assert_eq!(
        out["input"][3]["content"][0],
        json!({"type":"output_text","text":"No response requested."})
    );
    req["messages"].as_array_mut().unwrap().push(json!({
        "role":"assistant", "content":"No response requested."
    }));
    assert!(store.restore(&req, "gpt-test").await.is_ok());
    req["messages"][0]["content"] = json!("an ordinary user message");
    assert!(store.restore(&req, "gpt-test").await.is_err());
    tokio::fs::remove_dir_all(directory).await.unwrap();
}

#[tokio::test]
async fn live_sse_is_incremental_keeps_alive_and_never_finishes_failed_streams() {
    use axum::{Json, Router, body::Body, routing::post};
    use futures::StreamExt;
    let (upstream,up_task)=serve(Router::new().route("/responses",post(|Json(req):Json<Value>| async move {
        let output=async_stream::stream! {
            let first = stream_fixture()[..5].iter().map(|event|format!("event: {}\ndata: {event}\n\n",event["type"].as_str().unwrap())).collect::<String>();
            yield Ok::<_,std::io::Error>(bytes::Bytes::from(first));
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
    assert!(!std::fs::read_dir(&directory).unwrap().any(|entry| {
        entry
            .unwrap()
            .path()
            .extension()
            .is_some_and(|extension| extension == "json")
    }));
    gw_task.abort();
    up_task.abort();
    tokio::fs::remove_dir_all(directory).await.unwrap();
}
