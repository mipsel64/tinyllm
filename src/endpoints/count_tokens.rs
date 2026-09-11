//! Local estimate for `POST /v1/messages/count_tokens`.
//!
//! Claude Code sizes its context and decides when to compact from this. It is an
//! estimate, not a billing count: text uses OpenAI's `o200k_base`, while images
//! and protocol framing are approximated.

use crate::models::request::RequestBody;
use serde_json::Value;
use std::collections::HashSet;
use tiktoken_rs::o200k_base_singleton;

/// Per-message and per-content-part framing the wire format adds around text.
const MESSAGE_OVERHEAD: u64 = 4;
/// Flat allowance for an image; real cost depends on dimensions we do not decode.
const IMAGE_ESTIMATE: u64 = 1_600;

pub fn count(body: &RequestBody) -> u64 {
    let mut total = 0;
    if let Some(system) = body.fields.get("system") {
        total += content_tokens(system);
    }
    for message in body
        .fields
        .get("messages")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        total += MESSAGE_OVERHEAD + content_tokens(&message["content"]);
    }
    let loaded = loaded_tools(body);
    for tool in body
        .fields
        .get("tools")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        // A deferred tool costs nothing until a tool_reference loads it, so
        // counting it would report context the model never sees.
        let deferred = tool["defer_loading"] == true;
        if deferred
            && !tool["name"]
                .as_str()
                .is_some_and(|name| loaded.contains(name))
        {
            continue;
        }
        // Tool definitions reach the model as serialized schema.
        total += MESSAGE_OVERHEAD + text_tokens(&tool.to_string());
    }
    total.max(1)
}

/// Names that a `tool_reference` in some tool result has pulled into context.
/// Mirrors the same rule in the request translator.
fn loaded_tools(body: &RequestBody) -> HashSet<&str> {
    body.fields
        .get("messages")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .flat_map(|message| message["content"].as_array().into_iter().flatten())
        .flat_map(|block| block["content"].as_array().into_iter().flatten())
        .filter(|part| part["type"] == "tool_reference")
        .filter_map(|part| part["tool_name"].as_str())
        .collect()
}

fn content_tokens(content: &Value) -> u64 {
    match content {
        Value::String(text) => text_tokens(text),
        Value::Array(blocks) => blocks.iter().map(block_tokens).sum(),
        _ => 0,
    }
}

fn block_tokens(block: &Value) -> u64 {
    match block["type"].as_str() {
        Some("text") => text_tokens(block["text"].as_str().unwrap_or_default()),
        Some("thinking") => text_tokens(block["thinking"].as_str().unwrap_or_default()),
        // Carriers replay as opaque ciphertext; base64 is about four bytes a token.
        Some("redacted_thinking") => block["data"].as_str().map_or(0, |d| d.len() as u64 / 4),
        Some("image") => IMAGE_ESTIMATE,
        Some("tool_use") => MESSAGE_OVERHEAD + text_tokens(&block["input"].to_string()),
        Some("tool_result") => MESSAGE_OVERHEAD + content_tokens(&block["content"]),
        _ => text_tokens(&block.to_string()),
    }
}

fn text_tokens(text: &str) -> u64 {
    if text.is_empty() {
        return 0;
    }
    o200k_base_singleton()
        .encode_with_special_tokens(text)
        .len() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn body(value: Value) -> RequestBody {
        serde_json::from_value(value).unwrap()
    }

    #[test]
    fn counts_scale_with_content_and_never_reach_zero() {
        let empty = count(&body(json!({"model":"m","messages":[]})));
        assert_eq!(empty, 1, "a count of zero would read as an empty context");

        let short = count(&body(
            json!({"model":"m","messages":[{"role":"user","content":"hello"}]}),
        ));
        let long = count(&body(json!({"model":"m","messages":[
            {"role":"user","content":"hello ".repeat(500)}
        ]})));
        assert!(short > empty);
        assert!(long > short * 50, "long={long} short={short}");
    }

    #[test]
    fn tokenizes_text_rather_than_guessing_from_length() {
        // " hello" is one o200k token; a naive bytes/4 estimate would say two.
        let one = count(&body(
            json!({"model":"m","messages":[{"role":"user","content":" hello"}]}),
        ));
        assert_eq!(one, MESSAGE_OVERHEAD + 1);
    }

    #[test]
    fn every_content_shape_contributes() {
        let base = count(&body(json!({"model":"m","messages":[
            {"role":"user","content":[{"type":"text","text":"x"}]}
        ]})));
        for block in [
            json!({"type":"image","source":{"type":"base64","media_type":"image/png","data":"AA"}}),
            json!({"type":"tool_use","id":"t","name":"lookup","input":{"a":1}}),
            json!({"type":"tool_result","tool_use_id":"t","content":"result"}),
            json!({"type":"redacted_thinking","data":"tinyllm:v1:aaaa:gAAAA"}),
            json!({"type":"thinking","thinking":"considering","signature":"s"}),
        ] {
            let with = count(&body(json!({"model":"m","messages":[
                {"role":"user","content":[{"type":"text","text":"x"}, block]}
            ]})));
            assert!(with > base, "{block} added nothing");
        }
        // System prompts and tool schemas count too.
        let tooled = count(&body(json!({"model":"m","system":"be brief","messages":[
            {"role":"user","content":[{"type":"text","text":"x"}]}
        ],"tools":[{"name":"lookup","description":"find","input_schema":{"type":"object"}}]})));
        assert!(tooled > base);
    }
}
