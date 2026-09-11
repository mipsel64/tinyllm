use crate::{
    Result,
    error::Error,
    models::anthropic::{AnthropicResponse, ResponseContent, Usage},
    providers::openai::{models::Model, reasoning},
};
use base64::Engine;
use serde_json::{Value, json};
use std::collections::HashSet;

pub fn fields(value: &Value, allowed: &[&str]) -> Result<()> {
    let object = value
        .as_object()
        .ok_or_else(|| Error::invalid("expected a JSON object"))?;
    if let Some(key) = object.keys().find(|key| !allowed.contains(&key.as_str())) {
        let field: String = key
            .chars()
            .take(64)
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '_' {
                    c
                } else {
                    '?'
                }
            })
            .collect();
        tracing::warn!(field, "unsupported protocol field");
        return Err(Error::invalid(format!("unsupported field: {key}")));
    }
    Ok(())
}

pub fn string<'a>(value: &'a Value, key: &str) -> Result<&'a str> {
    value[key]
        .as_str()
        .ok_or_else(|| Error::invalid(format!("{key} must be a string")))
}

pub fn blocks(content: &Value) -> Result<Vec<Value>> {
    match content {
        Value::String(text) => Ok(vec![json!({"type":"text","text":text})]),
        Value::Array(blocks) => Ok(blocks.clone()),
        _ => Err(Error::invalid(
            "message content must be text or content blocks",
        )),
    }
}

/// Continuation metadata written on assistant text by pre-carrier builds.
const LEGACY_REFERENCE_FIELD: &str = "tinyllm_continuation";

pub fn request(req: &Value, model: &Model) -> Result<Value> {
    fields(
        req,
        &[
            "model",
            "messages",
            "max_tokens",
            "system",
            "temperature",
            "top_p",
            "top_k",
            "stop_sequences",
            "stream",
            "tools",
            "tool_choice",
            "metadata",
            "thinking",
            "output_config",
            "cache_control",
            "service_tier",
            "context_management",
        ],
    )?;
    string(req, "model")?;
    let max_tokens = req["max_tokens"]
        .as_u64()
        .filter(|n| *n > 0)
        .ok_or_else(|| Error::invalid("max_tokens must be positive"))?;
    let stops = super::stop::StopFilter::new(&req["stop_sequences"])?;
    if !req["top_k"].is_null() {
        return Err(Error::invalid("top_k is unsupported by Responses"));
    }
    if let Some(context) = req.get("context_management").filter(|v| !v.is_null()) {
        fields(context, &["edits"])?;
        if let Some(edits) = context.get("edits") {
            for edit in edits
                .as_array()
                .ok_or_else(|| Error::invalid("context_management.edits must be an array"))?
            {
                fields(edit, &["type", "keep"])?;
                if edit["type"] != "clear_thinking_20251015" || edit["keep"] != "all" {
                    return Err(Error::invalid(
                        "context_management supports clear_thinking_20251015 with keep=all; context clearing and server compaction are unsupported; client /compact remains available",
                    ));
                }
            }
        }
    }
    let stream = match req.get("stream") {
        None => false,
        Some(v) => v
            .as_bool()
            .ok_or_else(|| Error::invalid("stream must be boolean"))?,
    };
    let mut out = json!({"model":model.id,"max_output_tokens":max_tokens,"stream":stream,"store":false,"include":["reasoning.encrypted_content"],"truncation":"disabled"});
    let mut input = Vec::new();
    if let Some(system) = req.get("system").filter(|s| !s.is_null()) {
        input.push(json!({"role":"system","content":text_image_content(system, "system")?}));
    }
    let messages = req["messages"]
        .as_array()
        .filter(|m| !m.is_empty())
        .ok_or_else(|| Error::invalid("messages must be a nonempty array"))?;
    let mut pending = HashSet::new();
    let mut pending_client_calls = HashSet::new();
    let mut calls = HashSet::new();
    let mut loaded = HashSet::new();
    for message in messages {
        fields(message, &["role", "content", "cache_control"])?;
        let role = string(message, "role")?;
        if !matches!(role, "user" | "assistant" | "system") {
            return Err(Error::invalid("unsupported message role"));
        }
        if role == "assistant" {
            for block in message["content"]
                .as_array()
                .into_iter()
                .flatten()
                .filter(|block| block["type"] == "tool_use")
            {
                let id = string(block, "id")?;
                if id.is_empty() || !pending_client_calls.insert(id) {
                    return Err(Error::invalid("empty or duplicate tool call ID"));
                }
            }
        }
        let mut items = Vec::new();
        let mut parts = Vec::new();
        for mut block in blocks(&message["content"])? {
            match string(&block, "type")? {
                "text" | "image" => {
                    // Sessions started before carriers tagged assistant text with
                    // a continuation reference. Drop it rather than reject them.
                    if let Some(object) = block.as_object_mut() {
                        object.remove(LEGACY_REFERENCE_FIELD);
                    }
                    parts.extend(text_image_content(&json!([block]), role)?);
                }
                "tool_use" if role == "assistant" => {
                    flush(&mut items, &mut parts, role);
                    fields(&block, &["type", "id", "name", "input", "cache_control"])?;
                    let id = string(&block, "id")?;
                    let name = string(&block, "name")?;
                    valid_name(name)?;
                    if !block["input"].is_object() {
                        return Err(Error::invalid("tool input must be an object"));
                    }
                    items.push(json!({"type":"function_call","call_id":id,"name":name,"arguments":block["input"].to_string()}));
                }
                "tool_result" if role == "user" => {
                    flush(&mut items, &mut parts, role);
                    fields(
                        &block,
                        &[
                            "type",
                            "tool_use_id",
                            "content",
                            "is_error",
                            "cache_control",
                        ],
                    )?;
                    let id = string(&block, "tool_use_id")?;
                    if !pending_client_calls.remove(id) {
                        return Err(Error::invalid(
                            "tool result has no matching unresolved client tool call",
                        ));
                    }
                    let is_error = match block.get("is_error") {
                        None => false,
                        Some(v) => v
                            .as_bool()
                            .ok_or_else(|| Error::invalid("is_error must be boolean"))?,
                    };
                    let mut output = match &block["content"] {
                        Value::Null => json!(""),
                        Value::String(_) => block["content"].clone(),
                        Value::Array(_) => {
                            let mut parts = Vec::new();
                            for part in blocks(&block["content"])? {
                                if part["type"] == "tool_reference" {
                                    fields(&part, &["type", "tool_name", "cache_control"])?;
                                    let name = string(&part, "tool_name")?;
                                    loaded.insert(name.to_owned());
                                    let reference =
                                        json!({"type":"tool_reference","tool_name":name});
                                    parts.push(
                                        json!({"type":"input_text","text":reference.to_string()}),
                                    );
                                } else {
                                    parts.extend(text_image_content(&json!([part]), "user")?);
                                }
                            }
                            json!(parts)
                        }
                        v => json!(v.to_string()),
                    };
                    if is_error {
                        if let Some(parts) = output.as_array_mut() {
                            parts.insert(
                                0,
                                json!({"type":"input_text","text":"{\"is_error\":true}"}),
                            );
                        } else {
                            output = json!(json!({"is_error":true,"content":output}).to_string());
                        }
                    }
                    items.push(json!({"type":"function_call_output","call_id":id,"output":output}));
                }
                "thinking" if role == "assistant" => {
                    fields(&block, &["type", "thinking", "signature", "cache_control"])?;
                    string(&block, "thinking")?;
                    if block
                        .get("signature")
                        .is_some_and(|value| !value.is_null() && !value.is_string())
                    {
                        return Err(Error::invalid("thinking signature must be a string"));
                    }
                }
                "redacted_thinking" if role == "assistant" => {
                    fields(&block, &["type", "data", "cache_control"])?;
                    if let Some(replay) = reasoning::decode(string(&block, "data")?) {
                        flush(&mut items, &mut parts, role);
                        items.push(replay.item());
                    }
                }
                "thinking" | "redacted_thinking" => {
                    return Err(Error::invalid(
                        "thinking blocks belong to assistant messages",
                    ));
                }
                other => {
                    return Err(Error::invalid(format!(
                        "unsupported {role} content block: {other}"
                    )));
                }
            }
        }
        flush(&mut items, &mut parts, role);
        for item in &items {
            match item["type"].as_str() {
                Some("function_call") => {
                    let id = string(item, "call_id")?.to_owned();
                    if id.is_empty() || !calls.insert(id.clone()) {
                        return Err(Error::invalid("empty or duplicate tool call ID"));
                    }
                    pending.insert(id);
                }
                Some("function_call_output") if !pending.remove(string(item, "call_id")?) => {
                    return Err(Error::invalid(
                        "tool result has no matching unresolved tool call",
                    ));
                }
                _ => {}
            }
        }
        input.extend(items);
    }
    if !pending.is_empty() || !pending_client_calls.is_empty() {
        return Err(Error::invalid("history has tool calls without results"));
    }
    out["input"] = json!(input);
    let mut names = HashSet::new();
    let mut available = HashSet::new();
    let mut search = false;
    if let Some(tools) = req.get("tools") {
        let tools = tools
            .as_array()
            .ok_or_else(|| Error::invalid("tools must be an array"))?;
        let mut translated = Vec::new();
        for tool in tools {
            let name = string(tool, "name")?;
            valid_name(name)?;
            if !names.insert(name) {
                return Err(Error::invalid("duplicate tool name"));
            }
            if tool["type"] == "web_search_20250305" {
                if stops.enabled() {
                    return Err(Error::invalid(
                        "web search with stop_sequences is unsupported",
                    ));
                }
                translated.push(web_search_tool(tool)?);
                if let Some(limit) = tool.get("max_uses").filter(|v| !v.is_null()) {
                    out["max_tool_calls"] = json!(
                        limit
                            .as_u64()
                            .filter(|n| *n > 0)
                            .ok_or_else(|| Error::invalid(
                                "web search max_uses must be positive"
                            ))?
                    );
                }
                available.insert(name);
                search = true;
                continue;
            }
            if tool.get("type").is_some_and(|v| v != "custom") {
                return Err(Error::invalid(
                    "unsupported hosted tool type; only web_search_20250305 is supported",
                ));
            }
            fields(
                tool,
                &[
                    "type",
                    "name",
                    "description",
                    "input_schema",
                    "cache_control",
                    "strict",
                    "defer_loading",
                    "input_examples",
                ],
            )?;
            let deferred = match tool.get("defer_loading") {
                None => false,
                Some(value) => value
                    .as_bool()
                    .ok_or_else(|| Error::invalid("defer_loading must be boolean"))?,
            };
            if !tool["input_schema"].is_object() {
                return Err(Error::invalid("input_schema must be a JSON schema object"));
            }
            let strict = match tool.get("strict") {
                None => false,
                Some(v) => v
                    .as_bool()
                    .ok_or_else(|| Error::invalid("tool strict must be boolean"))?,
            };
            let mut t = json!({"type":"function","name":name,"parameters":tool["input_schema"],"strict":strict});
            if let Some(description) = tool.get("description") {
                t["description"] = json!(
                    description
                        .as_str()
                        .ok_or_else(|| Error::invalid("description must be text"))?
                );
            }
            if let Some(examples) = tool.get("input_examples") {
                if !examples.is_array() {
                    return Err(Error::invalid("input_examples must be an array"));
                }
                t["description"] = json!(format!(
                    "{}\nInput examples: {}",
                    t["description"].as_str().unwrap_or(""),
                    examples
                ));
            }
            if !deferred || loaded.contains(name) {
                available.insert(name);
                translated.push(t);
            }
        }
        out["tools"] = json!(translated);
    }
    if loaded.iter().any(|name| !names.contains(name.as_str())) {
        return Err(Error::invalid(
            "tool_reference names an undefined tool; include its definition in tools",
        ));
    }
    if let Some(choice) = req.get("tool_choice") {
        fields(choice, &["type", "name", "disable_parallel_tool_use"])?;
        out["tool_choice"] = match string(choice, "type")? {
            "auto" => json!("auto"),
            "none" => json!("none"),
            "any" if !available.is_empty() => json!("required"),
            "tool" if search && choice["name"] == "web_search" => {
                if available.len() == 1 {
                    json!("required")
                } else {
                    json!({"type":"web_search"})
                }
            }
            "tool" if available.contains(string(choice, "name")?) => {
                json!({"type":"function","name":choice["name"]})
            }
            _ => {
                return Err(Error::invalid(
                    "tool_choice requires a supported mode and a loaded tool",
                ));
            }
        };
        if let Some(disable) = choice.get("disable_parallel_tool_use") {
            out["parallel_tool_calls"] = json!(
                !disable
                    .as_bool()
                    .ok_or_else(|| Error::invalid("disable_parallel_tool_use must be boolean"))?
            );
        }
    }
    let mut effort = model.reasoning_effort.map(|effort| effort.as_str());
    if let Some(thinking) = req.get("thinking") {
        fields(thinking, &["type", "budget_tokens", "display"])?;
        match string(thinking, "type")? {
            "adaptive" => {}
            "disabled" => effort = Some("none"),
            "enabled" => {
                return Err(Error::invalid(
                    "capability_rejected: thinking; exact budget_tokens has no Responses equivalent; use adaptive thinking",
                ));
            }
            _ => return Err(Error::invalid("unsupported thinking mode")),
        }
        if thinking.get("budget_tokens").is_some()
            || thinking
                .get("display")
                .is_some_and(|display| display != "omitted" || thinking["type"] == "disabled")
        {
            return Err(Error::invalid(
                "thinking supports adaptive mode with display=omitted; exact budgets and displayed summaries are unsupported",
            ));
        }
    }
    if let Some(config) = req.get("output_config") {
        fields(config, &["effort", "format"])?;
        if let Some(e) = config.get("effort") {
            effort = Some(
                e.as_str()
                    .ok_or_else(|| Error::invalid("effort must be a string"))?,
            );
        }
        if let Some(format) = config.get("format") {
            if search {
                return Err(Error::invalid(
                    "web search with structured output is unsupported",
                ));
            }
            fields(format, &["type", "schema"])?;
            if format["type"] != "json_schema" || !format["schema"].is_object() {
                return Err(Error::invalid("unsupported output format"));
            }
            out["text"] = json!({"format":{"type":"json_schema","name":"claude_output","schema":format["schema"],"strict":true}});
        }
    }
    if req["thinking"]["type"] == "disabled" {
        effort = Some("none");
    }
    if let Some(effort) = effort {
        crate::providers::validate_effort(effort)?;
        out["reasoning"] = json!({"effort": model.cap_effort(effort)});
    } else if let Some(ceiling) = model.max_reasoning_effort {
        // No client choice to cap, but an unstated default can still exceed the
        // ceiling upstream, so state it.
        out["reasoning"] = json!({"effort":ceiling.as_str()});
    }
    for (key, max) in [("temperature", 2.0), ("top_p", 1.0)] {
        if let Some(value) = req.get(key) {
            if value.as_f64().is_none_or(|v| !(0.0..=max).contains(&v)) {
                return Err(Error::invalid(format!("invalid {key}")));
            }
            out[key] = value.clone();
        }
    }
    if let Some(tier) = req.get("service_tier") {
        out["service_tier"] = match tier.as_str() {
            Some("auto") => json!("auto"),
            Some("standard_only") => json!("default"),
            _ => return Err(Error::invalid("unsupported service_tier")),
        };
    }
    Ok(out)
}

fn web_search_tool(tool: &Value) -> Result<Value> {
    fields(
        tool,
        &[
            "type",
            "name",
            "max_uses",
            "allowed_domains",
            "blocked_domains",
            "user_location",
            "cache_control",
            "allowed_callers",
        ],
    )?;
    if tool["name"] != "web_search" {
        return Err(Error::invalid("hosted web search must be named web_search"));
    }
    if tool
        .get("allowed_callers")
        .is_some_and(|v| *v != json!(["direct"]))
    {
        return Err(Error::invalid("web search supports only direct calls"));
    }
    let mut out = json!({"type":"web_search"});
    let mut filters = serde_json::Map::new();
    for key in ["allowed_domains", "blocked_domains"] {
        if let Some(domains) = tool.get(key).filter(|v| !v.is_null()) {
            let domains = domains
                .as_array()
                .filter(|v| v.len() <= 100)
                .ok_or_else(|| {
                    Error::invalid("web search domain lists must contain at most 100 domains")
                })?;
            let mut values = Vec::new();
            for domain in domains {
                let domain = domain
                    .as_str()
                    .filter(|s| !s.is_empty() && s.len() <= 253)
                    .ok_or_else(|| {
                        Error::invalid("web search domains must be bare ASCII hostnames")
                    })?;
                if !domain.split('.').all(|label| {
                    !label.is_empty()
                        && label.len() <= 63
                        && !label.starts_with('-')
                        && !label.ends_with('-')
                        && label
                            .bytes()
                            .all(|c| c.is_ascii_alphanumeric() || c == b'-')
                }) {
                    return Err(Error::invalid(
                        "web search domains must be bare ASCII hostnames; paths, schemes and wildcards are unsupported",
                    ));
                }
                values.push(json!(domain.to_ascii_lowercase()));
            }
            if !values.is_empty() {
                filters.insert(key.into(), json!(values));
            }
        }
    }
    if filters.len() > 1 {
        return Err(Error::invalid(
            "web search cannot combine allowed_domains and blocked_domains",
        ));
    }
    if !filters.is_empty() {
        out["filters"] = Value::Object(filters);
    }
    if let Some(location) = tool.get("user_location").filter(|v| !v.is_null()) {
        fields(location, &["type", "city", "region", "country", "timezone"])?;
        if location["type"] != "approximate" {
            return Err(Error::invalid(
                "web search user_location must be approximate",
            ));
        }
        let mut present = false;
        for key in ["city", "region", "country", "timezone"] {
            if let Some(value) = location.get(key) {
                let text = value
                    .as_str()
                    .filter(|s| !s.is_empty() && !s.chars().any(char::is_control))
                    .ok_or_else(|| {
                        Error::invalid("web search location values must be nonempty strings")
                    })?;
                if key == "country"
                    && (text.len() != 2 || !text.bytes().all(|c| c.is_ascii_alphabetic()))
                {
                    return Err(Error::invalid(
                        "web search country must be a two-letter code",
                    ));
                }
                present = true;
            }
        }
        if !present {
            return Err(Error::invalid(
                "web search user_location must specify a location",
            ));
        }
        out["user_location"] = location.clone();
    }
    Ok(out)
}

pub fn subscription_request(req: &Value, mut out: Value) -> Result<Value> {
    if out.get("temperature").is_some() || out.get("top_p").is_some() {
        return Err(Error::invalid(
            "temperature and top_p are unsupported by the subscription backend",
        ));
    }
    let mut instructions = Vec::new();
    if let Some(system) = req.get("system").filter(|s| !s.is_null()) {
        for block in blocks(system)? {
            if block["type"] != "text" {
                return Err(Error::invalid(
                    "subscription system instructions must contain only text",
                ));
            }
            instructions.push(string(&block, "text")?.to_owned());
        }
        out["input"].as_array_mut().unwrap().remove(0);
    }
    for item in out["input"].as_array_mut().unwrap() {
        if item["role"] == "system" {
            item["role"] = json!("developer");
        }
    }
    if let Some(limit) = out.as_object_mut().unwrap().remove("max_tool_calls") {
        instructions.push(format!(
            "Search budget: use the web_search tool at most {limit} times for this response."
        ));
        tracing::warn!(max_uses = %limit, "subscription web search limit is best-effort; the backend does not support a hard cap");
    }
    out["instructions"] = json!(instructions.join("\n\n"));
    out["stream"] = json!(true);
    out.as_object_mut().unwrap().remove("max_output_tokens");
    out.as_object_mut().unwrap().remove("truncation");
    Ok(out)
}

fn flush(items: &mut Vec<Value>, parts: &mut Vec<Value>, role: &str) {
    if !parts.is_empty() {
        items.push(json!({"role":role,"content":std::mem::take(parts)}));
    }
}

pub(super) fn valid_name(name: &str) -> Result<()> {
    if name.is_empty()
        || name.len() > 64
        || !name
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'_' | b'-'))
    {
        return Err(Error::invalid(
            "OpenAI tool names must contain 1–64 ASCII letters, digits, underscores or hyphens",
        ));
    }
    Ok(())
}

fn text_image_content(content: &Value, role: &str) -> Result<Vec<Value>> {
    blocks(content)?
        .iter()
        .map(|block| match string(block, "type")? {
            "text" => {
                fields(block, &["type", "text", "cache_control"])?;
                let kind = if role == "assistant" {
                    "output_text"
                } else {
                    "input_text"
                };
                Ok(json!({"type":kind,"text":string(block,"text")?}))
            }
            "image" if role == "user" => {
                fields(block, &["type", "source", "cache_control"])?;
                let source = &block["source"];
                let url = match string(source, "type")? {
                    "url" => {
                        fields(source, &["type", "url"])?;
                        let url = string(source, "url")?;
                        let parsed = reqwest::Url::parse(url)
                            .map_err(|_| Error::invalid("invalid image URL"))?;
                        if !matches!(parsed.scheme(), "https" | "http")
                            || !parsed.username().is_empty()
                            || parsed.password().is_some()
                        {
                            return Err(Error::invalid(
                                "image URL must be HTTP(S) without credentials",
                            ));
                        }
                        url.to_owned()
                    }
                    "base64" => {
                        fields(source, &["type", "media_type", "data"])?;
                        let media = string(source, "media_type")?;
                        if !matches!(
                            media,
                            "image/png" | "image/jpeg" | "image/gif" | "image/webp"
                        ) {
                            return Err(Error::invalid("unsupported image media_type"));
                        }
                        let data = string(source, "data")?;
                        if data.is_empty()
                            || base64::engine::general_purpose::STANDARD
                                .decode(data)
                                .is_err()
                        {
                            return Err(Error::invalid("invalid base64 image data"));
                        }
                        format!("data:{media};base64,{data}")
                    }
                    _ => return Err(Error::invalid("unsupported image source")),
                };
                Ok(json!({"type":"input_image","image_url":url}))
            }
            other => Err(Error::invalid(format!(
                "unsupported {role} content: {other}"
            ))),
        })
        .collect()
}

pub fn usage(response: &Value) -> Result<Usage> {
    let u = &response["usage"];
    let input = u["input_tokens"]
        .as_u64()
        .ok_or_else(|| Error::upstream("upstream omitted input usage"))?;
    let output = u["output_tokens"]
        .as_u64()
        .ok_or_else(|| Error::upstream("upstream omitted output usage"))?;
    let cached = u["input_tokens_details"]["cached_tokens"]
        .as_u64()
        .unwrap_or(0);
    Ok(Usage {
        input_tokens: input
            .checked_sub(cached)
            .ok_or_else(|| Error::upstream("invalid upstream cached usage"))?,
        output_tokens: output,
        cache_read_input_tokens: cached,
        cache_creation_input_tokens: 0,
    })
}

pub(super) fn web_search_output(item: &Value) -> Result<()> {
    if item["id"].as_str().is_none_or(str::is_empty)
        || !matches!(
            item["status"].as_str(),
            Some("completed" | "failed" | "incomplete")
        )
    {
        return Err(Error::upstream(
            "web search output has no ID or a nonterminal status",
        ));
    }
    Ok(())
}

pub(super) fn citation_url(annotation: &Value) -> Result<String> {
    if annotation["type"] != "url_citation"
        || !annotation["title"].is_string()
        || annotation["start_index"]
            .as_u64()
            .zip(annotation["end_index"].as_u64())
            .is_none_or(|(start, end)| start > end)
    {
        return Err(Error::upstream(
            "unsupported or malformed output annotation",
        ));
    }
    let url = annotation["url"]
        .as_str()
        .filter(|s| !s.chars().any(char::is_control))
        .and_then(|s| reqwest::Url::parse(s).ok())
        .filter(|url| {
            matches!(url.scheme(), "http" | "https")
                && url.host_str().is_some()
                && url.username().is_empty()
                && url.password().is_none()
        })
        .ok_or_else(|| Error::upstream("citation URL must be HTTP(S) without credentials"))?;
    Ok(url.into())
}

pub(super) fn annotations(part: &Value) -> Result<&[Value]> {
    let annotations = match part.get("annotations") {
        None | Some(Value::Null) => return Ok(&[]),
        Some(Value::Array(values)) => values,
        _ => return Err(Error::upstream("output annotations must be an array")),
    };
    Ok(annotations)
}

pub(super) fn citation_sources(response: &Value) -> Result<String> {
    let mut seen = HashSet::new();
    let mut sources = String::new();
    for item in response["output"]
        .as_array()
        .ok_or_else(|| Error::upstream("upstream output missing"))?
    {
        if item["type"] != "message" {
            continue;
        }
        for part in item["content"]
            .as_array()
            .ok_or_else(|| Error::upstream("upstream message content missing"))?
        {
            for annotation in annotations(part)? {
                let url = citation_url(annotation)?;
                if seen.insert(url.clone()) {
                    if sources.is_empty() {
                        sources.push_str("\n\nSources:");
                    }
                    sources.push_str(&format!("\n- <{url}>"));
                }
            }
        }
    }
    Ok(sources)
}

pub fn response(response: &Value, alias: &str) -> Result<AnthropicResponse> {
    if response["status"] == "failed" {
        return Err(Error::openai(response));
    }
    let mut content = Vec::new();
    let mut tool_use = false;
    let mut call_ids = HashSet::new();
    let mut refusal = false;
    for item in response["output"]
        .as_array()
        .ok_or_else(|| Error::upstream("upstream output missing"))?
    {
        match item["type"].as_str() {
            Some("reasoning") => {
                if let Some(data) = reasoning::capture(item)
                    .as_ref()
                    .and_then(reasoning::encode)
                {
                    content.push(ResponseContent::RedactedThinking {
                        content_type: "redacted_thinking".into(),
                        data,
                    });
                }
            }
            Some("message") => {
                if item["role"] != "assistant" {
                    return Err(Error::upstream("upstream message must have assistant role"));
                }
                for part in item["content"]
                    .as_array()
                    .ok_or_else(|| Error::upstream("upstream message content missing"))?
                {
                    let text = match part["type"].as_str() {
                        Some("output_text") => part["text"].as_str(),
                        Some("refusal") => {
                            refusal = true;
                            part["refusal"].as_str()
                        }
                        _ => return Err(Error::upstream("unsupported upstream message content")),
                    }
                    .ok_or_else(|| Error::upstream("upstream text missing"))?;
                    content.push(ResponseContent::Text {
                        content_type: "text".into(),
                        text: text.into(),
                    });
                }
            }
            Some("web_search_call") => web_search_output(item)?,
            Some("function_call") => {
                let arguments = item["arguments"]
                    .as_str()
                    .ok_or_else(|| Error::upstream("tool arguments missing"))?;
                let name = item["name"].as_str().unwrap_or_default();
                let arguments = super::tool_args::sanitize(name, arguments).map_or(
                    std::borrow::Cow::Borrowed(arguments),
                    std::borrow::Cow::Owned,
                );
                let input: Value = serde_json::from_str(&arguments).map_err(|_| {
                    Error::upstream(
                        "upstream returned incomplete or invalid tool JSON; tool was not executed",
                    )
                })?;
                if !input.is_object() {
                    return Err(Error::upstream("tool arguments must be an object"));
                }
                let id = item["call_id"]
                    .as_str()
                    .filter(|id| !id.is_empty())
                    .ok_or_else(|| Error::upstream("tool call ID missing"))?;
                if !call_ids.insert(id) {
                    return Err(Error::upstream("duplicate upstream tool call ID"));
                }
                let name = item["name"]
                    .as_str()
                    .ok_or_else(|| Error::upstream("tool name missing"))?;
                content.push(ResponseContent::ToolUse {
                    content_type: "tool_use".into(),
                    id: id.into(),
                    name: name.into(),
                    input,
                });
                tool_use = true;
            }
            _ => return Err(Error::upstream("unsupported upstream output item")),
        }
    }
    let sources = citation_sources(response)?;
    if !sources.is_empty() {
        content.push(ResponseContent::Text {
            content_type: "text".into(),
            text: sources,
        });
    }
    // A completed turn carrying only reasoning is not something the client can
    // act on, and as a success it would surface as the model silently saying
    // nothing. Fail loudly instead.
    // ponytail: errors rather than retrying; retry upstream if these turn out common.
    if response["status"] == "completed"
        && !content.iter().any(|block| match block {
            // Empty text is as unusable as no text at all.
            ResponseContent::Text { text, .. } => !text.is_empty(),
            ResponseContent::ToolUse { .. } => true,
            _ => false,
        })
    {
        return Err(Error::upstream(
            "upstream completed without any text or tool call; nothing to return",
        ));
    }
    let reason = match response["status"].as_str() {
        Some("completed") => {
            if tool_use {
                "tool_use"
            } else if refusal {
                "refusal"
            } else {
                "end_turn"
            }
        }
        Some("incomplete") => match response["incomplete_details"]["reason"].as_str() {
            Some("max_output_tokens") => "max_tokens",
            Some("content_filter") => "refusal",
            _ => return Err(Error::upstream("unsupported incomplete response reason")),
        },
        _ => {
            return Err(Error::upstream(
                "upstream response failed or did not complete",
            ));
        }
    };
    Ok(AnthropicResponse {
        id: response["id"]
            .as_str()
            .ok_or_else(|| Error::upstream("response ID missing"))?
            .into(),
        response_type: "message".into(),
        role: "assistant".into(),
        content,
        model: alias.into(),
        stop_reason: Some(reason.into()),
        stop_sequence: None,
        usage: usage(response)?,
    })
}
