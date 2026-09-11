use super::super::{models::Model, protocol};
use crate::{Result, error::Error};
use serde_json::{Value, json};
use std::collections::{BTreeMap, HashSet};

pub fn native(mut body: Value, model: &Model, subscription: bool) -> Result<Value> {
    if crate::providers::http::contains_reference(&body) {
        return Err(Error::invalid(
            "tinyllm continuation references cannot be forwarded as native Responses input",
        ));
    }
    let object = body
        .as_object_mut()
        .ok_or_else(|| Error::invalid("request must be an object"))?;
    for field in ["background", "store"] {
        if object
            .get(field)
            .is_some_and(|v| !v.is_null() && v != false)
        {
            return Err(Error::invalid(format!(
                "{field}=true and stored response resources are unsupported"
            )));
        }
    }
    for field in ["previous_response_id", "conversation"] {
        if object.get(field).is_some_and(|v| !v.is_null()) {
            return Err(Error::invalid(format!(
                "{field} requires unsupported stored resources; replay native input items instead"
            )));
        }
    }
    if !object
        .get("input")
        .is_some_and(|v| v.is_string() || v.is_array())
        && !object.contains_key("prompt")
    {
        return Err(Error::invalid(
            "input must be text or an array of native Responses items",
        ));
    }
    if subscription {
        if let Some(input) = object.get_mut("input")
            && input.is_string()
        {
            *input = json!([{"role":"user","content":[{"type":"input_text","text":input}]}]);
        }
        for field in ["max_output_tokens", "truncation", "temperature", "top_p"] {
            if object.get(field).is_some_and(|v| !v.is_null()) {
                return Err(Error::invalid(format!(
                    "{field} is unsupported by the Codex subscription backend"
                )));
            }
            object.remove(field);
        }
        object.insert("stream".into(), json!(true));
        object.entry("instructions").or_insert(json!(""));
    }
    object.insert("model".into(), json!(model.id));
    object.entry("store").or_insert(json!(false));
    object
        .entry("include")
        .or_insert(json!(["reasoning.encrypted_content"]));
    if let Some(effort) = &model.reasoning_effort {
        let reasoning = object.entry("reasoning").or_insert(json!({}));
        if let Some(reasoning) = reasoning.as_object_mut() {
            reasoning.entry("effort").or_insert(json!(effort));
        }
    }
    Ok(body)
}

pub fn continuation(message: &Value) -> Result<Value> {
    protocol::fields(
        message,
        &[
            "role",
            "content",
            "refusal",
            "tool_calls",
            "annotations",
            "reasoning_details",
        ],
    )?;
    if message["role"] != "assistant" {
        return Err(Error::invalid(
            "continuation must be in an assistant message",
        ));
    }
    let details = message["reasoning_details"].as_array().filter(|d| d.len() == 1)
        .ok_or_else(|| Error::invalid("assistant history requires the unchanged reasoning_details continuation from tinyllm; start a new conversation with a user summary if the client stripped it"))?;
    protocol::fields(&details[0], &["type", "data"])?;
    if details[0]["type"] != "tinyllm_continuation" {
        return Err(Error::invalid("unsupported reasoning_details continuation"));
    }
    let reference = protocol::string(&details[0], "data")?;
    let mut content = message.clone();
    let content = content.as_object_mut().unwrap();
    content.remove("reasoning_details");
    content.retain(|_, value| !value.is_null());
    for key in ["content", "refusal"] {
        if content.get(key).is_some_and(|value| value == "") {
            content.remove(key);
        }
    }
    Ok(
        json!([{"type":"redacted_thinking","data":reference},{"type":"text","text":Value::Object(content.clone()).to_string()}]),
    )
}

pub fn history(request: &Value) -> Result<Value> {
    let messages = request["messages"]
        .as_array()
        .ok_or_else(|| Error::invalid("messages must be an array"))?;
    let history: Result<Vec<_>> = messages
        .iter()
        .map(|message| {
            if message["role"] == "assistant" {
                Ok(json!({"role":"assistant","content":continuation(message)?}))
            } else {
                Ok(json!({"role":"user","content":""}))
            }
        })
        .collect();
    Ok(json!({"messages":history?}))
}

pub fn chat(
    request: &Value,
    model: &Model,
    subscription: bool,
    restored: &BTreeMap<usize, Vec<Value>>,
) -> Result<Value> {
    protocol::fields(
        request,
        &[
            "model",
            "messages",
            "stream",
            "stream_options",
            "tools",
            "tool_choice",
            "parallel_tool_calls",
            "temperature",
            "top_p",
            "max_tokens",
            "max_completion_tokens",
            "response_format",
            "reasoning_effort",
            "verbosity",
            "metadata",
            "user",
            "safety_identifier",
            "prompt_cache_key",
            "prompt_cache_retention",
            "service_tier",
            "store",
            "n",
            "logprobs",
            "top_logprobs",
            "modalities",
        ],
    )?;
    if request.get("n").is_some_and(|v| !v.is_null() && v != 1) {
        return Err(Error::invalid("only n=1 is supported"));
    }
    if request
        .get("logprobs")
        .is_some_and(|v| !v.is_null() && v != false)
        || request.get("top_logprobs").is_some_and(|v| !v.is_null())
    {
        return Err(Error::invalid(
            "Chat logprobs are unsupported; use native Responses",
        ));
    }
    if request
        .get("modalities")
        .is_some_and(|v| !v.is_null() && *v != json!(["text"]))
    {
        return Err(Error::invalid(
            "only text output is supported by Chat conversion",
        ));
    }
    if let Some(options) = request.get("stream_options").filter(|v| !v.is_null()) {
        protocol::fields(options, &["include_usage"])?;
        if options
            .get("include_usage")
            .is_some_and(|v| !v.is_boolean())
        {
            return Err(Error::invalid(
                "stream_options.include_usage must be boolean",
            ));
        }
    }
    let messages = request["messages"]
        .as_array()
        .filter(|m| !m.is_empty())
        .ok_or_else(|| Error::invalid("messages must be a nonempty array"))?;
    let mut input = Vec::new();
    let mut calls = HashSet::new();
    let mut pending = HashSet::new();
    for (index, message) in messages.iter().enumerate() {
        let role = protocol::string(message, "role")?;
        let items = match role {
            "assistant" => {
                continuation(message)?;
                restored
                    .get(&index)
                    .cloned()
                    .ok_or_else(|| Error::invalid("assistant continuation was not restored"))?
            }
            "system" | "developer" | "user" => {
                protocol::fields(message, &["role", "content"])?;
                if !pending.is_empty() {
                    return Err(Error::invalid(
                        "tool results must follow assistant tool calls",
                    ));
                }
                vec![json!({"role":role,"content":content(&message["content"], role == "user")?})]
            }
            "tool" => {
                protocol::fields(message, &["role", "content", "tool_call_id"])?;
                let id = protocol::string(message, "tool_call_id")?;
                if !pending.remove(id) {
                    return Err(Error::invalid(
                        "tool result has no pending call or is duplicated",
                    ));
                }
                let output = if message["content"].is_string() {
                    message["content"].clone()
                } else {
                    json!(content(&message["content"], false)?)
                };
                vec![json!({"type":"function_call_output","call_id":id,"output":output})]
            }
            _ => return Err(Error::invalid("unsupported Chat message role")),
        };
        for item in &items {
            if item["type"] == "function_call" {
                let id = protocol::string(item, "call_id")?.to_owned();
                if !calls.insert(id.clone()) {
                    return Err(Error::invalid("duplicate tool call ID in history"));
                }
                pending.insert(id);
            }
        }
        input.extend(items);
    }
    if !pending.is_empty() {
        return Err(Error::invalid("history has tool calls without results"));
    }
    let mut output = json!({"model":model.id,"input":input,"stream":request["stream"].as_bool().unwrap_or(false),"store":false,"include":["reasoning.encrypted_content"]});
    for key in [
        "temperature",
        "top_p",
        "metadata",
        "user",
        "safety_identifier",
        "prompt_cache_key",
        "prompt_cache_retention",
        "service_tier",
        "parallel_tool_calls",
        "store",
    ] {
        if let Some(value) = request.get(key) {
            output[key] = value.clone();
        }
    }
    if request.get("max_tokens").is_some_and(|v| !v.is_null())
        && request
            .get("max_completion_tokens")
            .is_some_and(|v| !v.is_null())
    {
        return Err(Error::invalid(
            "use only one of max_tokens and max_completion_tokens",
        ));
    }
    if let Some(maximum) = request
        .get("max_completion_tokens")
        .filter(|v| !v.is_null())
        .or_else(|| request.get("max_tokens").filter(|v| !v.is_null()))
    {
        if maximum.as_u64().is_none_or(|n| n == 0) {
            return Err(Error::invalid("maximum tokens must be a positive integer"));
        }
        if !subscription {
            output["max_output_tokens"] = maximum.clone();
        }
    }
    if let Some(effort) = request.get("reasoning_effort") {
        if !effort.is_null() && !effort.is_string() {
            return Err(Error::invalid("reasoning_effort must be a string"));
        }
        output["reasoning"] = json!({"effort":effort});
    }
    if let Some(format) = request.get("response_format").filter(|v| !v.is_null()) {
        let converted = match protocol::string(format, "type")? {
            "text" | "json_object" => {
                protocol::fields(format, &["type"])?;
                format.clone()
            }
            "json_schema" => {
                protocol::fields(format, &["type", "json_schema"])?;
                let schema = &format["json_schema"];
                protocol::fields(schema, &["name", "description", "schema", "strict"])?;
                protocol::string(schema, "name")?;
                if !schema["schema"].is_object() {
                    return Err(Error::invalid("json_schema.schema must be an object"));
                }
                let mut schema = schema.clone();
                schema["type"] = json!("json_schema");
                schema
            }
            _ => return Err(Error::invalid("unsupported response_format")),
        };
        output["text"] = json!({"format":converted});
    }
    if let Some(verbosity) = request.get("verbosity").filter(|v| !v.is_null()) {
        if output["text"].is_null() {
            output["text"] = json!({});
        }
        output["text"]["verbosity"] = verbosity.clone();
    }
    let mut names = HashSet::new();
    if let Some(tools) = request.get("tools").filter(|v| !v.is_null()) {
        let tools = tools
            .as_array()
            .ok_or_else(|| Error::invalid("tools must be an array"))?;
        let converted: Result<Vec<_>> = tools
            .iter()
            .map(|tool| {
                protocol::fields(tool, &["type", "function"])?;
                if tool["type"] != "function" {
                    return Err(Error::invalid(
                        "only function tools are supported by Chat conversion",
                    ));
                }
                let function = &tool["function"];
                protocol::fields(function, &["name", "description", "parameters", "strict"])?;
                let name = protocol::string(function, "name")?;
                if name.is_empty()
                    || name.len() > 64
                    || !name
                        .bytes()
                        .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-'))
                    || !names.insert(name.to_owned())
                {
                    return Err(Error::invalid("invalid or duplicate function name"));
                }
                if function.get("parameters").is_some_and(|p| !p.is_object()) {
                    return Err(Error::invalid("function parameters must be an object"));
                }
                if function
                    .get("strict")
                    .is_some_and(|s| !s.is_null() && !s.is_boolean())
                {
                    return Err(Error::invalid("function strict must be boolean"));
                }
                let mut converted = function.clone();
                converted["type"] = json!("function");
                if converted["strict"].is_null() {
                    converted["strict"] = json!(false);
                }
                Ok(converted)
            })
            .collect();
        output["tools"] = json!(converted?);
    }
    if let Some(choice) = request.get("tool_choice").filter(|v| !v.is_null()) {
        output["tool_choice"] = if let Some(choice) = choice.as_str() {
            if !matches!(choice, "auto" | "none" | "required")
                || choice == "required" && names.is_empty()
            {
                return Err(Error::invalid("unsupported tool_choice or missing tools"));
            }
            json!(choice)
        } else {
            protocol::fields(choice, &["type", "function"])?;
            protocol::fields(&choice["function"], &["name"])?;
            let name = protocol::string(&choice["function"], "name")?;
            if choice["type"] != "function" || !names.contains(name) {
                return Err(Error::invalid("tool_choice must name a defined function"));
            }
            json!({"type":"function","name":name})
        };
    }
    native(output, model, subscription)
}

fn content(value: &Value, images: bool) -> Result<Vec<Value>> {
    if let Some(text) = value.as_str() {
        return Ok(vec![json!({"type":"input_text","text":text})]);
    }
    value
        .as_array()
        .ok_or_else(|| Error::invalid("message content must be text or an array"))?
        .iter()
        .map(|part| match protocol::string(part, "type")? {
            "text" => {
                protocol::fields(part, &["type", "text"])?;
                Ok(json!({"type":"input_text","text":protocol::string(part,"text")?}))
            }
            "image_url" if images => {
                protocol::fields(part, &["type", "image_url"])?;
                protocol::fields(&part["image_url"], &["url", "detail"])?;
                let url = protocol::string(&part["image_url"], "url")?;
                if !(url.starts_with("https://")
                    || url.starts_with("http://")
                    || url.starts_with("data:image/"))
                {
                    return Err(Error::invalid(
                        "image URL must be HTTP(S) or an image data URL",
                    ));
                }
                let mut image = json!({"type":"input_image","image_url":url});
                if let Some(detail) = part["image_url"].get("detail") {
                    if !matches!(detail.as_str(), Some("auto" | "low" | "high" | "original")) {
                        return Err(Error::invalid("unsupported image detail"));
                    }
                    image["detail"] = detail.clone();
                }
                Ok(image)
            }
            _ => Err(Error::invalid("unsupported Chat content part")),
        })
        .collect()
}
