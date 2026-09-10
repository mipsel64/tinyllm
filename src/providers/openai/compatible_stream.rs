use crate::{Result, error::Error};
use serde_json::{Value, json};
use std::collections::{BTreeMap, HashSet};

struct Item {
    initial: Value,
    complete: Option<Value>,
    fragments: BTreeMap<(usize, String), String>,
}

pub struct Native {
    sparse: bool,
    created: Option<Value>,
    items: Vec<Item>,
    completed: Option<Value>,
}

impl Native {
    pub fn new(sparse: bool) -> Self {
        Self {
            sparse,
            created: None,
            items: Vec::new(),
            completed: None,
        }
    }

    pub fn accept(&mut self, mut event: Value) -> Result<Value> {
        if self.completed.is_some() {
            return Err(Error::upstream("event after response completion"));
        }
        let kind = text(&event, "type")?.to_owned();
        if kind == "error" {
            return Err(Error::openai(&event));
        }
        if kind == "response.failed" {
            return Err(Error::openai(&event["response"]));
        }
        if matches!(
            kind.as_str(),
            "response.metadata" | "codex.response.metadata" | "ping"
        ) {
            return Ok(event);
        }
        if kind == "response.created" {
            if self.created.is_some() {
                return Err(Error::upstream("duplicate response.created"));
            }
            text(&event["response"], "id")?;
            self.created = Some(event["response"].clone());
            return Ok(event);
        }
        let created = self
            .created
            .as_ref()
            .ok_or_else(|| Error::upstream("SSE response.created is missing"))?;
        match kind.as_str() {
            "response.output_item.added" => {
                if index(&event, "output_index")? != self.items.len() {
                    return Err(Error::upstream("out-of-order or duplicate output item"));
                }
                let item = &event["item"];
                text(item, "id")?;
                text(item, "type")?;
                let mut fragments = BTreeMap::new();
                if item["type"] == "function_call" {
                    fragments.insert((0, "arguments".into()), text(item, "arguments")?.into());
                }
                self.items.push(Item {
                    initial: item.clone(),
                    complete: None,
                    fragments,
                });
            }
            "response.content_part.added" => {
                let part = &event["part"];
                let key = match part["type"].as_str() {
                    Some("output_text") => Some("text"),
                    Some("refusal") => Some("refusal"),
                    _ => None,
                };
                if let Some(key) = key {
                    let content_index = index(&event, "content_index")?;
                    let item = self.item(&event)?;
                    if item
                        .fragments
                        .insert((content_index, key.into()), text(part, key)?.into())
                        .is_some()
                    {
                        return Err(Error::upstream("duplicate content part"));
                    }
                }
            }
            "response.output_text.delta"
            | "response.refusal.delta"
            | "response.function_call_arguments.delta" => {
                let (part, key) = fragment(&event, &kind)?;
                let delta = text(&event, "delta")?;
                let item = self.item(&event)?;
                let value = item
                    .fragments
                    .get_mut(&(part, key.into()))
                    .ok_or_else(|| Error::upstream("delta before content or tool call start"))?;
                value.push_str(delta);
            }
            "response.output_text.done"
            | "response.refusal.done"
            | "response.function_call_arguments.done" => {
                let (part, key) = fragment(&event, &kind)?;
                let expected = text(&event, key)?;
                if self
                    .item(&event)?
                    .fragments
                    .get(&(part, key.into()))
                    .map(String::as_str)
                    != Some(expected)
                {
                    return Err(Error::upstream(
                        "completed content differs from streamed deltas",
                    ));
                }
            }
            "response.output_item.done" => {
                let item = self.item(&event)?;
                let final_item = &event["item"];
                if final_item["id"] != item.initial["id"]
                    || final_item["type"] != item.initial["type"]
                {
                    return Err(Error::upstream("completed output item identity changed"));
                }
                if final_item["type"] == "function_call" {
                    if final_item["call_id"] != item.initial["call_id"]
                        || final_item["name"] != item.initial["name"]
                    {
                        return Err(Error::upstream("completed tool identity changed"));
                    }
                    if item
                        .fragments
                        .get(&(0, "arguments".into()))
                        .map(String::as_str)
                        != Some(text(final_item, "arguments")?)
                    {
                        return Err(Error::upstream(
                            "completed tool arguments differ from streamed deltas",
                        ));
                    }
                }
                if final_item["type"] == "message" {
                    let parts = final_item["content"]
                        .as_array()
                        .ok_or_else(|| Error::upstream("upstream message content is missing"))?;
                    if parts
                        .iter()
                        .filter(|part| {
                            matches!(part["type"].as_str(), Some("output_text" | "refusal"))
                        })
                        .count()
                        != item.fragments.len()
                    {
                        return Err(Error::upstream(
                            "completed message removed streamed content",
                        ));
                    }
                    for (index, part) in parts.iter().enumerate() {
                        let key = match part["type"].as_str() {
                            Some("output_text") => "text",
                            Some("refusal") => "refusal",
                            _ => continue,
                        };
                        if item.fragments.get(&(index, key.into())).map(String::as_str)
                            != Some(text(part, key)?)
                        {
                            return Err(Error::upstream(
                                "completed message differs from streamed deltas",
                            ));
                        }
                    }
                }
                item.complete = Some(final_item.clone());
            }
            "response.completed" | "response.incomplete" => {
                let response = event["response"]
                    .as_object_mut()
                    .ok_or_else(|| Error::upstream("terminal event has no response"))?;
                if response.get("id") != created.get("id") {
                    return Err(Error::upstream("response identity changed"));
                }
                if self.sparse {
                    for (key, value) in created.as_object().unwrap() {
                        if !matches!(
                            key.as_str(),
                            "output" | "status" | "usage" | "error" | "incomplete_details"
                        ) {
                            response.entry(key.clone()).or_insert(value.clone());
                        }
                    }
                    if response
                        .get("output")
                        .is_none_or(|v| v.is_null() || v.as_array().is_some_and(Vec::is_empty))
                    {
                        let output: Option<Vec<_>> =
                            self.items.iter().map(|i| i.complete.clone()).collect();
                        response.insert(
                            "output".into(),
                            json!(output.ok_or_else(|| Error::upstream(
                                "subscription completion has unfinished output items"
                            ))?),
                        );
                    }
                    response
                        .entry("status")
                        .or_insert(json!(kind.strip_prefix("response.").unwrap()));
                }
                if response.get("status").and_then(Value::as_str) != kind.strip_prefix("response.")
                {
                    return Err(Error::upstream(
                        "terminal response status disagrees with its event",
                    ));
                }
                let output = response
                    .get("output")
                    .and_then(Value::as_array)
                    .ok_or_else(|| Error::upstream("terminal response output is missing"))?;
                if output.len() != self.items.len()
                    || self
                        .items
                        .iter()
                        .zip(output)
                        .any(|(item, output)| item.complete.as_ref() != Some(output))
                {
                    return Err(Error::upstream(
                        "terminal output differs from streamed items",
                    ));
                }
                super::validate_response(&event["response"])?;
                self.completed = Some(event["response"].clone());
            }
            _ => {}
        }
        Ok(event)
    }

    fn item(&mut self, event: &Value) -> Result<&mut Item> {
        let item = self
            .items
            .get_mut(index(event, "output_index")?)
            .ok_or_else(|| Error::upstream("event refers to an unknown output item"))?;
        if item.complete.is_some() {
            return Err(Error::upstream("event after output item completion"));
        }
        if event
            .get("item_id")
            .is_some_and(|id| *id != item.initial["id"])
        {
            return Err(Error::upstream("event item ID disagrees with output index"));
        }
        Ok(item)
    }

    pub fn is_complete(&self) -> bool {
        self.completed.is_some()
    }
    pub fn finish(self) -> Result<Value> {
        self.completed
            .ok_or_else(|| Error::upstream("upstream stream ended without a terminal response"))
    }
}

pub struct Chat {
    model: String,
    id: Option<String>,
    created: Value,
    tools: BTreeMap<usize, usize>,
    calls: HashSet<String>,
}

impl Chat {
    pub fn new(model: String) -> Self {
        Self {
            model,
            id: None,
            created: Value::Null,
            tools: BTreeMap::new(),
            calls: HashSet::new(),
        }
    }

    pub fn chunk(&self, delta: Value, finish_reason: Value) -> Result<Value> {
        let id = self
            .id
            .as_ref()
            .ok_or_else(|| Error::upstream("missing response.created"))?;
        Ok(
            json!({"id":id,"object":"chat.completion.chunk","created":self.created,"model":self.model,"choices":[{"index":0,"delta":delta,"finish_reason":finish_reason,"logprobs":null}]}),
        )
    }

    pub fn accept(&mut self, event: &Value) -> Result<Vec<Value>> {
        let delta = match text(event, "type")? {
            "response.created" => {
                self.id = Some(text(&event["response"], "id")?.into());
                self.created = super::created_at(&event["response"])?;
                json!({"role":"assistant","content":""})
            }
            "response.output_item.added" if event["item"]["type"] == "function_call" => {
                let item = &event["item"];
                let id = text(item, "call_id")?;
                if id.is_empty() || !self.calls.insert(id.into()) {
                    return Err(Error::upstream("empty or duplicate tool call ID"));
                }
                let tool_index = self.tools.len();
                self.tools.insert(index(event, "output_index")?, tool_index);
                json!({"tool_calls":[{"index":tool_index,"id":id,"type":"function","function":{"name":text(item,"name")?,"arguments":text(item,"arguments")?}}]})
            }
            "response.function_call_arguments.delta" => {
                let tool_index =
                    self.tools
                        .get(&index(event, "output_index")?)
                        .ok_or_else(|| {
                            Error::upstream("tool argument delta refers to an unknown call")
                        })?;
                json!({"tool_calls":[{"index":tool_index,"function":{"arguments":text(event,"delta")?}}]})
            }
            "response.content_part.added" => match event["part"]["type"].as_str() {
                Some("output_text") => json!({"content":text(&event["part"],"text")?}),
                Some("refusal") => json!({"refusal":text(&event["part"],"refusal")?}),
                _ => return Err(Error::upstream("unsupported Chat output content")),
            },
            "response.output_text.delta" => json!({"content":text(event,"delta")?}),
            "response.refusal.delta" => json!({"refusal":text(event,"delta")?}),
            "response.output_item.added"
                if !matches!(
                    event["item"]["type"].as_str(),
                    Some("reasoning" | "message")
                ) =>
            {
                return Err(Error::upstream(
                    "unsupported Chat output item; use native Responses for hosted tools",
                ));
            }
            _ => return Ok(Vec::new()),
        };
        Ok(vec![self.chunk(delta, Value::Null)?])
    }

    pub fn finish(&self, response: &Value, include_usage: bool) -> Result<Vec<Value>> {
        let message = &response["choices"][0]["message"];
        let mut delta = json!({"reasoning_details":message["reasoning_details"]});
        if let Some(annotations) = message.get("annotations") {
            delta["annotations"] = annotations.clone();
        }
        let mut chunks = vec![self.chunk(delta, response["choices"][0]["finish_reason"].clone())?];
        if include_usage {
            let mut usage = self.chunk(json!({}), Value::Null)?;
            usage["choices"] = json!([]);
            usage["usage"] = response["usage"].clone();
            chunks.push(usage);
        }
        Ok(chunks)
    }
}

fn fragment(event: &Value, kind: &str) -> Result<(usize, &'static str)> {
    if kind.starts_with("response.function_call_arguments.") {
        Ok((0, "arguments"))
    } else {
        Ok((
            index(event, "content_index")?,
            if kind.starts_with("response.refusal.") {
                "refusal"
            } else {
                "text"
            },
        ))
    }
}

fn text<'a>(value: &'a Value, key: &str) -> Result<&'a str> {
    value[key]
        .as_str()
        .ok_or_else(|| Error::upstream(format!("upstream omitted {key}")))
}

fn index(value: &Value, key: &str) -> Result<usize> {
    value[key]
        .as_u64()
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| Error::upstream(format!("invalid upstream {key}")))
}
