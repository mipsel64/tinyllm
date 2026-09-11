use super::{protocol, reasoning};
use crate::{Result, error::Error, models::anthropic::*};
use bytes::Bytes;
use eventsource_stream::Eventsource;
use futures::{Stream, StreamExt};
use serde_json::{Value, json};
use std::collections::{HashSet, VecDeque};

pub fn decode<S, E>(upstream: S, limit: usize) -> impl Stream<Item = Result<Value>> + Send
where
    S: Stream<Item = std::result::Result<Bytes, E>> + Send + 'static,
    E: std::fmt::Display + Send + 'static,
{
    let bounded = async_stream::try_stream! {
        futures::pin_mut!(upstream);
        let mut bytes = 0usize;
        while let Some(chunk) = upstream.next().await {
            let chunk = chunk.map_err(|_| std::io::Error::other("upstream connection failed"))?;
            bytes = bytes.checked_add(chunk.len()).filter(|n| *n <= limit).ok_or_else(|| std::io::Error::other("max_response_bytes exceeded"))?;
            yield chunk;
        }
    };
    let typed: std::pin::Pin<
        Box<dyn Stream<Item = std::result::Result<Bytes, std::io::Error>> + Send + '_>,
    > = Box::pin(bounded);
    async_stream::stream! {
        let events = typed.eventsource();
        futures::pin_mut!(events);
        while let Some(event) = events.next().await {
            match event {
                Ok(event) if event.data.is_empty() => continue,
                Ok(event) => {
                    let value = serde_json::from_str::<Value>(&event.data).map_err(|_| Error::upstream("invalid upstream SSE JSON or premature [DONE]"));
                    match value {
                        Ok(value) if event.event.is_empty() || event.event == "message" || value["type"] == event.event => {
                            if !matches!(value["type"].as_str(), Some("ping" | "keepalive")) {
                                yield Ok(value);
                            }
                        }
                        Ok(_) => {yield Err(Error::upstream("SSE event name does not match its type")); break;}
                        Err(error) => {yield Err(error); break;}
                    }
                }
                Err(_) => {yield Err(Error::upstream("upstream SSE failed, was malformed, or exceeded max_response_bytes")); break;}
            }
        }
    }
}

struct Block {
    start: ContentBlockStart,
    text: String,
    deltas: VecDeque<Delta>,
    annotations: Vec<Value>,
    annotations_done: bool,
    done: bool,
    index: Option<usize>,
}

impl Block {
    fn new(start: ContentBlockStart) -> Self {
        Self {
            start,
            text: String::new(),
            deltas: VecDeque::new(),
            annotations: Vec::new(),
            annotations_done: false,
            done: false,
            index: None,
        }
    }
    fn append(&mut self, text: &str) -> Result<()> {
        if self.done {
            return Err(Error::upstream("delta after content block completion"));
        }
        self.text.push_str(text);
        self.deltas.push_back(match self.start {
            ContentBlockStart::ToolUse { .. } => Delta::InputJson {
                partial_json: text.into(),
            },
            _ => Delta::Text { text: text.into() },
        });
        Ok(())
    }
    fn observe_annotations(&mut self, part: &Value, done: bool) -> Result<()> {
        let annotations = protocol::annotations(part)?;
        if !annotations.starts_with(&self.annotations)
            || (self.annotations_done && annotations != self.annotations)
        {
            return Err(Error::upstream(
                "completed annotations differ from streamed annotations",
            ));
        }
        let added = &annotations[self.annotations.len()..];
        for annotation in added {
            protocol::citation_url(annotation)?;
        }
        self.annotations.extend_from_slice(added);
        self.annotations_done = done;
        Ok(())
    }
    fn finish(&mut self, expected: &str) -> Result<()> {
        if self.text != expected {
            return Err(Error::upstream(
                "completed content differs from streamed deltas",
            ));
        }
        if let ContentBlockStart::ToolUse { name, .. } = &self.start {
            if !serde_json::from_str::<Value>(expected).is_ok_and(|v| v.is_object()) {
                return Err(Error::upstream(
                    "upstream returned invalid tool argument JSON",
                ));
            }
            // Arguments are buffered until here, so a repair still reaches the
            // client as the only version of the call it ever sees. `text` stays
            // as streamed: upstream repeats these arguments on item completion
            // and they must still match.
            if !self.done
                && let Some(repaired) = super::tool_args::sanitize(name, expected)
            {
                self.deltas.clear();
                self.deltas.push_back(Delta::InputJson {
                    partial_json: repaired,
                });
            }
        }
        self.done = true;
        Ok(())
    }

    /// Tool arguments are only actionable once complete, and holding them lets
    /// them be repaired before the client sees a call it would reject.
    fn buffered(&self) -> bool {
        !self.done && matches!(self.start, ContentBlockStart::ToolUse { .. })
    }
}

struct Item {
    id: String,
    kind: String,
    blocks: Vec<Block>,
    current: usize,
    complete: Option<Value>,
}

pub struct Translator {
    alias: String,
    id: Option<String>,
    items: Vec<Item>,
    call_ids: HashSet<String>,
    current: usize,
    next_index: usize,
    pub completed: Option<Value>,
    pub sparse_completion: bool,
}

impl Translator {
    pub fn new(alias: String) -> Self {
        Self {
            alias,
            id: None,
            items: Vec::new(),
            call_ids: HashSet::new(),
            current: 0,
            next_index: 0,
            completed: None,
            sparse_completion: false,
        }
    }

    pub fn accept(&mut self, event: &Value) -> Result<Vec<StreamEvent>> {
        if self.completed.is_some() {
            return Err(Error::upstream("event after response completion"));
        }
        let kind = text(event, "type")?;
        let mut events = Vec::new();
        if self.sparse_completion && matches!(kind, "response.metadata" | "codex.response.metadata")
        {
            return Ok(events);
        }
        if kind == "response.created" {
            if self.id.is_some() {
                return Err(Error::upstream("duplicate response.created"));
            }
            let id = text(&event["response"], "id")?.to_owned();
            self.id = Some(id.clone());
            events.push(StreamEvent::MessageStart {
                message: MessageStartData {
                    id,
                    message_type: "message".into(),
                    role: "assistant".into(),
                    model: self.alias.clone(),
                    content: Vec::new(),
                    stop_reason: None,
                    stop_sequence: None,
                    usage: Usage {
                        input_tokens: 0,
                        output_tokens: 0,
                        cache_read_input_tokens: 0,
                        cache_creation_input_tokens: 0,
                    },
                },
            });
            return Ok(events);
        }
        if self.id.is_none() {
            return Err(Error::upstream("SSE response.created is missing"));
        }
        match kind {
            "response.in_progress" | "ping" => {}
            "response.output_item.added" => {
                let index = number(event, "output_index")?;
                if index != self.items.len() {
                    return Err(Error::upstream("out-of-order or duplicate output item"));
                }
                let item = &event["item"];
                let mut state = Item {
                    id: text(item, "id")?.into(),
                    kind: text(item, "type")?.into(),
                    blocks: Vec::new(),
                    current: 0,
                    complete: None,
                };
                match state.kind.as_str() {
                    "function_call" => {
                        let call_id = text(item, "call_id")?;
                        if call_id.is_empty() || !self.call_ids.insert(call_id.into()) {
                            return Err(Error::upstream(
                                "empty or duplicate upstream tool call ID",
                            ));
                        }
                        let mut block = Block::new(ContentBlockStart::ToolUse {
                            id: text(item, "call_id")?.into(),
                            name: text(item, "name")?.into(),
                            input: json!({}),
                        });
                        let arguments = text(item, "arguments")?;
                        if !arguments.is_empty() {
                            block.append(arguments)?;
                        }
                        state.blocks.push(block);
                    }
                    "message" => {
                        if item["content"].as_array().is_none_or(|c| !c.is_empty()) {
                            return Err(Error::upstream(
                                "streamed message must start with empty content",
                            ));
                        }
                    }
                    "reasoning" | "web_search_call" => {}
                    _ => return Err(Error::upstream("unsupported upstream output item")),
                }
                self.items.push(state);
            }
            "response.content_part.added" => {
                let item = self.item(event)?;
                if item.kind != "message" || number(event, "content_index")? != item.blocks.len() {
                    return Err(Error::upstream("out-of-order content part"));
                }
                let part = &event["part"];
                let part_kind = text(part, "type")?;
                if !matches!(part_kind, "output_text" | "refusal") {
                    return Err(Error::upstream("unsupported upstream content part"));
                }
                let mut block = Block::new(ContentBlockStart::Text {
                    text: String::new(),
                });
                let initial = text(
                    part,
                    if part_kind == "refusal" {
                        "refusal"
                    } else {
                        "text"
                    },
                )?;
                if !initial.is_empty() {
                    block.append(initial)?;
                }
                block.observe_annotations(part, false)?;
                item.blocks.push(block);
            }
            "response.web_search_call.in_progress"
            | "response.web_search_call.searching"
            | "response.web_search_call.completed" => {
                let item = self.item(event)?;
                if item.kind != "web_search_call" || item.complete.is_some() {
                    return Err(Error::upstream(
                        "web search event refers to a non-search or completed item",
                    ));
                }
            }
            "response.output_text.annotation.added" => {
                let item = self.item(event)?;
                if item.kind != "message" || item.complete.is_some() {
                    return Err(Error::upstream(
                        "annotation event refers to a non-message or completed item",
                    ));
                }
                let block = item
                    .blocks
                    .get_mut(number(event, "content_index")?)
                    .ok_or_else(|| {
                        Error::upstream("annotation refers to an unknown content block")
                    })?;
                if block.annotations_done
                    || number(event, "annotation_index")? != block.annotations.len()
                {
                    return Err(Error::upstream("out-of-order or late output annotation"));
                }
                protocol::citation_url(&event["annotation"])?;
                block.annotations.push(event["annotation"].clone());
            }
            "response.output_text.delta"
            | "response.refusal.delta"
            | "response.function_call_arguments.delta" => {
                let block_index = if kind == "response.function_call_arguments.delta" {
                    0
                } else {
                    number(event, "content_index")?
                };
                self.block(event, block_index)?
                    .append(text(event, "delta")?)?;
            }
            "response.output_text.done"
            | "response.refusal.done"
            | "response.function_call_arguments.done" => {
                let (index, key) = if kind == "response.function_call_arguments.done" {
                    (0, "arguments")
                } else {
                    (
                        number(event, "content_index")?,
                        if kind == "response.refusal.done" {
                            "refusal"
                        } else {
                            "text"
                        },
                    )
                };
                self.block(event, index)?.finish(text(event, key)?)?;
            }
            "response.content_part.done" => {
                let part = &event["part"];
                let key = if part["type"] == "refusal" {
                    "refusal"
                } else {
                    "text"
                };
                let block = self.block(event, number(event, "content_index")?)?;
                block.observe_annotations(part, true)?;
                block.finish(text(part, key)?)?;
            }
            "response.output_item.done" => {
                let item = self.item(event)?;
                let final_item = &event["item"];
                if final_item["id"] != item.id
                    || final_item["type"] != item.kind
                    || item.complete.is_some()
                {
                    return Err(Error::upstream("completed item identity changed"));
                }
                match item.kind.as_str() {
                    "function_call" => {
                        if let ContentBlockStart::ToolUse { id, name, .. } = &item.blocks[0].start
                            && (final_item["call_id"] != *id || final_item["name"] != *name)
                        {
                            return Err(Error::upstream("completed tool identity changed"));
                        }
                        item.blocks[0].finish(text(final_item, "arguments")?)?;
                    }
                    "message" => {
                        let parts = final_item["content"]
                            .as_array()
                            .ok_or_else(|| Error::upstream("completed message has no content"))?;
                        if parts.len() != item.blocks.len() {
                            return Err(Error::upstream("completed message content changed"));
                        }
                        for (block, part) in item.blocks.iter_mut().zip(parts) {
                            block.observe_annotations(part, true)?;
                            block.finish(text(
                                part,
                                if part["type"] == "refusal" {
                                    "refusal"
                                } else {
                                    "text"
                                },
                            )?)?;
                        }
                    }
                    "web_search_call" => protocol::web_search_output(final_item)?,
                    "reasoning" => {
                        if let Some(data) = reasoning::capture(final_item)
                            .as_ref()
                            .and_then(reasoning::encode)
                        {
                            let mut block =
                                Block::new(ContentBlockStart::RedactedThinking { data });
                            block.done = true;
                            item.blocks.push(block);
                        }
                    }
                    _ => {}
                }
                item.complete = Some(final_item.clone());
            }
            "response.completed" | "response.incomplete" => {
                let mut response = event["response"].clone();
                if response["id"].as_str() != self.id.as_deref() {
                    return Err(Error::upstream("response identity changed"));
                }
                if self.sparse_completion
                    && (response["output"].is_null()
                        || response["output"].as_array().is_some_and(Vec::is_empty))
                {
                    let output: Option<Vec<_>> = self
                        .items
                        .iter()
                        .map(|item| item.complete.clone())
                        .collect();
                    response["output"] = json!(output.ok_or_else(|| Error::upstream(
                        "subscription completion has unfinished output items"
                    ))?);
                }
                if self.sparse_completion && response["status"].is_null() {
                    response["status"] = json!(kind.strip_prefix("response.").unwrap());
                }
                let expected = response["output"]
                    .as_array()
                    .ok_or_else(|| Error::upstream("completed response has no output"))?;
                if expected.len() != self.items.len()
                    || self
                        .items
                        .iter()
                        .zip(expected)
                        .any(|(item, value)| item.complete.as_ref() != Some(value))
                {
                    return Err(Error::upstream(
                        "terminal output differs from streamed items",
                    ));
                }
                self.completed = Some(response);
            }
            "response.failed" => return Err(Error::openai(&event["response"])),
            "error" => return Err(Error::openai(event)),
            name if name.starts_with("response.reasoning_summary_")
                || name.starts_with("response.reasoning_text.") => {}
            _ => {
                return Err(Error::upstream(format!(
                    "unsupported upstream SSE event: {kind}"
                )));
            }
        }
        self.drain(&mut events);
        if let Some(response) = &self.completed {
            let sources = protocol::citation_sources(response)?;
            if !sources.is_empty() {
                let index = self.next_index;
                self.next_index += 1;
                events.extend([
                    StreamEvent::ContentBlockStart {
                        index,
                        content_block: ContentBlockStart::Text {
                            text: String::new(),
                        },
                    },
                    StreamEvent::ContentBlockDelta {
                        index,
                        delta: Delta::Text { text: sources },
                    },
                    StreamEvent::ContentBlockStop { index },
                ]);
            }
        }
        Ok(events)
    }

    fn item(&mut self, event: &Value) -> Result<&mut Item> {
        let item = self
            .items
            .get_mut(number(event, "output_index")?)
            .ok_or_else(|| Error::upstream("event refers to an unknown output item"))?;
        if event.get("item_id").is_some_and(|id| *id != item.id) {
            return Err(Error::upstream("event item_id does not match output_index"));
        }
        Ok(item)
    }

    fn block(&mut self, event: &Value, index: usize) -> Result<&mut Block> {
        self.item(event)?
            .blocks
            .get_mut(index)
            .ok_or_else(|| Error::upstream("event refers to an unknown content block"))
    }

    fn drain(&mut self, events: &mut Vec<StreamEvent>) {
        while let Some(item) = self.items.get_mut(self.current) {
            while let Some(block) = item.blocks.get_mut(item.current) {
                let index = *block.index.get_or_insert_with(|| {
                    let index = self.next_index;
                    self.next_index += 1;
                    events.push(StreamEvent::ContentBlockStart {
                        index,
                        content_block: block.start.clone(),
                    });
                    index
                });
                if block.buffered() {
                    return;
                }
                events.extend(
                    block
                        .deltas
                        .drain(..)
                        .map(|delta| StreamEvent::ContentBlockDelta { index, delta }),
                );
                if !block.done {
                    return;
                }
                events.push(StreamEvent::ContentBlockStop { index });
                item.current += 1;
            }
            if item.complete.is_none() {
                return;
            }
            self.current += 1;
        }
    }
}

fn text<'a>(value: &'a Value, key: &str) -> Result<&'a str> {
    value[key]
        .as_str()
        .ok_or_else(|| Error::upstream(format!("upstream event omitted {key}")))
}
fn number(value: &Value, key: &str) -> Result<usize> {
    value[key]
        .as_u64()
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| Error::upstream(format!("invalid upstream {key}")))
}

pub fn finish(response: &AnthropicResponse) -> [StreamEvent; 2] {
    [
        StreamEvent::MessageDelta {
            delta: MessageDeltaData {
                stop_reason: response.stop_reason.clone(),
                stop_sequence: response.stop_sequence.clone(),
            },
            usage: DeltaUsage {
                input_tokens: Some(response.usage.input_tokens),
                output_tokens: response.usage.output_tokens,
                cache_read_input_tokens: response.usage.cache_read_input_tokens,
                cache_creation_input_tokens: response.usage.cache_creation_input_tokens,
            },
        },
        StreamEvent::MessageStop,
    ]
}
