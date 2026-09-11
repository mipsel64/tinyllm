use crate::{Result, error::Error, models::anthropic::*};
use serde_json::Value;
use std::collections::VecDeque;

pub(super) struct StopFilter {
    sequences: Vec<String>,
    pending: VecDeque<StreamEvent>,
    text: String,
    open: Option<(usize, usize)>,
    matched: Option<(String, usize, usize)>,
}

impl StopFilter {
    pub fn new(value: &Value) -> Result<Self> {
        let sequences = if value.is_null() {
            Vec::new()
        } else {
            let values = value.as_array().filter(|v| v.len() <= 16).ok_or_else(|| {
                Error::invalid("stop_sequences must be an array of at most 16 strings")
            })?;
            values
                .iter()
                .map(|value| {
                    value
                        .as_str()
                        .filter(|s| !s.is_empty() && s.len() <= 1024)
                        .map(str::to_owned)
                        .ok_or_else(|| {
                            Error::invalid("each stop sequence must contain 1–1024 UTF-8 bytes")
                        })
                })
                .collect::<Result<Vec<_>>>()?
        };
        Ok(Self {
            sequences,
            pending: VecDeque::new(),
            text: String::new(),
            open: None,
            matched: None,
        })
    }

    pub fn enabled(&self) -> bool {
        !self.sequences.is_empty()
    }

    pub fn push(&mut self, event: StreamEvent) -> Vec<StreamEvent> {
        if !self.enabled() {
            return vec![event];
        }
        if self.matched.is_some() {
            return Vec::new();
        }
        match &event {
            StreamEvent::ContentBlockDelta {
                delta: Delta::Text { text },
                ..
            } => {
                self.text.push_str(text);
                self.pending.push_back(event);
                let found = self
                    .sequences
                    .iter()
                    .enumerate()
                    .filter_map(|(order, sequence)| {
                        self.text
                            .find(sequence)
                            .map(|start| (start + sequence.len(), start, order))
                    })
                    .min();
                if let Some((_, start, order)) = found {
                    let mut events = self.flush(start);
                    let (index, offset) = self.open.expect("text delta has an open block");
                    self.matched = Some((self.sequences[order].clone(), index, offset));
                    events.push(StreamEvent::ContentBlockStop { index });
                    self.open = None;
                    self.pending.clear();
                    self.text.clear();
                    return events;
                }
                let keep = self.sequences.iter().map(String::len).max().unwrap() - 1;
                let mut safe = self.text.len().saturating_sub(keep);
                while !self.text.is_char_boundary(safe) {
                    safe -= 1;
                }
                let events = self.flush(safe);
                self.text.drain(..safe);
                events
            }
            // Carrier blocks buffer with the text so a stop sequence can still
            // span two messages separated by reasoning.
            StreamEvent::ContentBlockStart {
                content_block:
                    ContentBlockStart::Text { .. } | ContentBlockStart::RedactedThinking { .. },
                ..
            }
            | StreamEvent::ContentBlockStop { .. } => {
                self.pending.push_back(event);
                self.flush(0)
            }
            _ => {
                let mut events = self.finish();
                self.emit(event, &mut events);
                events
            }
        }
    }

    pub fn finish(&mut self) -> Vec<StreamEvent> {
        let events = self.flush(self.text.len());
        self.text.clear();
        events
    }

    fn emit(&mut self, event: StreamEvent, events: &mut Vec<StreamEvent>) {
        match &event {
            StreamEvent::ContentBlockStart { index, .. } => self.open = Some((*index, 0)),
            StreamEvent::ContentBlockDelta {
                delta: Delta::Text { text },
                ..
            } => {
                if let Some((_, offset)) = &mut self.open {
                    *offset += text.len();
                }
            }
            StreamEvent::ContentBlockStop { .. } => self.open = None,
            _ => {}
        }
        events.push(event);
    }

    fn flush(&mut self, mut bytes: usize) -> Vec<StreamEvent> {
        let mut events = Vec::new();
        while let Some(event) = self.pending.pop_front() {
            if let StreamEvent::ContentBlockDelta {
                index,
                delta: Delta::Text { mut text },
            } = event
            {
                if text.len() > bytes {
                    let tail = text.split_off(bytes);
                    self.pending.push_front(StreamEvent::ContentBlockDelta {
                        index,
                        delta: Delta::Text { text: tail },
                    });
                    if !text.is_empty() {
                        self.emit(
                            StreamEvent::ContentBlockDelta {
                                index,
                                delta: Delta::Text { text },
                            },
                            &mut events,
                        );
                    }
                    break;
                }
                bytes -= text.len();
                self.emit(
                    StreamEvent::ContentBlockDelta {
                        index,
                        delta: Delta::Text { text },
                    },
                    &mut events,
                );
            } else {
                self.emit(event, &mut events);
            }
        }
        events
    }

    pub fn json(&mut self, response: &AnthropicResponse) {
        if !self.enabled() {
            return;
        }
        for (index, content) in response.content.iter().enumerate() {
            let start = match content {
                ResponseContent::Text { .. } => ContentBlockStart::Text {
                    text: String::new(),
                },
                ResponseContent::ToolUse {
                    id, name, input, ..
                } => ContentBlockStart::ToolUse {
                    id: id.clone(),
                    name: name.clone(),
                    input: input.clone(),
                },
                ResponseContent::RedactedThinking { data, .. } => {
                    ContentBlockStart::RedactedThinking { data: data.clone() }
                }
            };
            self.push(StreamEvent::ContentBlockStart {
                index,
                content_block: start,
            });
            if let ResponseContent::Text { text, .. } = content {
                self.push(StreamEvent::ContentBlockDelta {
                    index,
                    delta: Delta::Text { text: text.clone() },
                });
            }
            self.push(StreamEvent::ContentBlockStop { index });
        }
        self.finish();
    }

    pub fn apply(&self, native: &mut Value, response: &mut AnthropicResponse) -> Result<()> {
        let Some((sequence, index, offset)) = &self.matched else {
            return Ok(());
        };
        let Some(ResponseContent::Text { text, .. }) = response.content.get_mut(*index) else {
            return Err(Error::upstream(
                "stop sequence position does not match completed output",
            ));
        };
        if !text.is_char_boundary(*offset) {
            return Err(Error::upstream("invalid stop sequence text offset"));
        }
        text.truncate(*offset);
        response.content.truncate(index + 1);
        if response.stop_reason.as_deref() != Some("refusal") {
            response.stop_reason = Some("stop_sequence".into());
            response.stop_sequence = Some(sequence.clone());
        }
        let output = native["output"]
            .as_array_mut()
            .ok_or_else(|| Error::upstream("upstream output missing"))?;
        let mut block_index = 0;
        for (item_index, item) in output.iter_mut().enumerate() {
            match item["type"].as_str() {
                // Mirrors the carrier blocks protocol::response emits.
                Some("reasoning")
                    if super::reasoning::capture(item)
                        .as_ref()
                        .and_then(super::reasoning::encode)
                        .is_some() =>
                {
                    block_index += 1;
                }
                Some("message") => {
                    let parts = item["content"]
                        .as_array_mut()
                        .ok_or_else(|| Error::upstream("upstream message content missing"))?;
                    for (part_index, part) in parts.iter_mut().enumerate() {
                        if block_index == *index {
                            let key = if part["type"] == "refusal" {
                                "refusal"
                            } else {
                                "text"
                            };
                            let text = part[key]
                                .as_str()
                                .filter(|s| s.is_char_boundary(*offset))
                                .ok_or_else(|| {
                                Error::upstream("invalid stop sequence native offset")
                            })?;
                            part[key] = Value::String(text[..*offset].into());
                            parts.truncate(part_index + 1);
                            // The shortened message is client-owned, not the original OpenAI item.
                            item.as_object_mut().unwrap().remove("id");
                            output.truncate(item_index + 1);
                            return Ok(());
                        }
                        block_index += 1;
                    }
                }
                Some("function_call") => block_index += 1,
                _ => {}
            }
        }
        Err(Error::upstream(
            "stop sequence position is missing from native output",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::openai::{protocol, reasoning, stream};
    use serde_json::json;

    fn response(parts: &[&str]) -> Value {
        json!({"id":"resp_test","status":"completed","usage":{"input_tokens":20,"output_tokens":30},"output":[
            {"type":"reasoning","id":"rs_before","summary":[],"encrypted_content":"opaque-before"},
            {"type":"message","id":"m","role":"assistant","content":parts.iter().map(|text|json!({"type":"output_text","text":text,"annotations":[]})).collect::<Vec<_>>()},
            {"type":"reasoning","id":"rs_after","summary":[],"encrypted_content":"opaque-after"}
        ]})
    }

    fn visible(events: &[StreamEvent]) -> Value {
        let mut content = Vec::<Value>::new();
        let mut open = None;
        for event in events {
            match event {
                StreamEvent::ContentBlockStart {
                    index,
                    content_block,
                } => {
                    assert_eq!(*index, content.len());
                    assert!(open.replace(*index).is_none());
                    content.push(serde_json::to_value(content_block).unwrap());
                }
                StreamEvent::ContentBlockDelta {
                    index,
                    delta: Delta::Text { text },
                } => {
                    assert_eq!(open, Some(*index));
                    let value = content[*index]["text"].as_str().unwrap().to_owned() + text;
                    content[*index]["text"] = json!(value);
                }
                StreamEvent::ContentBlockStop { index } => assert_eq!(open.take(), Some(*index)),
                _ => {}
            }
        }
        assert!(open.is_none());
        json!(content)
    }

    #[test]
    fn stops_preserve_content_across_unicode_fragments_and_blocks() {
        for (parts, sequences, expected, matched) in [
            (
                vec!["hé</bl", "ock>hidden"],
                json!(["</block>"]),
                vec!["hé"],
                json!("</block>"),
            ),
            (
                vec!["</block>hidden"],
                json!(["</block>"]),
                vec![""],
                json!("</block>"),
            ),
            (
                vec!["hé🙂ENDhidden"],
                json!(["🙂END"]),
                vec!["hé"],
                json!("🙂END"),
            ),
            (vec!["abc"], json!(["abc", "b"]), vec!["a"], json!("b")),
            (vec!["abc"], json!(["bc", "abc"]), vec![""], json!("abc")),
            (
                vec!["hé</blo", "z"],
                json!(["</block>"]),
                vec!["hé</blo", "z"],
                Value::Null,
            ),
            (vec!["hello"], json!([]), vec!["hello"], Value::Null),
        ] {
            for fragmented in [false, true] {
                let mut native = response(&parts);
                let mut converted = protocol::response(&native, "openai/gpt-test").unwrap();
                let mut filter = StopFilter::new(&sequences).unwrap();
                // The stream carries the same reasoning blocks the JSON response does.
                let carriers: Vec<String> = native["output"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .filter_map(|item| {
                        reasoning::capture(item)
                            .as_ref()
                            .and_then(reasoning::encode)
                    })
                    .collect();
                let mut events = filter.push(StreamEvent::ContentBlockStart {
                    index: 0,
                    content_block: ContentBlockStart::RedactedThinking {
                        data: carriers[0].clone(),
                    },
                });
                events.extend(filter.push(StreamEvent::ContentBlockStop { index: 0 }));
                for (index, part) in parts.iter().enumerate() {
                    let index = index + 1;
                    events.extend(filter.push(StreamEvent::ContentBlockStart {
                        index,
                        content_block: ContentBlockStart::Text {
                            text: String::new(),
                        },
                    }));
                    let chunks = if fragmented {
                        part.chars().map(|c| c.to_string()).collect()
                    } else {
                        vec![part.to_string()]
                    };
                    for text in chunks {
                        events.extend(filter.push(StreamEvent::ContentBlockDelta {
                            index,
                            delta: Delta::Text { text },
                        }));
                    }
                    events.extend(filter.push(StreamEvent::ContentBlockStop { index }));
                }
                let trailing = parts.len() + 1;
                events.extend(filter.push(StreamEvent::ContentBlockStart {
                    index: trailing,
                    content_block: ContentBlockStart::RedactedThinking {
                        data: carriers[1].clone(),
                    },
                }));
                events.extend(filter.push(StreamEvent::ContentBlockStop { index: trailing }));
                events.extend(filter.finish());
                filter.apply(&mut native, &mut converted).unwrap();
                assert_eq!(
                    converted
                        .content
                        .iter()
                        .filter_map(|c| if let ResponseContent::Text { text, .. } = c {
                            Some(text.as_str())
                        } else {
                            None
                        })
                        .collect::<Vec<_>>(),
                    expected
                );
                assert_eq!(json!(converted.stop_sequence), matched);
                assert_eq!(visible(&events), json!(converted.content));
                let terminal = serde_json::to_value(stream::finish(&converted)[0].clone()).unwrap();
                assert_eq!(terminal["delta"]["stop_sequence"], matched);
                assert_eq!(converted.usage.output_tokens, 30);
                let mut json_native = response(&parts);
                let mut json_response =
                    protocol::response(&json_native, "openai/gpt-test").unwrap();
                let mut json_filter = StopFilter::new(&sequences).unwrap();
                json_filter.json(&json_response);
                json_filter
                    .apply(&mut json_native, &mut json_response)
                    .unwrap();
                assert_eq!(json!(converted), json!(json_response));
                assert_eq!(native, json_native);
                if !matched.is_null() {
                    assert_eq!(native["output"].as_array().unwrap().len(), 2);
                    assert_eq!(native["output"][0]["encrypted_content"], "opaque-before");
                    assert!(native["output"][1].get("id").is_none());
                }
            }
        }
    }

    #[test]
    fn cutoff_discards_reasoning_between_native_messages() {
        let mut native = response(&["visible</bl"]);
        native["output"].as_array_mut().unwrap().push(json!({
            "type":"message","id":"n","role":"assistant","content":[{"type":"output_text","text":"ock>hidden","annotations":[]}]
        }));
        let mut converted = protocol::response(&native, "openai/gpt-test").unwrap();
        let mut filter = StopFilter::new(&json!(["</block>"])).unwrap();
        filter.json(&converted);
        filter.apply(&mut native, &mut converted).unwrap();
        assert_eq!(native["output"].as_array().unwrap().len(), 2);
        assert_eq!(native["output"][0]["encrypted_content"], "opaque-before");
        assert_eq!(native["output"][1]["content"][0]["text"], "visible");
        assert_eq!(converted.content.len(), 2);
        assert_eq!(converted.stop_sequence.as_deref(), Some("</block>"));
    }

    #[test]
    fn tool_arguments_do_not_match_or_join_text_stop_sequences() {
        let mut native = response(&["</bl"]);
        native["output"].as_array_mut().unwrap().extend([
            json!({"type":"function_call","id":"f","call_id":"call","name":"lookup","arguments":"{\"text\":\"</block>\"}"}),
            json!({"type":"message","id":"n","role":"assistant","content":[{"type":"output_text","text":"ock>","annotations":[]}]})
        ]);
        let expected = native.clone();
        let mut converted = protocol::response(&native, "openai/gpt-test").unwrap();
        let mut filter = StopFilter::new(&json!(["</block>"])).unwrap();
        filter.json(&converted);
        filter.apply(&mut native, &mut converted).unwrap();
        assert_eq!(native, expected);
        assert_eq!(converted.stop_reason.as_deref(), Some("tool_use"));
        assert!(converted.stop_sequence.is_none());
    }

    #[test]
    fn stop_filter_preserves_refusal_and_validates_limits() {
        let mut native = response(&["denied</block>"]);
        native["status"] = json!("incomplete");
        native["incomplete_details"] = json!({"reason":"content_filter"});
        let mut converted = protocol::response(&native, "openai/gpt-test").unwrap();
        let mut filter = StopFilter::new(&json!(["</block>"])).unwrap();
        filter.json(&converted);
        filter.apply(&mut native, &mut converted).unwrap();
        assert_eq!(converted.stop_reason.as_deref(), Some("refusal"));
        assert!(converted.stop_sequence.is_none());
        for invalid in [
            json!("STOP"),
            json!([1]),
            json!([""]),
            json!(["x".repeat(1025)]),
            json!(vec!["x"; 17]),
        ] {
            assert!(StopFilter::new(&invalid).is_err(), "{invalid}");
        }
    }
}
