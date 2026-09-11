use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD};
use serde_json::Value;

pub const PREFIX: &str = "tinyllm:v1:";
const MAX_ID_BYTES: usize = 4 * 1024;
const MAX_ENCRYPTED_BYTES: usize = 8 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Replay {
    pub id: String,
    pub encrypted_content: String,
}

impl Replay {
    /// The Responses input item that resumes this reasoning turn upstream.
    pub fn item(&self) -> Value {
        serde_json::json!({
            "type": "reasoning",
            "id": self.id,
            "summary": [],
            "encrypted_content": self.encrypted_content,
        })
    }
}

/// Reads a `reasoning` output item into a replayable carrier.
pub fn capture(item: &Value) -> Option<Replay> {
    Some(Replay {
        id: non_empty(item.get("id"))?.to_owned(),
        encrypted_content: non_empty(item.get("encrypted_content"))?.to_owned(),
    })
}

pub fn encode(replay: &Replay) -> Option<String> {
    if replay.id.len() > MAX_ID_BYTES || replay.encrypted_content.len() > MAX_ENCRYPTED_BYTES {
        return None;
    }
    let id = URL_SAFE_NO_PAD.encode(replay.id.as_bytes());
    Some(format!("{PREFIX}{id}:{}", replay.encrypted_content))
}

/// Foreign or malformed carriers decode to `None` so they replay as plain history.
pub fn decode(data: &str) -> Option<Replay> {
    let payload = data.strip_prefix(PREFIX)?;
    let (id, encrypted_content) = payload.split_once(':')?;
    if id.len() > encoded_id_limit()
        || encrypted_content.is_empty()
        || encrypted_content.len() > MAX_ENCRYPTED_BYTES
    {
        return None;
    }
    let id = URL_SAFE_NO_PAD.decode(id).ok()?;
    if id.is_empty() || id.len() > MAX_ID_BYTES {
        return None;
    }
    Some(Replay {
        id: String::from_utf8(id).ok()?,
        encrypted_content: encrypted_content.to_owned(),
    })
}

fn encoded_id_limit() -> usize {
    MAX_ID_BYTES.div_ceil(3) * 4
}

fn non_empty(value: Option<&Value>) -> Option<&str> {
    value?.as_str().filter(|value| !value.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn round_trip_preserves_identity_and_ciphertext() {
        let replay = Replay {
            id: "rs_68a".into(),
            encrypted_content: "gAAAAAop+aque/==".into(),
        };
        let carrier = encode(&replay).unwrap();
        assert!(carrier.starts_with(PREFIX));
        assert_eq!(decode(&carrier), Some(replay.clone()));
        assert_eq!(
            replay.item()["encrypted_content"],
            json!(replay.encrypted_content)
        );
    }

    #[test]
    fn foreign_and_malformed_carriers_decode_to_none() {
        for data in [
            "anthropic-signature",
            "tinyllm:v1:not-base64!:x",
            "tinyllm:v1:cnNfMQ",
            "tinyllm:v1:cnNfMQ:",
            "tinyllm:v0:cnNfMQ:x",
            "",
        ] {
            assert_eq!(decode(data), None, "{data}");
        }
        assert_eq!(
            decode(&format!(
                "{PREFIX}cnNfMQ:{}",
                "A".repeat(MAX_ENCRYPTED_BYTES + 1)
            )),
            None
        );
    }

    #[test]
    fn capture_requires_both_identity_and_ciphertext() {
        assert!(capture(&json!({"id":"rs_1","encrypted_content":"x"})).is_some());
        assert!(capture(&json!({"id":"rs_1","encrypted_content":""})).is_none());
        assert!(capture(&json!({"encrypted_content":"x"})).is_none());
    }
}
