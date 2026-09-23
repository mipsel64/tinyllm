//! Claude Code billing header for Anthropic subscription OAuth.
//!
//! Without an `x-anthropic-billing-header` system block, Anthropic classifies
//! an OAuth request as third-party app usage and rejects it with a 400
//! disguised as "You're out of extra usage." The recipe must match what
//! Anthropic's backend expects (ported from @gotgenes/pi-anthropic-auth).

use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const SALT: &str = "59cf53e54c78";
const POSITIONS: [usize; 3] = [4, 7, 20];
const ENTRYPOINT: &str = "sdk-cli";
const MARKER: &str = "x-anthropic-billing-header:";

/// Floor only: a claude-cli user-agent version wins when the client sends one.
pub(crate) const FALLBACK_CLAUDE_CODE_VERSION: &str = "2.1.280";

pub(crate) fn is_bare_version(version: &str) -> bool {
    let mut parts = version.split('.');
    (0..3).all(|_| {
        parts
            .next()
            .is_some_and(|part| !part.is_empty() && part.bytes().all(|byte| byte.is_ascii_digit()))
    }) && parts.next().is_none()
}

/// Reads the `claude-cli/X.Y.Z` token out of a user-agent. The whole-token
/// match rejects `notclaude-cli/...` and versions with extra components.
pub(crate) fn claude_cli_version(user_agent: Option<&str>) -> Option<&str> {
    let version = user_agent?
        .split_whitespace()
        .find_map(|token| token.strip_prefix("claude-cli/"))?;
    is_bare_version(version).then_some(version)
}

/// Prepends the billing header as the first `system` block, so Anthropic bills
/// the request as the client's Claude Code instead of third-party usage.
/// No user text to hash means no header to build, matching Claude Code.
pub(crate) fn prepend(value: &mut Value, version: &str) {
    let Some(text) = first_user_text(value) else {
        return;
    };
    let header = build(text, version);
    let block = json!({"type": "text", "text": header});
    match value.get_mut("system") {
        None => {
            value["system"] = json!([block]);
        }
        Some(system) => match &mut *system {
            Value::String(prompt) => {
                let prompt = std::mem::take(prompt);
                *system = json!([block, {"type": "text", "text": prompt}]);
            }
            Value::Array(blocks) => {
                let already_shaped = blocks.iter().any(|block| {
                    block["text"]
                        .as_str()
                        .is_some_and(|text| text.contains(MARKER))
                });
                if !already_shaped {
                    blocks.insert(0, block);
                }
            }
            // Anthropic rejects other system shapes anyway; leave them for that error.
            _ => {}
        },
    }
}

fn first_user_text(value: &Value) -> Option<&str> {
    let message = value["messages"]
        .as_array()?
        .iter()
        .find(|message| message["role"] == "user")?;
    match &message["content"] {
        Value::String(text) => Some(text.as_str()),
        Value::Array(blocks) => blocks
            .iter()
            .find(|block| block["type"] == "text")
            .and_then(|block| block["text"].as_str()),
        _ => None,
    }
}

fn build(text: &str, version: &str) -> String {
    let cch = &format!("{:x}", Sha256::digest(text.as_bytes()))[..5];
    let sampled: String = POSITIONS.iter().map(|&index| sample(text, index)).collect();
    let suffix = &format!(
        "{:x}",
        Sha256::digest(format!("{SALT}{sampled}{version}").as_bytes())
    )[..3];
    format!("{MARKER} cc_version={version}.{suffix}; cc_entrypoint={ENTRYPOINT}; cch={cch};")
}

// Claude Code samples UTF-16 code units, like a JS client; astral characters
// past the position degrade to U+FFFD instead of a surrogate half.
fn sample(text: &str, index: usize) -> String {
    text.encode_utf16()
        .nth(index)
        .map(|unit| String::from_utf16_lossy(&[unit]))
        .unwrap_or_else(|| "0".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Expected values produced by the reference JS implementation.
    #[test]
    fn billing_header_matches_the_client_recipe() {
        assert_eq!(
            build("hello", "2.1.300"),
            "x-anthropic-billing-header: cc_version=2.1.300.2d3; cc_entrypoint=sdk-cli; cch=2cf24;"
        );
        assert_eq!(
            build("hello world from claude code", "2.1.280"),
            "x-anthropic-billing-header: cc_version=2.1.280.fdb; cc_entrypoint=sdk-cli; cch=9c7fc;"
        );
        // "ab" samples "0" past its end, and non-ASCII BMP text still matches.
        assert_eq!(
            build("Vietnamese text: Hà Nội", "2.1.300"),
            "x-anthropic-billing-header: cc_version=2.1.300.e00; cc_entrypoint=sdk-cli; cch=0a782;"
        );
    }

    #[test]
    fn prepend_shapes_every_system_form_exactly_once() {
        let message = json!({"role": "user", "content": "hello"});
        let expected = build("hello", "2.1.280");

        let mut absent = json!({"messages": [message]});
        prepend(&mut absent, "2.1.280");
        assert_eq!(
            absent["system"],
            json!([{"type": "text", "text": expected}])
        );

        let mut prompt = json!({"messages": [message], "system": "be brief"});
        prepend(&mut prompt, "2.1.280");
        assert_eq!(
            prompt["system"],
            json!([{"type": "text", "text": expected}, {"type": "text", "text": "be brief"}])
        );

        let mut blocks = json!({"messages": [message.clone()], "system": [
            {"type": "text", "text": "be brief", "cache_control": {"type": "ephemeral"}}
        ]});
        prepend(&mut blocks, "2.1.280");
        assert_eq!(
            blocks["system"],
            json!([
                {"type": "text", "text": expected},
                {"type": "text", "text": "be brief", "cache_control": {"type": "ephemeral"}}
            ])
        );
        prepend(&mut blocks, "2.1.999");
        assert_eq!(blocks["system"][0]["text"], json!(expected));
    }

    #[test]
    fn prepend_skips_payloads_without_user_text() {
        let mut tool_only = json!({"messages": [
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": "ok"}]}
        ], "system": "be brief"});
        prepend(&mut tool_only, "2.1.280");
        assert_eq!(tool_only["system"], json!("be brief"));

        let mut empty = json!({"messages": []});
        prepend(&mut empty, "2.1.280");
        assert!(empty.get("system").is_none());
    }

    #[test]
    fn claude_cli_version_requires_a_bare_version_token() {
        assert_eq!(
            claude_cli_version(Some("claude-cli/2.1.300 (external, cli)")),
            Some("2.1.300")
        );
        assert_eq!(
            claude_cli_version(Some("claude-cli/2.1.300")),
            Some("2.1.300")
        );
        assert_eq!(claude_cli_version(None), None);
        assert_eq!(
            claude_cli_version(Some("notclaude-cli/2.1.300 tinyllm/0.1.0")),
            None
        );
        assert_eq!(
            claude_cli_version(Some("claude-cli/2.1 tinyllm/0.1.0")),
            None
        );
        assert_eq!(
            claude_cli_version(Some("claude-cli/2.1.280-beta tinyllm/0.1.0")),
            None
        );
        assert!(is_bare_version("2.1.280"));
        assert!(!is_bare_version("2.1.280.1"));
        assert!(!is_bare_version("2.1."));
    }
}
