//! Repairs tool arguments a model emits that the client would reject.
//!
//! Claude Code validates tool input against its own schema and fails the call
//! when it does not fit. Two shapes recur with `Read` and cost a turn each, so
//! they are dropped here rather than bounced back through the model.

use serde_json::Value;

/// Past this an offset is a hallucinated line number rather than a real one;
/// dropping it reads the file from the start instead of failing the call.
const ABSURD_OFFSET: i64 = 1_000_000;

/// Returns the repaired arguments, or None when nothing needed changing.
pub fn sanitize(name: &str, arguments: &str) -> Option<String> {
    if name != "Read" || arguments.is_empty() {
        return None;
    }
    let mut parsed: Value = serde_json::from_str(arguments).ok()?;
    let object = parsed.as_object_mut()?;

    // `pages` is meaningful only for PDFs; empty means the model filled a field
    // it should have omitted.
    let empty_pages = object
        .get("pages")
        .and_then(Value::as_str)
        .is_some_and(str::is_empty);
    let absurd_offset = object
        .get("offset")
        .and_then(Value::as_i64)
        .is_some_and(|offset| offset >= ABSURD_OFFSET);
    if !empty_pages && !absurd_offset {
        return None;
    }
    if empty_pages {
        object.remove("pages");
    }
    if absurd_offset {
        object.remove("offset");
    }
    let repaired = serde_json::to_string(&parsed).ok()?;
    tracing::debug!(
        tool = name,
        empty_pages,
        absurd_offset,
        "repaired tool arguments"
    );
    Some(repaired)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drops_empty_pages_and_absurd_offsets_keeping_the_rest() {
        let repaired = sanitize("Read", r#"{"file_path":"/tmp/a","pages":"","limit":20}"#).unwrap();
        let value: Value = serde_json::from_str(&repaired).unwrap();
        assert!(value.get("pages").is_none());
        assert_eq!(value["file_path"], "/tmp/a");
        assert_eq!(value["limit"], 20);

        let repaired = sanitize("Read", r#"{"file_path":"/tmp/a","offset":1300000}"#).unwrap();
        let value: Value = serde_json::from_str(&repaired).unwrap();
        assert!(value.get("offset").is_none());
    }

    #[test]
    fn leaves_usable_arguments_alone() {
        for arguments in [
            r#"{"file_path":"/tmp/a","offset":1300,"limit":20}"#,
            r#"{"file_path":"/tmp/a","pages":"1-5"}"#,
            r#"{"file_path":"/tmp/a"}"#,
        ] {
            assert_eq!(sanitize("Read", arguments), None, "{arguments}");
        }
    }

    #[test]
    fn touches_nothing_else() {
        // Another tool may use these names with its own meaning.
        assert_eq!(sanitize("Grep", r#"{"pages":"","offset":9999999}"#), None);
        assert_eq!(sanitize("Read", ""), None);
        assert_eq!(sanitize("Read", "not json"), None);
        assert_eq!(sanitize("Read", "[1,2]"), None);
    }
}
