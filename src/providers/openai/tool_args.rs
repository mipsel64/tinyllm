//! Repairs tool arguments a model emits that the client would reject.
//!
//! Claude Code validates tool input against its own schema and fails the call
//! when it does not fit, costing a turn. Only arguments that carry no meaning at
//! all are dropped: anything the schema would accept is forwarded untouched,
//! because silently changing a valid argument is worse than a visible error.

use serde_json::Value;

/// Returns the repaired arguments, or None when nothing needed changing.
pub fn sanitize(name: &str, arguments: &str) -> Option<String> {
    if name != "Read" || arguments.is_empty() {
        return None;
    }
    let mut parsed: Value = serde_json::from_str(arguments).ok()?;
    let object = parsed.as_object_mut()?;

    // `pages` selects PDF pages, so an empty string selects nothing: the model
    // filled a field it should have omitted. No page range means the same thing,
    // so removing it cannot change which pages are read.
    if !object
        .get("pages")
        .and_then(Value::as_str)
        .is_some_and(str::is_empty)
    {
        return None;
    }
    object.remove("pages");
    let repaired = serde_json::to_string(&parsed).ok()?;
    tracing::debug!(tool = name, "dropped an empty pages argument");
    Some(repaired)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drops_an_empty_pages_argument_keeping_the_rest() {
        let repaired = sanitize("Read", r#"{"file_path":"/tmp/a","pages":"","limit":20}"#).unwrap();
        let value: Value = serde_json::from_str(&repaired).unwrap();
        assert!(value.get("pages").is_none());
        assert_eq!(value["file_path"], "/tmp/a");
        assert_eq!(value["limit"], 20);
    }

    #[test]
    fn never_rewrites_an_argument_the_schema_would_accept() {
        for arguments in [
            r#"{"file_path":"/tmp/a","pages":"1-5"}"#,
            r#"{"file_path":"/tmp/a"}"#,
            // A large offset is unusual but legal, and quietly dropping it would
            // read different lines than the call asked for.
            r#"{"file_path":"/tmp/a","offset":1000000}"#,
            r#"{"file_path":"/tmp/a","offset":1300000,"limit":20}"#,
            r#"{"file_path":"/tmp/a","offset":1300,"limit":20}"#,
        ] {
            assert_eq!(sanitize("Read", arguments), None, "{arguments}");
        }
    }

    #[test]
    fn touches_nothing_else() {
        // Another tool may use this name with its own meaning.
        assert_eq!(sanitize("Grep", r#"{"pages":""}"#), None);
        assert_eq!(sanitize("Read", ""), None);
        assert_eq!(sanitize("Read", "not json"), None);
        assert_eq!(sanitize("Read", "[1,2]"), None);
    }
}
