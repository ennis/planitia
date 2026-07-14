//! Parses the metadata header at the start of shader files.
//!
//! Metadata headers are comment lines of the form `// key: value` at the start of the file.
//! Parsing of metadata headers stops at the first non-comment line. Empty lines are ignored.
//!
//! # Example
//! ```
//! // An example shader file with metadata header
//! //
//! // Manifest: shaders.toml
//! // Name: my_shader
//! // Description: This is an example shader.
//! //
//! ```

use std::collections::BTreeMap;

pub(crate) fn parse_metadata_header(source: &str) -> BTreeMap<String, String> {
    let mut metadata = BTreeMap::new();

    for line in source.lines() {
        if line.is_empty() {
            // skip empty lines
            continue;
        }

        let Some(comment) = line.strip_prefix("//") else {
            // stop parsing when we reach a non-comment line
            break;
        };

        let comment = comment.trim();
        let Some((key, value)) = comment.split_once(':') else {
            // skip lines that don't have a key-value pair
            continue;
        };

        metadata.insert(key.trim().to_string(), value.trim().to_string());
    }

    metadata
}
