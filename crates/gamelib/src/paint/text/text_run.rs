use crate::paint::text::format::FormatProperty;

/// String slice with associated formatting attributes.
#[derive(Copy, Clone)]
pub struct TextRun<'a> {
    pub str: &'a str,
    pub styles: &'a [FormatProperty],
}
