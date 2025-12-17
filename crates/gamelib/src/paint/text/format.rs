use crate::paint::Srgba8;
use crate::paint::text::Font;

/// Default font size.
const DEFAULT_SIZE: u32 = 16;

/// Represents a text format property.
#[derive(Clone)]
pub enum FormatProperty {
    /// Which font to use.
    Font(Font),
    /// Size of the font in pixels.
    FontSize(f32),
    /// Text color.
    Color(Srgba8),
    Underline,
    //FontStyle(FontStyle),
    //FontStretch(FontStretch),
}

/// Fully-specified text format.
#[derive(Clone, Debug)]
pub struct TextFormat {
    pub font: Font,
    pub size: f32,
    pub color: Srgba8,
}

impl Default for TextFormat {
    fn default() -> Self {
        Self {
            font: Font::default_regular().clone(),
            size: DEFAULT_SIZE as f32,
            color: Srgba8::BLACK,
        }
    }
}
