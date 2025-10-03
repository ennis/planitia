use crate::paint::Srgba32;
use crate::paint::text::Font;

/// Represents a text format property.
#[derive(Clone)]
pub enum FormatProperty {
    /// Which font to use.
    Font(Font),
    /// Size of the font in pixels.
    FontSize(f32),
    /// Text color.
    Color(Srgba32),
    Underline,
    //FontStyle(FontStyle),
    //FontStretch(FontStretch),
}

/// Fully-specified text format.
#[derive(Clone, Debug)]
pub struct TextFormat {
    pub font: Font,
    pub size: f32,
    pub color: Srgba32,
}

impl Default for TextFormat {
    fn default() -> Self {
        Self {
            font: Font::default_regular().clone(),
            size: 16.0,
            color: Srgba32::BLACK,
        }
    }
}
