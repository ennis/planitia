//! Styling of editor cells.
use gamelib::color::Srgba8;

/// Text underline style.
#[derive(Debug, Clone, Copy)]
pub struct Underline {
    pub color: Srgba8,
    pub thickness: f32,
}

/// Cell border style.
#[derive(Debug, Clone, Copy)]
pub struct Border {
    pub color: Srgba8,
    pub thickness: f32,
}

/// Cell style options.
#[derive(Debug, Clone, Copy)]
pub struct Style {
    pub font: u32 = 0,
    pub baseline: Option<f32> = None,
    pub text_color: Option<Srgba8> = None,
    pub background_color: Option<Srgba8> = None,
    pub underline: Option<Underline> = None,
    pub border: Option<Border> = None,
    pub punctuation_left: bool = false,
    pub punctuation_right: bool = false,
}

pub static DEFAULT_STYLE: Style = Style { .. };

impl Default for Style {
    fn default() -> Self {
        Style { .. }
    }
}