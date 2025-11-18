use crate::paint::text::GlyphId;
use ab_glyph::{Font as FontTrait, FontArc, ScaleFont};
use std::cmp::Ordering;
use std::sync::OnceLock;
use std::sync::atomic::AtomicUsize;

/// An identifier for a font.
///
/// TODO: replace this with AssetId when we have an asset system.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FontId(pub usize);

impl FontId {
    pub fn next() -> Self {
        static NEXT_FONT_ID: AtomicUsize = AtomicUsize::new(1);
        let id = NEXT_FONT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        FontId(id)
    }
}

/// Represents a font file in memory.
#[derive(Clone, Debug)]
pub struct Font {
    pub(crate) data: FontArc,
    pub(crate) id: FontId,
}

impl Font {
    pub fn ascent(&self, size: f32) -> f32 {
        let scaled = self.data.as_scaled(size);
        scaled.ascent()
    }

    pub fn descent(&self, size: f32) -> f32 {
        let scaled = self.data.as_scaled(size);
        scaled.descent()
    }

    pub fn line_gap(&self, size: f32) -> f32 {
        let scaled = self.data.as_scaled(size);
        scaled.line_gap()
    }

    pub fn glyph_id(&self, c: char) -> GlyphId {
        self.data.glyph_id(c)
    }

    pub fn h_advance(&self, id: GlyphId, size: f32) -> f32 {
        self.data.as_scaled(size).h_advance(id)
    }

    pub fn kern(&self, first: GlyphId, second: GlyphId, size: f32) -> f32 {
        self.data.as_scaled(size).kern(first, second)
    }

    /// Loads a font file.
    pub fn load_static_font_from_bytes(bytes: &'static [u8]) -> Font {
        let font = FontArc::try_from_slice(bytes).expect("failed to load font");
        Font {
            data: font,
            id: FontId::next(),
        }
    }

    /// Returns the default font for regular text.
    pub fn default_regular() -> &'static Self {
        static INTER_DISPLAY_REGULAR: &[u8] = include_bytes!("Zurich Condensed BT.ttf");
        static FONT: OnceLock<Font> = OnceLock::new();
        FONT.get_or_init(|| Font::load_static_font_from_bytes(INTER_DISPLAY_REGULAR))
    }

    /// Returns the default font for semibold text.
    pub fn default_semibold() -> &'static Self {
        static INTER_DISPLAY_SEMIBOLD: &[u8] = include_bytes!("InterDisplay-SemiBold.ttf");
        static FONT: OnceLock<Font> = OnceLock::new();
        FONT.get_or_init(|| Font::load_static_font_from_bytes(INTER_DISPLAY_SEMIBOLD))
    }
}

// Implement comparison of fonts via font IDs.

impl PartialEq for Font {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for Font {}

impl PartialOrd for Font {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Font {
    fn cmp(&self, other: &Self) -> Ordering {
        self.id.cmp(&other.id)
    }
}
