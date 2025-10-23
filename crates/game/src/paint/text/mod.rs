mod format;
mod layout;
mod text_run;
mod glyph_cache;
mod font;

use ab_glyph::{Font as FontTrait, ScaleFont};
use math::geom::IRect;
use math::{vec2, Vec2};
use std::ops::Range;

pub use font::{Font, FontId};
pub use format::TextFormat;
pub use layout::{GlyphRun, TextLayout};
pub(crate) use glyph_cache::{GlyphEntry, GlyphCache};


/// Glyph identifier.
pub type GlyphId = ab_glyph::GlyphId;

/// Glyph.
#[derive(Clone, Copy, Debug)]
pub struct Glyph {
    pub id: GlyphId,
    /// Offset (relative to the origin of the cluster).
    pub offset: Vec2,
    /// Advance.
    pub advance: f32,
}

/*
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Boundary {
    None,
    Word,
    Line,
    MandatoryLine,
}*/

/// glyph cluster.
#[derive(Clone)]
pub struct GlyphCluster<'a> {
    /// Text source range corresponding to this cluster.
    pub source_range: Range<usize>,
    pub glyphs: &'a [Glyph],
    /// Total advance of the glyph cluster.
    pub advance: f32,
    pub bounds: IRect,
    /// Indicates whether this cluster is a possible line break point.
    pub possible_line_break: bool,
    /// Indicates whether this cluster is a mandatory line break point.
    pub mandatory_line_break: bool,
    /// Indicates that this cluster is white space.
    pub whitespace: bool,
}

///
pub struct ShapingParams {
    pub font: Font,
    pub size: f32,
}

/// Performs text analysis and shape the given text run into a sequence of glyph clusters.
///
/// NOTE: currently it doesn't do any CTL shaping or ligatures.
pub fn shape_text(format: &ShapingParams, text: &str, callback: &mut dyn FnMut(&GlyphCluster)) {
    let font = &format.font;
    let size = format.size;
    let mut prev_glyph_id: Option<GlyphId> = None;

    for (ch_pos, ch) in text.char_indices() {
        let ch_len_utf8 = ch.len_utf8();
        let src_range = ch_pos..(ch_pos + ch_len_utf8);

        // determine boundary type
        // TODO: more sophisticated boundary detection
        let whitespace = ch.is_whitespace();
        let possible_line_break = whitespace || ch == '-';
        let mandatory_line_break = ch == '\n' || ch == '\r';

        let glyph_id = font.glyph_id(ch);

        if glyph_id.0 != 0 {
            //debug!("Shaping char '{ch}' (id={glyph_id:?}) at {src_range:?}");

            let mut advance = font.h_advance(glyph_id, size);
            // adjust for kerning
            // note that this could also be an offset on the glyph position
            let next_glyph_id = text[src_range.end..].chars().next().map(|c| font.glyph_id(c));
            if let Some(next_glyph_id) = next_glyph_id {
                let kern = font.kern(glyph_id, next_glyph_id, size);
                advance += kern;
            }

            let glyph = Glyph {
                id: glyph_id,
                // for simple shaping the offset is always zero
                offset: vec2(0.0, 0.0),
                advance,
            };

            callback(&GlyphCluster {
                source_range: src_range,
                glyphs: std::slice::from_ref(&glyph),
                advance,
                bounds: Default::default(),
                possible_line_break,
                mandatory_line_break,
                whitespace,
            });
            prev_glyph_id = Some(glyph_id);
        } else {
            // cluster without associated glyph
            callback(&GlyphCluster {
                source_range: src_range,
                glyphs: &[],
                advance: 0.0,
                bounds: Default::default(),
                possible_line_break,
                mandatory_line_break,
                whitespace,
            });
            prev_glyph_id = None;
        }
    }
}

/*
/// A collection of fonts.
pub struct FontCollection {
    fonts: HashMap<FontId, Font>,
}

impl FontCollection {
    /// Creates a new font collection. Loads default fonts.
    pub fn new() -> Self {
        let mut this = Self {
            atlas: Default::default(),
            fonts: Default::default(),
        };
        this.load_static_font_from_bytes_inner(FONT_REGULAR, INTER_DISPLAY_REGULAR);
        this.load_static_font_from_bytes_inner(FONT_SEMIBOLD, INTER_DISPLAY_SEMIBOLD);
        this
    }



    fn load_ascii_chars(&mut self, font_id: FontId, size: u32) {
        let mut font = self.fonts.get_mut(&font_id).unwrap();
        for c in 32u8..127u8 {
            let char = c as char;
            rasterize_glyph(&mut font, &mut self.atlas, char, size);
        }
    }
}
*/
