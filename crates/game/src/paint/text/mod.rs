mod format;
mod layout;
mod text_run;

use crate::paint::atlas::Atlas;
use crate::paint::color::srgba32;
use ab_glyph::{Font as FontTrait, FontArc, ScaleFont};
pub use format::TextFormat;
pub use layout::{GlyphRun, TextLayout};
use log::debug;
use math::geom::IRect;
use math::{IVec2, U16Vec2, Vec2, ivec2, u16vec2, vec2};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::ops::Range;
use std::sync::OnceLock;
use std::sync::atomic::AtomicUsize;
pub use text_run::TextRun;

const DEFAULT_SIZE: u32 = 16;

/// TODO support ligatures
/// TODO subpixel offsets
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct GlyphKey {
    glyph_id: GlyphId,
    font_id: FontId,
    height: u32,
}

#[derive(Copy, Clone, Debug)]
pub struct GlyphEntry {
    /// Pixel glyph bounds.
    pub px_bounds: IRect,
    /// Position of the glyph in the atlas texture.
    pub atlas_pos: IVec2,
    /// Normalized texture coordinates (min/max)
    pub normalized_texcoords: [U16Vec2; 2],
    pub advance: f32,
}

impl GlyphEntry {
    pub fn placeholder() -> Self {
        Self {
            px_bounds: Default::default(),
            atlas_pos: Default::default(),
            normalized_texcoords: [U16Vec2::default(); 2],
            advance: 0.0,
        }
    }

    pub fn texture_rect(&self) -> IRect {
        IRect {
            min: self.atlas_pos,
            max: self.atlas_pos + (self.px_bounds.max - self.px_bounds.min),
        }
    }
}

/// Texture atlas of rasterized glyphs for many fonts & sizes.
pub(crate) struct GlyphCache {
    atlas: Atlas,
    /// Map from char key ((char,font,size) tuple) to its rectangle in the texture atlas.
    entries: HashMap<GlyphKey, GlyphEntry>,
}

impl Default for GlyphCache {
    fn default() -> Self {
        Self::new()
    }
}

impl GlyphCache {
    /// Creates a new empty glyph cache.
    pub(crate) fn new() -> Self {
        Self {
            atlas: Default::default(),
            entries: Default::default(),
        }
    }

    /// Rasterizes a glyph and stores it in the atlas if not already present.
    ///
    /// TODO: subpixel offsets
    pub fn rasterize_glyph(&mut self, font: &Font, id: GlyphId, size: u32) -> GlyphEntry {
        //debug!("Rasterizing glyph id {:?} size {} for font id {:?}", id, size, font.id);

        let key = GlyphKey {
            glyph_id: id,
            font_id: font.id,
            height: size,
        };
        if let Some(rect) = self.entries.get(&key) {
            return *rect;
        }

        let glyph = id.with_scale_and_position(size as f32, ab_glyph::point(0.0, 0.0));

        // retrieve the glyph outline
        let Some(outline) = font.data.outline_glyph(glyph) else {
            // missing glyph, return placeholder rect
            // TODO: actual placeholder
            debug!("Missing glyph id {:?} in font id {:?}", id, font.id);
            return GlyphEntry::placeholder();
        };

        // reserve space in the atlas
        let bounds = outline.px_bounds();
        let mut atlas_mut = self.atlas.allocate(bounds.width() as u32, bounds.height() as u32);

        // rasterize the outline
        outline.draw(|x, y, v| {
            // v is a coverage, convert to alpha
            // TODO: maybe there's some correction to do
            let alpha = (v * 255.0) as u8;
            atlas_mut.write(x, y, srgba32(255, 255, 255, alpha));
        });

        let atlas_rect = atlas_mut.rect;
        let scaled_font = font.data.as_scaled(size as f32);
        let h_advance = scaled_font.h_advance(id);

        // compute normalized texture coordinates
        let normalized_texcoords = [
            u16vec2(
                ((atlas_rect.min.x as f32) / (self.atlas.width as f32) * 65535.0) as u16,
                ((atlas_rect.min.y as f32) / (self.atlas.height as f32) * 65535.0) as u16,
            ),
            u16vec2(
                ((atlas_rect.max.x as f32) / (self.atlas.width as f32) * 65535.0) as u16,
                ((atlas_rect.max.y as f32) / (self.atlas.height as f32) * 65535.0) as u16,
            ),
        ];

        let entry = GlyphEntry {
            px_bounds: IRect {
                min: ivec2(bounds.min.x as i32, bounds.min.y as i32),
                max: ivec2(bounds.max.x as i32, bounds.max.y as i32),
            },
            atlas_pos: atlas_rect.top_left(),
            normalized_texcoords,
            advance: h_advance,
        };
        debug!("glyph id {:?} bounds {:?} atlas_rect={:?}", id, bounds, atlas_rect);

        self.entries.insert(key, entry);
        entry
    }

    pub fn texture_handle(&self) -> gpu::TextureHandle {
        self.atlas.texture_handle()
    }

    pub fn use_texture(&mut self, cmd: &mut gpu::CommandStream) -> gpu::TextureHandle {
        self.atlas.use_texture(cmd)
    }

    /*pub fn texture_handle(&self) -> gpu::ImageHandle {
        self.atlas.te
    }*/
}

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

pub type GlyphId = ab_glyph::GlyphId;

/// Represents a font file in memory.
#[derive(Clone, Debug)]
pub struct Font {
    data: FontArc,
    id: FontId,
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
        static INTER_DISPLAY_REGULAR: &[u8] = include_bytes!("../InterDisplay-Regular.ttf");
        static FONT: OnceLock<Font> = OnceLock::new();
        FONT.get_or_init(|| Font::load_static_font_from_bytes(INTER_DISPLAY_REGULAR))
    }

    /// Returns the default font for semibold text.
    pub fn default_semibold() -> &'static Self {
        static INTER_DISPLAY_SEMIBOLD: &[u8] = include_bytes!("../InterDisplay-SemiBold.ttf");
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
    let mut pos = 0.0;
    let mut prev_glyph_id: Option<GlyphId> = None;

    for (ch_pos, ch) in text.char_indices() {
        let ch_len_utf8 = ch.len_utf8();
        let src_range = ch_pos..(ch_pos + ch_len_utf8);
        let glyph_id = font.glyph_id(ch);

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

        // determine boundary type
        // TODO: more sophisticated boundary detection

        let whitespace = ch.is_whitespace();
        let possible_line_break = whitespace || ch == '-';
        let mandatory_line_break = ch == '\n' || ch == '\r';

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
