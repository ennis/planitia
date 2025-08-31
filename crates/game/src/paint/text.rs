use crate::paint::atlas::Atlas;
use crate::paint::color::srgba32;
use ab_glyph::{Font as FontTrait, FontArc, ScaleFont};
use math::geom::IRect;
use std::collections::HashMap;

const DEFAULT_SIZE: u32 = 16;

/// TODO support ligatures
/// TODO subpixel offsets
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct CharKey {
    char: char,
    height: u32,
}

#[derive(Copy, Clone, Debug)]
struct GlyphEntry {
    atlas_rect: IRect,
    advance: f32,
}

impl GlyphEntry {
    fn placeholder() -> Self {
        Self {
            atlas_rect: IRect::ZERO,
            advance: 0.0,
        }
    }
}

struct Font {
    data: FontArc,
    /// Map from character to its rectangle in the texture atlas.
    atlas_entries: HashMap<CharKey, GlyphEntry>,
}

/// Rasterizes a glyph and stores it in the atlas if not already present.
fn rasterize_glyph(font: &mut Font, atlas: &mut Atlas, ch: char, size: u32) -> GlyphEntry {
    let key = CharKey { char: ch, height: size };
    if let Some(rect) = font.atlas_entries.get(&key) {
        return *rect;
    }

    let glyph_id = font.data.glyph_id(ch);
    let glyph = glyph_id.with_scale_and_position(size as f32, ab_glyph::point(0.0, 0.0));

    // retrieve the glyph outline
    let Some(outline) = font.data.outline_glyph(glyph) else {
        // missing glyph, return placeholder rect
        // TODO: actual placeholder
        return GlyphEntry::placeholder();
    };

    // reserve space in the atlas
    let bounds = outline.px_bounds();
    let mut atlas_mut = atlas.allocate(bounds.width() as u32, bounds.height() as u32);

    // rasterize the outline
    outline.draw(|x, y, v| {
        // v is a coverage, convert to alpha
        // TODO: maybe there's some correction to do
        let alpha = (v * 255.0) as u8;
        atlas_mut.write(x, y, srgba32(255, 255, 255, alpha));
    });

    let rect = atlas_mut.rect;
    let scaled_font = font.data.as_scaled(size as f32);
    let h_advance = scaled_font.h_advance(glyph_id);
    let entry = GlyphEntry {
        atlas_rect: rect,
        advance: h_advance,
    };
    font.atlas_entries.insert(key, entry);
    entry
}

/// An identifier for a font.
///
/// TODO: replace this with AssetId when we have an asset system.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FontId(pub usize);

pub struct FontCollection {
    /// Shared font atlas.
    atlas: Atlas,
    fonts: HashMap<FontId, Font>,
}

impl FontCollection {
    pub fn new() -> Self {
        Self {
            atlas: Default::default(),
            fonts: Default::default(),
        }
    }

    /// Loads a font file.
    fn load_static_font_from_bytes_inner(&mut self, font_id: FontId, bytes: &'static [u8]) {
        let font = FontArc::try_from_slice(bytes).expect("Failed to load font");
        self.fonts.insert(
            font_id,
            Font {
                data: font,
                atlas_entries: HashMap::new(),
            },
        );
        // preload ASCII characters for default size
        self.load_ascii_chars(font_id, DEFAULT_SIZE);
    }

    fn load_ascii_chars(&mut self, font_id: FontId, size: u32) {
        let mut font = self.fonts.get_mut(&font_id).unwrap();
        for c in 32u8..127u8 {
            let char = c as char;
            rasterize_glyph(&mut font, &mut self.atlas, char, size);
        }
    }
}
