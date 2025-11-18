use crate::paint::atlas::Atlas;
use crate::paint::texel_to_normalized_texcoord;
use crate::paint::text::{Font, FontId, GlyphId};
use ab_glyph::{Font as FontTrait, ScaleFont};
use math::geom::IRect;
use math::{IVec2, U16Vec2, Vec2, ivec2, u16vec2, uvec2, vec2};
use std::collections::HashMap;
use color::srgba32;

const SUBPIXEL_X_GRID_SIZE: u32 = 8;
const SUBPIXEL_Y_GRID_SIZE: u32 = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct GlyphKey {
    glyph_id: GlyphId,
    font_id: FontId,
    height: u32,
    subpixel_x_key: u8, // 0..=7
    subpixel_y_key: u8, // 0..=7
}

/// An entry in the glyph cache.
#[derive(Copy, Clone, Debug)]
pub(crate) struct GlyphEntry {
    /// Extents of the glyph image in pixels relative to the glyph origin point (which is usually on the baseline of the glyph).
    pub(crate) px_bounds: IRect,
    /// Position of the glyph in the atlas texture.
    pub(crate) atlas_pos: IVec2,
    /// Normalized texture coordinates (min/max) in the atlas texture.
    pub(crate) normalized_texcoords: [U16Vec2; 2],
    /// Horizontal advance of the glyph in layout coordinates.
    pub(crate) advance: f32,
}

impl GlyphEntry {
    /// Returns a placeholder glyph entry.
    ///
    /// TODO currently this is just an empty rectangle, but we should have a proper placeholder glyph.
    pub fn placeholder() -> Self {
        Self {
            px_bounds: Default::default(),
            atlas_pos: Default::default(),
            normalized_texcoords: [U16Vec2::default(); 2],
            advance: 0.0,
        }
    }

    /*pub fn texture_rect(&self) -> IRect {
        IRect {
            min: self.atlas_pos,
            max: self.atlas_pos + (self.px_bounds.max - self.px_bounds.min),
        }
    }*/
}

impl Default for GlyphEntry {
    fn default() -> Self {
        Self::placeholder()
    }
}

/// Texture atlas of rasterized glyphs for many fonts & sizes.
pub(crate) struct GlyphCache {
    /// Map from char key ((char,font,size) tuple) to its rectangle in the texture atlas.
    entries: HashMap<GlyphKey, GlyphEntry>,
}

impl GlyphCache {
    /// Creates a new empty glyph cache.
    pub(crate) fn new() -> Self {
        Self {
            entries: Default::default(),
        }
    }

    /// Rasterizes a glyph and stores it in the atlas if not already present.
    ///
    /// The position is used to determine the subpixel offset for better quality.
    ///
    /// # Return value
    ///
    /// A tuple `(GlyphEntry, quantized_position)` where `GlyphEntry` is the cached glyph entry,
    /// and `quantized_position` is the position rounded down to integer coordinates.
    pub fn rasterize_glyph(
        &mut self,
        atlas: &mut Atlas,
        font: &Font,
        id: GlyphId,
        size: u32,
        position: Vec2,
    ) -> (GlyphEntry, Vec2) {
        //debug!("Rasterizing glyph id {:?} size {} for font id {:?}", id, size, font.id);

        // quantize to 8x8 subpixel grid
        let subpixel_x_key = ((position.x - position.x.floor()) * (SUBPIXEL_X_GRID_SIZE as f32)) as u8;
        let subpixel_y_key = ((position.y - position.y.floor()) * (SUBPIXEL_Y_GRID_SIZE as f32)) as u8;
        let subpixel_x = (subpixel_x_key as f32) / (SUBPIXEL_X_GRID_SIZE as f32);
        let subpixel_y = (subpixel_y_key as f32) / (SUBPIXEL_Y_GRID_SIZE as f32);
        let quantized_pos = vec2(position.x.floor(), position.y.floor());

        let key = GlyphKey {
            glyph_id: id,
            font_id: font.id,
            height: size,
            subpixel_x_key,
            subpixel_y_key,
        };

        if let Some(entry) = self.entries.get(&key) {
            return (GlyphEntry { ..*entry }, quantized_pos);
        }

        let glyph = id.with_scale_and_position(size as f32, ab_glyph::point(subpixel_x, subpixel_y));

        // explicitly skip newlines, as

        // retrieve the glyph outline
        let Some(outline) = font.data.outline_glyph(glyph) else {
            // missing glyph, return placeholder rect
            // TODO: actual placeholder
            //let char = font.data.codepoint_ids().find(|(gid, _)| *gid == id).map(|(_, c)| c);
            //debug!("Missing glyph id {:04x?} char={char:?} in font id {:?}", id, font.id);
            return (GlyphEntry::placeholder(), Vec2::ZERO);
        };

        // reserve space in the atlas
        let bounds = outline.px_bounds();
        let mut atlas_mut = atlas.allocate(bounds.width() as u32, bounds.height() as u32, 1, 1);

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
            texel_to_normalized_texcoord(atlas_rect.min.as_vec2(), uvec2(atlas.width, atlas.height)),
            texel_to_normalized_texcoord(atlas_rect.max.as_vec2(), uvec2(atlas.width, atlas.height)),
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

        //let char = font.data.codepoint_ids().find(|(gid, _)| *gid == id).map(|(_, c)| c);
        /*debug!(
            "glyph id {:?} bounds=[{},{} → {},{}] atlas_rect=[{},{} → {},{}], char={char:?}",
            id,
            bounds.min.x,
            bounds.min.y,
            bounds.max.x,
            bounds.max.y,
            atlas_rect.min.x,
            atlas_rect.min.y,
            atlas_rect.max.x,
            atlas_rect.max.y
        );*/

        self.entries.insert(key, entry);
        (entry, quantized_pos)
    }
}

impl Default for GlyphCache {
    fn default() -> Self {
        Self::new()
    }
}
