mod atlas;
mod fill;
mod flatten;
mod gradient;
mod path;
mod renderer;
mod scene;
mod shape;
mod stroke;
mod text;

pub use fill::*;
pub use gradient::*;
pub use path::*;
pub use scene::{DrawGlyphRunOptions, PaintScene, StrokeOptions, render_scene};
pub use shape::*;
pub use text::{Font, GlyphRun, TextFormat, TextLayout};

use crate::paint::atlas::Atlas;
use crate::paint::text::{GlyphCache, GlyphEntry, GlyphId};
use crate::render::RenderTarget;
use color::Srgba8;
use gpu::{ImageUsage, Sampler, Vertex as GpuVertex, vk};
use math::{Camera, U16Vec2, UVec2, Vec2, u16vec2, uvec2, vec2};

/// Blend mode.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[repr(u8)]
pub enum BlendMode {
    Normal,
    Multiply,
    Screen,
    Overlay,
    Darken,
    Lighten,
    ColorDodge,
    ColorBurn,
    HardLight,
    SoftLight,
    Difference,
    Exclusion,
    Mask,
}

/// Controls where a stroke is drawn relative to a guiding path.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum StrokeLocation {
    /// Stroke is centered on the path.
    Center,
    /// Stroke is drawn inside the path.
    Inside,
    /// Stroke is drawn outside the path.
    Outside,
}

/// Vertex used in the painting shaders.
#[repr(C)]
#[derive(Clone, Copy, Debug, GpuVertex)]
struct PaintVertex {
    /// Position
    p: Vec2,
    /// Feather factor
    feather: f32,
}

impl PaintVertex {
    const SIZE_CHECK: () = assert!(size_of::<Self>() == 16);

    pub const fn new(p: Vec2, feather: f32) -> Self {
        Self { p, feather }
    }
}

/// Converts a texel coordinate into u16 normalized UV coordinates.
///
/// Equivalent to `pos / texture_size * 65535`.
pub fn texel_to_normalized_texcoord(pos: Vec2, texture_size: UVec2) -> U16Vec2 {
    u16vec2(((pos.x / texture_size.x as f32) * 65535.0) as u16, ((pos.y / texture_size.y as f32) * 65535.0) as u16)
}

/// Graphics context for 2D painting.
pub struct Painter {
    texture_atlas: Atlas,
    white_pixel_uv: U16Vec2,
    white_pixel_uv_f: Vec2,
    glyph_cache: GlyphCache,
    /// Default texture sampler.
    sampler: gpu::Sampler,
    /// FIXME this has nothing to do with the painter "frontend", it's something specific to the renderer.
    ///       Either we move it to a "RendererPainter" type, or we remove the split between "frontend" and "renderer" entirely.
    render_target: RenderTarget,
}

impl Painter {
    /// Creates a new painting context.
    pub fn new() -> Painter {
        let (atlas, white_pixel_uv) = init_atlas();
        let sampler =
            Sampler::new(gpu::SamplerParams { mag_filter: vk::Filter::LINEAR, min_filter: vk::Filter::LINEAR, .. });
        let white_pixel_uv_f =
            vec2(white_pixel_uv.x as f32 / (u16::MAX as f32), white_pixel_uv.y as f32 / (u16::MAX as f32));

        Painter {
            glyph_cache: Default::default(),
            texture_atlas: atlas,
            white_pixel_uv,
            white_pixel_uv_f,
            sampler,
            render_target: RenderTarget::new(
                vk::Format::R8G8B8A8_UNORM,
                ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSFER_DST | ImageUsage::STORAGE,
            ),
        }
    }

    /// Rasterizes a glyph and adds it to the texture atlas if it's not already cached.
    pub(crate) fn rasterize_glyph(
        &mut self,
        font: &Font,
        id: GlyphId,
        size: u32,
        position: Vec2,
    ) -> (GlyphEntry, Vec2) {
        let (entry, quantized_pos) =
            self.glyph_cache.rasterize_glyph(&mut self.texture_atlas, &font, id, size, position);
        (entry, quantized_pos)
    }

    /// Flushes pending changes to the glyph texture atlas.
    pub(crate) fn update_texture_atlas(&mut self) {
        let _ = self.texture_atlas.prepare_texture();
    }
}

/// Initializes the paint texture atlas.
///
/// Returns the atlas and the UV coordinate of a white pixel in the atlas.
fn init_atlas() -> (Atlas, U16Vec2) {
    let mut atlas = Atlas::new(1024, 1024);
    // Add a white pixel at (0,0) for drawing solid colors without needing additional logic in the
    // shaders
    let rect = atlas.write(1, 1, &[Srgba8::WHITE], 1, 1);
    let pos = texel_to_normalized_texcoord(
        vec2(rect.min.x as f32 + 0.5, rect.min.y as f32 + 0.5),
        uvec2(atlas.width, atlas.height),
    );
    (atlas, pos)
}

#[cfg(test)]
mod test {}
