mod atlas;
mod fill;
mod flatten;
mod path;
mod pipelines;
mod renderer;
mod scene;
mod shape;
mod tessellation;
mod text;
mod stroke;
mod gradient;

pub use path::*;
pub use scene::{DrawGlyphRunOptions, PaintScene};
pub use shape::*;
pub use gradient::*;
pub use fill::*;
pub use text::{GlyphRun, TextFormat, TextLayout};

use crate::paint::atlas::Atlas;
use crate::paint::pipelines::Pipelines;
use crate::paint::text::GlyphCache;
use crate::render::RenderTarget;
use color::Srgba8;
use gpu::{vk, ImageUsage, Sampler, Vertex as GpuVertex};
use math::Camera;
use math::{u16vec2, uvec2, vec2, U16Vec2, UVec2, Vec2};


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

pub struct PaintRenderParams<'a> {
    pub camera: Camera,
    pub color_target: &'a gpu::Image,
    pub depth_target: Option<&'a gpu::Image>,
}

/// Holds resources for painting.
pub struct Painter {
    pipelines: Pipelines,
    texture_atlas: Atlas,
    white_pixel_uv: U16Vec2,
    white_pixel_uv_f: Vec2,
    glyph_cache: GlyphCache,
    sampler: gpu::Sampler,
    color_format: vk::Format,
    depth_format: Option<vk::Format>,
    // V2 renderer
    coverage_target: RenderTarget,
}

impl Painter {
    /// Creates a new painter.
    ///
    /// `target_color_format` and `target_depth_format` specify the formats of the render targets that will be used during rendering.
    pub fn new(target_color_format: gpu::Format, target_depth_format: Option<gpu::Format>) -> Painter {
        let (atlas, white_pixel_uv) = init_atlas();
        let sampler =
            Sampler::new(gpu::SamplerCreateInfo { mag_filter: vk::Filter::LINEAR, min_filter: vk::Filter::LINEAR, .. });
        let white_pixel_uv_f =
            vec2(white_pixel_uv.x as f32 / (u16::MAX as f32), white_pixel_uv.y as f32 / (u16::MAX as f32));

        Painter {
            pipelines: Pipelines::create(target_color_format, target_depth_format),
            color_format: target_color_format,
            depth_format: target_depth_format,
            glyph_cache: Default::default(),
            texture_atlas: atlas,
            white_pixel_uv,
            white_pixel_uv_f,
            sampler,
            coverage_target: RenderTarget::new(
                vk::Format::R32G32B32A32_SFLOAT,
                ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSFER_DST | ImageUsage::STORAGE,
            ),
        }
    }

    /// Returns a scene builder.
    pub fn build_scene(&mut self) -> PaintScene<'_> {
        PaintScene::new(self)
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
