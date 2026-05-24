//! Fill styles for painted shapes.
use crate::paint::GradientExtendMode;
use crate::paint::scene::GradientRamp;
use color::Srgba8;
use math::{Mat3, Rect, Vec2, rect_transform};

/// Describes a linear gradient.
#[derive(Debug, Clone, Copy)]
pub struct LinearGradientFill {
    /// Start point of the gradient line.
    pub start: Vec2,
    /// End point of the gradient line.
    pub end: Vec2,
    /// Gradient ramp
    pub ramp: GradientRamp,
    pub extend_mode: GradientExtendMode,
}

#[derive(Debug, Clone, Copy)]
pub struct TextureFill {
    /// The texture to fill the shape with.
    pub texture: gpu::TextureHandle,
    pub sampler: gpu::SamplerHandle,
    /// Affine transform that maps local coordinates to UV coordinates.
    pub local_to_uv: Mat3,
    /// Modulation color.
    pub color: Srgba8 = Srgba8::WHITE,
}

/// Describes how to paint the interior of a shape.
///
/// It is convertible from [`Srgba8`] via `From`, so a solid color can be passed directly to functions
/// taking `Into<Fill>`.
pub enum Fill {
    /// Solid color fill.
    Solid(Srgba8),
    /// Textured fill.
    Texture(TextureFill),
    /// Linear gradient fill.
    LinearGradient(LinearGradientFill),
}

impl From<Srgba8> for Fill {
    fn from(color: Srgba8) -> Self {
        Fill::Solid(color)
    }
}

impl From<LinearGradientFill> for Fill {
    fn from(gradient: LinearGradientFill) -> Self {
        Fill::LinearGradient(gradient)
    }
}

impl From<TextureFill> for Fill {
    fn from(texture_fill: TextureFill) -> Self {
        Fill::Texture(texture_fill)
    }
}

impl Fill {
    /// Creates a [`Fill::Texture`] that maps `source_local_rect` in shape-local space to
    /// `target_texel_rect` in texel space (pixel coordinates).
    pub fn make_texture_fill(
        texture: gpu::TextureHandle,
        sampler: gpu::SamplerHandle,
        source_local_rect: Rect,
        target_uv_rect: Rect,
        color: Srgba8,
    ) -> Self {
        Fill::Texture(TextureFill {
            texture,
            sampler,
            local_to_uv: rect_transform(source_local_rect, target_uv_rect),
            color,
        })
    }
}
