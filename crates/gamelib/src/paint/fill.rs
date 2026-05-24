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

/// Describes how to paint the interior of a shape.
///
/// It is convertible from [`Srgba8`] via `From`, so a solid color can be passed directly to functions
/// taking `Into<Fill>`.
pub enum Fill {
    /// Solid color fill.
    Solid(Srgba8),

    /// Textured fill.
    Texture {
        /// The texture to fill the shape with.
        texture: gpu::TextureHandle,
        sampler: gpu::SamplerHandle,
        /// Affine transform that maps local coordinates to UV coordinates.
        local_to_uv: math::Mat3,
    },

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

impl Fill {
    /// Creates a [`Fill::Texture`] that maps `source_local_rect` in shape-local space to
    /// `target_texel_rect` in texel space (pixel coordinates).
    pub fn make_texture_fill(
        texture: gpu::TextureHandle,
        sampler: gpu::SamplerHandle,
        source_local_rect: Rect,
        target_uv_rect: Rect,
    ) -> Self {
        Fill::Texture { texture, sampler, local_to_uv: rect_transform(source_local_rect, target_uv_rect) }
    }

    pub fn transform(&self, transform: &Mat3) -> Self {
        match self {
            Fill::Solid(color) => Fill::Solid(*color),
            Fill::Texture { texture, sampler, local_to_uv } => {
                Fill::Texture { texture: *texture, sampler: *sampler, local_to_uv: *local_to_uv }
            }
            Fill::LinearGradient(gradient) => Fill::LinearGradient(LinearGradientFill {
                start: transform.transform_point2(gradient.start),
                end: transform.transform_point2(gradient.end),
                ramp: gradient.ramp,
                extend_mode: gradient.extend_mode,
            }),
        }
    }
}
