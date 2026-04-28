use color::Srgba8;
use math::{mat3, vec3, Affine2, Affine3A, Mat3, Mat3A};

/// Represents a fill style for painting shapes.
pub enum Fill {
    /// Solid color fill
    Solid(Srgba8),
    /// Texture fill
    Texture {
        texture: gpu::TextureHandle,
        /// How to transform local coordinates to uvs.
        uv_transform: math::Mat3,
    },
}

impl From<Srgba8> for Fill {
    fn from(color: Srgba8) -> Self {
        Fill::Solid(color)
    }
}

/// Returns the affine transform mapping the source rectangle to the target rectangle.
pub fn rect_transform(source_local_rect: math::Rect, target_uv_rect: math::Rect) -> math::Mat3 {
    let src = source_local_rect;
    let dst = target_uv_rect;
    let scale_x = dst.width() / src.width();
    let scale_y = dst.height() / src.height();
    mat3(
        vec3(scale_x, 0.0, 0.0),
        vec3(0.0, scale_y, 0.0),
        vec3(dst.min.x - src.min.x * scale_x, dst.min.y - src.min.y * scale_y, 1.0),
    )
}


