use crate::Rect;
use glam::{mat3, vec3, Mat3, Vec3};

/// Computes the minimum distance from point `p` to the line segment defined by points `a` and `b`.
pub fn point_line_dist(p: Vec3, a: Vec3, b: Vec3) -> f32 {
    let ab = b - a;
    let d = (p - a).dot(ab) / ab.dot(ab);
    //d = clamp(d, 0.0, 1.0);
    let p0 = a + d * ab;
    (p - p0).length()
}

/// Computes the affine 2D transform that maps `source_rect` onto
/// `target_rect`.
pub fn rect_transform(source_local_rect: Rect, target_uv_rect: Rect) -> Mat3 {
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
