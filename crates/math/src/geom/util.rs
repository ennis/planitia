use glam::Vec3;

/// Computes the minimum distance from point `p` to the line segment defined by points `a` and `b`.
pub fn point_line_dist(p: Vec3, a: Vec3, b: Vec3) -> f32 {
    let ab = b - a;
    let d = (p - a).dot(ab) / ab.dot(ab);
    //d = clamp(d, 0.0, 1.0);
    let p0 = a + d * ab;
    (p - p0).length()
}