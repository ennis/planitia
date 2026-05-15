use math::Vec2;

#[derive(Clone, Copy, Debug)]
pub struct Ellipse {
    pub center: Vec2,
    pub rx: f32,
    pub ry: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct RRect {
    pub rect: math::Rect,
    pub radius: f32,
}
