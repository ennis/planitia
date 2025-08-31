use math::Vec2;
use crate::paint::Srgba32;

#[derive(Clone, Copy, Debug)]
pub struct EllipseShape {
    pub center: Vec2,
    pub rx: f32,
    pub ry: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct RectShape {
    pub rect: math::Rect,
    pub radius: f32,
    /// Colors for the four corners in the order: top-left (min.y, min.x), top-right, bottom-right, bottom-left
    pub colors: [Srgba32;4],
    /// Feather radius
    pub feather: f32,
}

pub enum Shape {
    Rect(RectShape),
    Ellipse(EllipseShape),
}