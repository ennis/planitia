use math::geom::Camera;
use math::{Rect, Vec2};
use crate::paint::tessellation::{Mesh, Tessellator};
use crate::paint::{FeatherVertex, Srgba32};
use crate::paint::shape::RectShape;

#[derive(Default,Debug)]
pub struct Primitive {
    pub mesh: Mesh,
    pub clip: Rect,
}


/// Draws shapes onto a target image.
pub struct PaintScene {
    pub(crate) tess: Tessellator,
    prims: Vec<Primitive>,
    clip_stack: Vec<Rect>,
}

impl Default for PaintScene {
    fn default() -> Self {
        Self::new()
    }
}

impl PaintScene {
    pub fn new() -> Self {
        Self {
            tess: Tessellator::new(),
            prims: vec![],
            clip_stack: vec![Rect::INFINITE],
        }
    }

    fn end_prim(&mut self) {
        if !self.tess.is_empty() {
            let mesh = self.tess.finish_and_reset();
            let prim = Primitive {
                mesh,
                clip: self.clip_stack.pop().unwrap(),
            };
            self.prims.push(prim);
        }
    }

    /// Draws a rounded rectangle at the specified position with the given size and corner radius.
    pub fn fill_rrect(&mut self, rect: Rect, radius: f32, color: impl Into<Srgba32>) {
        let color = color.into();
        self.tess.fill_rrect(RectShape {
            rect,
            radius,
            colors: [color; 4],
            feather: 0.0,
        });
    }

    fn clip_rect(&self) -> Rect {
        *self.clip_stack.last().unwrap()
    }

    /// Pushes a clip rectangle onto the stack. All subsequent drawing operations will be clipped to this rectangle.
    pub fn push_clip(&mut self, rect: Rect) {
        self.end_prim();
        let clip = self.clip_rect().intersect(&rect).unwrap_or_default();
        self.clip_stack.push(clip);
    }

    /// Pops the last clip rectangle from the stack.
    pub fn pop_clip(&mut self) {
        self.end_prim();
        self.clip_stack.pop();
    }

    pub fn finish(mut self) -> Vec<Primitive> {
        self.end_prim();
        self.prims
    }
}
