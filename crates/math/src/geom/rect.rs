use glam::IVec2;
use crate::{vec2, Vec2};

#[derive(Clone, Copy, Debug, PartialEq, Default)]
#[repr(C)]
pub struct Rect {
    pub min: Vec2,
    pub max: Vec2,
}

impl Rect {
    pub const INFINITE: Self = Self {
        min: Vec2::new(f32::NEG_INFINITY, f32::NEG_INFINITY),
        max: Vec2::new(f32::INFINITY, f32::INFINITY),
    };
    pub const ZERO: Self = Self {
        min: Vec2::ZERO,
        max: Vec2::ZERO,
    };

    pub const fn from_min_max(min: Vec2, max: Vec2) -> Self {
        Self { min, max }
    }

    pub const fn from_origin_size(origin: Vec2, size: Vec2) -> Self {
        Self {
            min: origin,
            max: vec2(origin.x + size.x, origin.y + size.y),
        }
    }

    pub const fn top_left(&self) -> Vec2 {
        self.min
    }

    pub const fn top_right(&self) -> Vec2 {
        Vec2::new(self.max.x, self.min.y)
    }

    pub const fn bottom_left(&self) -> Vec2 {
        Vec2::new(self.min.x, self.max.y)
    }

    pub const fn bottom_right(&self) -> Vec2 {
        self.max
    }

    pub fn intersect(&self, other: &Rect) -> Option<Rect> {
        let min = Vec2::new(self.min.x.max(other.min.x), self.min.y.max(other.min.y));
        let max = Vec2::new(self.max.x.min(other.max.x), self.max.y.min(other.max.y));
        if min.x < max.x && min.y < max.y {
            Some(Rect { min, max })
        } else {
            None
        }
    }

    pub const fn is_null(&self) -> bool {
        self.min.x >= self.max.x || self.min.y >= self.max.y
    }

    pub const fn width(&self) -> f32 {
        self.max.x - self.min.x
    }

    pub const fn height(&self) -> f32 {
        self.max.y - self.min.y
    }

    pub const fn is_infinite(&self) -> bool {
        self.min.x == f32::NEG_INFINITY
            && self.min.y == f32::NEG_INFINITY
            && self.max.x == f32::INFINITY
            && self.max.y == f32::INFINITY
    }

    pub const fn translate(&self, offset: Vec2) -> Self {
        Self {
            min: vec2(self.min.x + offset.x, self.min.y + offset.y),
            max: vec2(self.max.x + offset.x, self.max.y + offset.y),
        }
    }
}

pub const fn rect_xywh(x: f32, y: f32, w: f32, h: f32) -> Rect {
    Rect {
        min: Vec2::new(x, y),
        max: Vec2::new(x + w, y + h),
    }
}


#[derive(Clone, Copy, Debug, PartialEq, Default)]
#[repr(C)]
pub struct IRect {
    pub min: IVec2,
    pub max: IVec2,
}

impl IRect {
    pub const ZERO: Self = Self {
        min: IVec2::ZERO,
        max: IVec2::ZERO,
    };

    pub const fn from_min_max(min: IVec2, max: IVec2) -> Self {
        Self { min, max }
    }

    pub const fn from_origin_size(origin: IVec2, size: IVec2) -> Self {
        Self {
            min: origin,
            max: IVec2::new(origin.x + size.x, origin.y + size.y),
        }
    }

    pub const fn top_left(&self) -> IVec2 {
        self.min
    }

    pub const fn top_right(&self) -> IVec2 {
        IVec2::new(self.max.x, self.min.y)
    }

    pub const fn bottom_left(&self) -> IVec2 {
        IVec2::new(self.min.x, self.max.y)
    }

    pub const fn bottom_right(&self) -> IVec2 {
        self.max
    }

    pub fn intersect(&self, other: &IRect) -> Option<IRect> {
        let min = IVec2::new(self.min.x.max(other.min.x), self.min.y.max(other.min.y));
        let max = IVec2::new(self.max.x.min(other.max.x), self.max.y.min(other.max.y));
        if min.x < max.x && min.y < max.y {
            Some(IRect { min, max })
        } else {
            None
        }
    }

    pub const fn is_null(&self) -> bool {
        self.min.x >= self.max.x || self.min.y >= self.max.y
    }

    pub const fn width(&self) -> i32 {
        self.max.x - self.min.x
    }

    pub const fn height(&self) -> i32 {
        self.max.y - self.min.y
    }

    pub const fn to_rect(&self) -> Rect {
        Rect {
            min: Vec2::new(self.min.x as f32, self.min.y as f32),
            max: Vec2::new(self.max.x as f32, self.max.y as f32),
        }
    }
}
pub const fn irect_xywh(x: i32, y: i32, w: i32, h: i32) -> IRect {
    IRect {
        min: IVec2::new(x, y),
        max: IVec2::new(x + w, y + h),
    }
}
