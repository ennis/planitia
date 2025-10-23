use crate::{vec2, Vec2};
use glam::{ivec3, IVec2, IVec3, Mat4, Vec3, Vec3A};

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

////////////////////////////////////////////////////////////////////////////////////////////////////

/// 3D axis-aligned box with integer coordinates.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct IBox3D {
    pub min: IVec3,
    pub max: IVec3,
}

impl IBox3D {
    pub const fn width(&self) -> u32 {
        (self.max.x - self.min.x) as u32
    }
    pub const fn height(&self) -> u32 {
        (self.max.y - self.min.y) as u32
    }
    pub const fn depth(&self) -> u32 {
        (self.max.z - self.min.z) as u32
    }

    pub const fn size(&self) -> IVec3 {
        ivec3(
            self.max.x - self.min.x,
            self.max.y - self.min.y,
            self.max.z - self.min.z,
        )
    }

    pub const fn from_origin_size_2d(origin: IVec2, size: IVec2) -> Self {
        Self {
            min: origin.extend(0),
            max: ivec3(origin.x + size.x, origin.y + size.y, 1),
        }
    }

    pub const fn from_min_max_2d(min: IVec2, max: IVec2) -> Self {
        Self {
            min: min.extend(0),
            max: max.extend(1),
        }
    }

    pub const fn from_xywh(x: i32, y: i32, width: u32, height: u32) -> Self {
        Self {
            min: ivec3(x, y, 0),
            max: ivec3(x + width as i32, y + height as i32, 1),
        }
    }

    pub const fn from_irect(rect: IRect) -> Self {
        Self {
            min: ivec3(rect.min.x, rect.min.y, 0),
            max: ivec3(rect.max.x, rect.max.y, 1),
        }
    }
}

/// 3D axis-aligned box.
#[derive(Copy, Clone, Debug)]
pub struct Box3D {
    pub min: Vec3,
    pub max: Vec3,
}

impl Box3D {

    pub const NULL: Self = Self {
        min: Vec3::ZERO,
        max: Vec3::ZERO,
    };

    /// Returns the size of the box.
    pub fn size(&self) -> Vec3 {
        self.max - self.min
    }

    /*/// Transforms the bounding box with the provided matrix.
    ///
    /// Reference:
    /// http://dev.theomader.com/transform-bounding-boxes/
    pub fn transform_aabb(&self, tr: &Mat4) -> Box3D {
        let xa = tr.x_axis * self.min.x;
        let xb = tr.x_axis * self.max.x;
        let ya = tr.y_axis * self.min.y;
        let yb = tr.y_axis * self.max.y;
        let za = tr.z_axis * self.min.z;
        let zb = tr.z_axis * self.max.z;

        let min = xa.min(xb) + ya.min(yb) + za.min(zb) + tr.w_axis;
        let max = xa.max(xb) + ya.max(yb) + za.max(zb) + tr.w_axis;

        Box3D {
            min: min.into(),
            max: max.into(),
        }
    }*/

    /// Returns the center of the box.
    pub fn center(&self) -> Vec3 {
        0.5 * (self.min + self.max)
    }

    /// Returns the union of this box with another.
    pub fn union(&self, other: &Box3D) -> Box3D {
        Box3D {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }
}

impl Default for Box3D {
    fn default() -> Self {
        Box3D::NULL
    }
}
