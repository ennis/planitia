//! Mesh archive files.

use math::Vec3;
use utils::archive::Offset;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
// NoPadding
pub struct PolylineVertex {
    pub position: Vec3,
    pub arclength: f32,
    pub normal: Vec3,
    // TODO the rest
}


#[repr(C)]
#[derive(Clone, Copy)]
pub struct PolylineData(Offset<[PolylineVertex]>);