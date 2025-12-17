//! Mesh archive files.

use color::Srgba8;
use math::{Vec2, Vec3};
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

#[repr(C)]
#[derive(Clone, Copy)]
pub struct MeshVertex {
    pub position: Vec3,
    pub color: Srgba8,
    pub normal: Vec3,
    pub uv: Vec2,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct MeshPart {
    pub start_index: u32,
    pub index_count: u32,
    pub base_vertex: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Mesh {
    pub parts: Offset<[MeshPart]>,
}