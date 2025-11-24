use math::Vec3;
use color::Srgba32;
use utils::archive::Offset;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct Stroke {
    /// Reference position on the surface mesh.
    pub ref_position: Vec3,
    pub start_vertex: u32,
    pub vertex_count: u32,
    pub brush_index: u16,
    /// Width profile (1D texture index).
    pub width_profile: u16,
    /// Color/opacity profile (1D texture index).
    pub color_ramp: u16,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct StrokeVertex {
    pub position: Vec3,
    pub color: Srgba32,
    pub normal: Vec3,
    pub arclength: f32,
    pub width: u8,
    pub noise: u8,
    pub falloff: u8,
    pub stamp_texture: u8,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct Coat {
    pub strokes: Offset<[Stroke]>,
    pub vertices: Offset<[StrokeVertex]>,
    pub color_ramps: Offset<[ColorRampData]>,
    pub width_profiles: Offset<[WidthProfileData]>,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct ColorRampData {
    pub colors: Offset<[Srgba32]>,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct WidthProfileData {
    pub widths: Offset<[u8]>,
}