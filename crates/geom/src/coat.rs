use math::Vec3;
use color::Srgba8;
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
    pub color: Srgba8,
    pub normal: Vec3,
    pub arclength: f32,
    pub tangent: Vec3,
    //pub width: u8,
    //pub noise: u8,
    //pub falloff: u8,
    //pub stamp_texture: u8,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct Coat {
    /// Index into `geo.strokes`.
    pub start_stroke: u32,
    pub stroke_count: u32,
    // Index into `geo.stroke_vertices`.
    pub start_vertex: u32,
    pub vertex_count: u32,
    pub color_ramps: Offset<[ColorRampData]>,
    pub width_profiles: Offset<[WidthProfileData]>,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct ColorRampData {
    pub colors: Offset<[Srgba8]>,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct WidthProfileData {
    pub widths: Offset<[u8]>,
}


/// Sweep macro-stroke
#[repr(C)]
#[derive(Copy, Clone)]
pub struct SweptStroke {
    pub ref_position: Vec3,
    pub start_vertex: u32,
    pub vertex_count: u32,
    pub cross_section_start_vertex: u32,
    pub cross_section_vertex_count: u32,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct CrossSectionVertex {
    pub position: Vec3,
    pub normal: Vec3,
}