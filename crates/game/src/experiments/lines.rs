use crate::{SceneInfo, SceneInfoUniforms};
use gamelib::pipeline_cache::get_graphics_pipeline;
use gpu::PrimitiveTopology::{TriangleList, TriangleStrip};
use gpu::Ptr;

#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct LineVertex {
    pub position: math::Vec3,
    pub color: color::Srgba8,
}

const _: () = assert!(size_of::<LineVertex>() == 16);

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Line {
    pub start_vertex: u32,
    pub vertex_count: u32,
    pub width : f32 = 1.0,
    pub filter_width: f32 = 1.0,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RootParams {
    scene_info: Ptr<SceneInfoUniforms>,
    vertices: Ptr<[LineVertex]>,
    start_vertex: u32,
    vertex_count: u32,
    line_width: f32,
    filter_width: f32,
}

/// Draws 3d polylines with given line width in world units.
pub fn draw_lines<'a>(
    encoder: &mut gpu::RenderEncoder,
    vertices: &[LineVertex],
    lines: impl IntoIterator<Item = &'a Line>,
    scene_info: &SceneInfo,
) {
    let line_vertex_buffer = gpu::Buffer::from_slice(&vertices, "line_vertices");

    let pipeline = get_graphics_pipeline("/shaders/pipelines.parc#lines");
    let Ok(pipeline) = pipeline.read() else {
        return;
    };

    encoder.bind_graphics_pipeline(&pipeline);

    for line in lines {
        encoder.reference_resource(&line_vertex_buffer);
        encoder.draw(
            TriangleStrip,
            0..line.vertex_count * 2,
            0..1,
            &RootParams {
                scene_info: scene_info.gpu,
                vertices: line_vertex_buffer.ptr(),
                line_width: line.width,
                filter_width: line.filter_width,
                start_vertex: line.start_vertex,
                vertex_count: line.vertex_count,
            },
        );
    }
}
