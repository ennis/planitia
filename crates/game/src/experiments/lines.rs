use crate::{SceneInfo, SceneInfoUniforms};
use gamelib::pipeline_cache::get_graphics_pipeline;
use gpu::PrimitiveTopology::{TriangleList, TriangleStrip};
use gpu::{BufferCreateInfo, DrawIndirectCommand, MemoryLocation, Ptr, vk};

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
    vertices: Ptr<LineVertex>,
    lines: Ptr<Line>,
    //start_vertex: u32,
    //vertex_count: u32,
    //line_width: f32,
    //filter_width: f32,
}

/// Draws 3d polylines with given line width in world units.
pub fn draw_lines<'a>(
    encoder: &mut gpu::RenderEncoder,
    vertices: &[LineVertex],
    lines: &[Line],
    scene_info: &SceneInfo,
) {
    let pipeline = get_graphics_pipeline("/shaders/game_shaders.sharc#lines");
    let Ok(pipeline) = pipeline.read() else {
        return;
    };

    let vertices_buffer = gpu::Buffer::from_slice(vertices, "line_vertices");
    let lines_buffer = gpu::Buffer::from_slice(lines, "lines");

    let mut commands = gpu::Buffer::new(BufferCreateInfo {
        len: lines.len(),
        memory_location: MemoryLocation::CpuToGpu,
        ..
    });

    for (i, line) in lines.iter().enumerate() {
        unsafe {
            commands.as_mut_slice()[i].write(DrawIndirectCommand {
                vertex_count: line.vertex_count * 2,
                instance_count: 1,
                first_vertex: line.start_vertex,
                first_instance: 0,
            });
        }
    }

    encoder.bind_graphics_pipeline(&pipeline);
    encoder.draw_indirect(
        TriangleStrip,
        None,
        &commands,
        0..lines.len() as u32,
        gpu::RootParams::Immediate(&RootParams {
            scene_info: scene_info.gpu,
            vertices: vertices_buffer.ptr(),
            lines: lines_buffer.ptr(),
        }),
    );
}
