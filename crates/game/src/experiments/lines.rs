use crate::{SceneInfo, SceneInfoUniforms};
use gamelib::{static_assets, tweak};
use gpu::PrimitiveTopology::TriangleStrip;
use gpu::{BufferCreateInfo, DrawIndirectCommand, MemoryLocation, Ptr};

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
    static_assets! {
        static LINES_PIPELINE: gpu::GraphicsPipeline = "/shaders/game_shaders.sharc#draw_lines";
    }

    let Ok(pipeline) = LINES_PIPELINE.read() else {
        return;
    };

    let vertices_buffer = gpu::Buffer::from_slice(vertices);
    let lines_buffer = gpu::Buffer::from_slice(lines);

    // use indirect draws to reduce the overhead a bit when drawing many (~1000+) lines
    let mut commands = gpu::Buffer::new(BufferCreateInfo {
        len: lines.len(),
        memory_location: MemoryLocation::CpuToGpu,
        ..
    });
    for (i, line) in lines.iter().enumerate() {
        unsafe {
            // SAFETY: we have exclusive access to the buffer, the GPU is not using it right now
            commands.as_mut_slice()[i].write(DrawIndirectCommand {
                // one quad per line segment, so 2 vertices per line vertex
                vertex_count: line.vertex_count * 2,
                instance_count: 1,
                first_vertex: line.start_vertex,
                first_instance: 0,
            });
        }
    }

    encoder.bind_graphics_pipeline(&pipeline);
    encoder.set_depth_bias(Some(gpu::DepthBias {
        constant_factor: tweak!(line_depth_bias_constant: f32 = 1.0),
        slope_factor: tweak!(line_depth_bias_slope: f32 = 1.0),
        clamp: 0.0,
    }));
    encoder.draw_indirect(
        TriangleStrip,
        None,
        &commands,
        0..lines.len() as u32,
        &RootParams {
            scene_info: scene_info.gpu,
            vertices: vertices_buffer.ptr(),
            lines: lines_buffer.ptr(),
        },
    );
}
