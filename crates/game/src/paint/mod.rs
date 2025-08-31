mod color;
mod painter;
mod scene;
mod shape;
mod tessellation;
mod atlas;
mod text;

use gpu::prelude::DeviceExt;
use gpu::{CommandStream, RenderPassInfo, Vertex as GpuVertex, vk};
use math::geom::Camera;
use math::{Mat4, Vec2, Vec3};
use shader_bridge::ShaderLibrary;
use std::mem;

use crate::paint::scene::Primitive;
pub use color::Srgba32;
pub use scene::PaintScene;

#[repr(C)]
#[derive(Clone, Copy)]
struct LineVertex {
    position: [f32; 3],
    color: [u8; 4],
    flags: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, GpuVertex)]
pub struct FeatherVertex {
    /// Position
    pub p: Vec2,
    /// Feather factor
    pub feather: f32,
    // Color
    pub color: Srgba32,
}

const SIZE_CHECK: () = assert!(size_of::<FeatherVertex>() == 16);

impl FeatherVertex {
    pub const fn new(p: Vec2, feather: f32, color: Srgba32) -> Self {
        Self { p, feather, color }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct PushConstants {
    matrix: Mat4,
    screen_size: [f32; 2],
    line_width: f32,
}

/// GPU pipelines for drawing.
struct Pipelines {
    paint: gpu::GraphicsPipeline,
}

impl Pipelines {
    fn create(
        device: &gpu::RcDevice,
        target_color_format: gpu::Format,
        target_depth_format: Option<gpu::Format>,
    ) -> Pipelines {
        let shader = ShaderLibrary::new("crates/game/shaders/paint.slang").unwrap();
        let vertex = shader.get_compiled_entry_point("paint_vertex_main").unwrap();
        let fragment = shader.get_compiled_entry_point("paint_fragment_main").unwrap();

        let rasterization_state = gpu::RasterizationState {
            polygon_mode: vk::PolygonMode::FILL,
            cull_mode: Default::default(),
            front_face: vk::FrontFace::CLOCKWISE,
            depth_clamp_enable: true,
            ..Default::default()
        };

        let depth_stencil_state = target_depth_format.map(|format| gpu::DepthStencilState {
            format,
            depth_write_enable: true,
            depth_compare_op: vk::CompareOp::LESS_OR_EQUAL,
            stencil_state: gpu::StencilState::default(),
        });

        let color_targets = [gpu::ColorTargetState {
            format: target_color_format,
            blend_equation: Some(gpu::ColorBlendEquation::ALPHA_BLENDING),
            ..Default::default()
        }];

        // Polygon pipeline
        let create_info = gpu::GraphicsPipelineCreateInfo {
            set_layouts: &[],
            push_constants_size: size_of::<PushConstants>(),
            vertex_input: FeatherVertex::vertex_input_state(),
            pre_rasterization_shaders: gpu::PreRasterizationShaders::PrimitiveShading {
                vertex: vertex.as_gpu_entry_point(),
            },
            rasterization: rasterization_state,
            depth_stencil: depth_stencil_state,
            fragment: gpu::FragmentState {
                shader: fragment.as_gpu_entry_point(),
                multisample: Default::default(),
                color_targets: &color_targets,
                blend_constants: [0.; 4],
            },
        };

        let paint_pipeline = device
            .create_graphics_pipeline(create_info)
            .expect("failed to create pipeline");

        Pipelines { paint: paint_pipeline }
    }
}

//----------------------------------------------------------------

pub struct PaintRenderParams {
    pub camera: Camera,
    pub color_target: gpu::ImageView,
    pub depth_target: Option<gpu::ImageView>,
}

pub struct Painter {
    pipelines: Pipelines,
    color_format: vk::Format,
    depth_format: Option<vk::Format>,
}

impl Painter {
    /// Creates a new painter.
    ///
    /// `target_color_format` and `target_depth_format` specify the formats of the render targets that will be used during rendering.
    pub fn new(
        device: &gpu::RcDevice,
        target_color_format: gpu::Format,
        target_depth_format: Option<gpu::Format>,
    ) -> Painter {
        Painter {
            pipelines: Pipelines::create(device, target_color_format, target_depth_format),
            color_format: target_color_format,
            depth_format: target_depth_format,
        }
    }

    pub fn draw_scene(&self, cmd: &mut CommandStream, scene: PaintScene, params: &PaintRenderParams) {
        assert_eq!(
            params.color_target.format(),
            self.color_format,
            "mismatched color target format"
        );
        assert_eq!(
            params.depth_target.as_ref().map(|d| d.format()),
            self.depth_format,
            "mismatched depth target format"
        );

        // setup encoder
        let mut encoder = cmd.begin_rendering(RenderPassInfo {
            color_attachments: &[gpu::ColorAttachment {
                image_view: &params.color_target,
                clear_value: None,
            }],
            depth_stencil_attachment: params.depth_target.as_ref().map(|d| gpu::DepthStencilAttachment {
                image_view: d,
                depth_clear_value: None,
                stencil_clear_value: None,
            }),
        });

        let width = params.color_target.width();
        let height = params.color_target.height();
        encoder.set_viewport(0.0, height as f32, width as f32, -(height as f32), 0.0, 1.0);
        encoder.set_scissor(0, 0, width, height);
        encoder.bind_graphics_pipeline(&self.pipelines.paint);
        encoder.set_primitive_topology(gpu::vk::PrimitiveTopology::TRIANGLE_LIST);
        encoder.push_constants(&PushConstants {
            matrix: params.camera.view_projection(),
            screen_size: [width as f32, height as f32],
            line_width: 1.0,
        });

        let prims = scene.finish();
        for prim in prims.iter() {
            self.render_prim(&mut encoder, params, prim);
        }
        encoder.finish();
    }

    fn render_prim(&self, encoder: &mut gpu::RenderEncoder, params: &PaintRenderParams, prim: &Primitive) {
        let vertex_buffer = encoder
            .device()
            .upload_slice(gpu::BufferUsage::VERTEX, &prim.mesh.vertices);
        let index_buffer = encoder
            .device()
            .upload_slice(gpu::BufferUsage::INDEX, &prim.mesh.indices);
        encoder.bind_vertex_buffer(0, vertex_buffer.slice(..).untyped);
        encoder.bind_index_buffer(vk::IndexType::UINT32, index_buffer.slice(..).untyped);

        let width = params.color_target.width();
        let height = params.color_target.height();
        let mut clip = prim.clip;
        if clip.is_null() {
            return;
        }

        // Transform clip rect to physical pixels
        let pixels_per_point = 1.0;
        let clip_min_x = ((pixels_per_point * clip.min.x).round() as i32).clamp(0, width as i32);
        let clip_min_y = ((pixels_per_point * clip.min.y).round() as i32).clamp(0, height as i32);
        let clip_max_x = ((pixels_per_point * clip.max.x).round() as i32).clamp(clip_min_x, width as i32);
        let clip_max_y = ((pixels_per_point * clip.max.y).round() as i32).clamp(clip_min_y, height as i32);

        encoder.set_scissor(clip_min_x, clip_min_y, clip_max_x as u32, clip_max_y as u32);
        encoder.draw_indexed(0..prim.mesh.indices.len() as u32, 0, 0..1);
    }
}
