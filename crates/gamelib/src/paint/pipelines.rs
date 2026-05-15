use crate::paint::PaintVertex;
use crate::paint::render::PaintRootParams;
use gpu::{Vertex, vk};
use shader_bridge::ShaderLibrary;

/// Painter pipelines.
pub(super) struct Pipelines {
    pub(super) paint: gpu::GraphicsPipeline,
}


impl Pipelines {
    /// Creates the pipelines from the shaders.
    ///
    /// # Arguments
    /// * `target_color_format` format of the color attachment to render to
    /// * `target_depth_format` format of the depth attachment to render to
    pub(super) fn create(
        target_color_format: gpu::Format,
        target_depth_format: Option<gpu::Format>,
    ) -> Pipelines {
        // TODO use asset system, and replace with embedded pipeline archive
        let shader = ShaderLibrary::new("crates/gamelib/assets/gamelib/shaders/paint.slang").unwrap();
        let vertex = shader.get_compiled_entry_point("paint_vertex_main").unwrap();
        let fragment = shader.get_compiled_entry_point("paint_fragment_main").unwrap();
        //let glyph_shader = ShaderLibrary::new("crates/game/shaders/paint_glyphs.slang").unwrap();
        //let glyph_vertex = glyph_shader.get_compiled_entry_point("glyph_vertex_main").unwrap();
        //let glyph_fragment = glyph_shader.get_compiled_entry_point("glyph_fragment_main").unwrap();

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
            push_constants_size: size_of::<PaintRootParams>(),
            vertex_input: PaintVertex::vertex_input_state(),
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

        let paint_pipeline = gpu::GraphicsPipeline::new(create_info).expect("failed to create pipeline");

        /*// Glyph pipeline
        let create_info = gpu::GraphicsPipelineCreateInfo {
            set_layouts: &[],
            push_constants_size: size_of::<PushConstants>(),
            vertex_input: GlyphVertex::vertex_input_state(),
            pre_rasterization_shaders: gpu::PreRasterizationShaders::PrimitiveShading {
                vertex: glyph_vertex.as_gpu_entry_point(),
            },
            rasterization: rasterization_state,
            depth_stencil: depth_stencil_state,
            fragment: gpu::FragmentState {
                shader: glyph_fragment.as_gpu_entry_point(),
                multisample: Default::default(),
                color_targets: &color_targets,
                blend_constants: [0.; 4],
            },
        };

        let glyph_pipeline = device
            .create_graphics_pipeline(create_info)
            .expect("failed to create glyph pipeline");*/


        Pipelines { paint: paint_pipeline }
    }
}
