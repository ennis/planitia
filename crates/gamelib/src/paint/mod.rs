mod atlas;
mod painter;
mod scene;
mod shape;
mod tessellation;
mod text;

use crate::paint::shape::RectShape;
use crate::paint::tessellation::{Mesh, Tessellator};
use crate::paint::text::GlyphCache;
use color::Srgba8;
use gpu::{vk, CommandBuffer, Ptr, RootParams, Sampler, Vertex as GpuVertex};
use math::geom::Camera;
use math::{u16vec2, uvec2, vec2, Mat4, Rect, U16Vec2, UVec2, Vec2};
use shader_bridge::ShaderLibrary;

use crate::paint::atlas::Atlas;
use gpu::PrimitiveTopology::TriangleList;
pub use text::{GlyphRun, TextFormat, TextLayout};

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
    /// Texture coordinates
    #[normalized]
    pub uv: U16Vec2,
    /// Color
    pub color: Srgba8,
    /// Feather factor
    pub feather: f32,
}

impl FeatherVertex {
    const SIZE_CHECK: () = assert!(size_of::<Self>() == 16);

    pub const fn new(p: Vec2, uv: U16Vec2, feather: f32, color: Srgba8) -> Self {
        Self { p, feather, color, uv }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct PaintRootParams {
    matrix: Mat4,
    screen_size: [f32; 2],
    line_width: f32,
    texture: gpu::TextureHandle = gpu::TextureHandle::INVALID,
    sampler: gpu::SamplerHandle = gpu::SamplerHandle::INVALID,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct GlyphPushConstants {
    atlas: gpu::TextureHandle,
}

/// GPU pipelines for drawing.
struct Pipelines {
    paint: gpu::GraphicsPipeline,
}

impl Pipelines {
    fn create(target_color_format: gpu::Format, target_depth_format: Option<gpu::Format>) -> Pipelines {
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

//----------------------------------------------------------------

/// Converts a texel coordinate into u16 normalized UV coordinates.
///
/// Equivalent to `pos / texture_size * 65535`.
pub fn texel_to_normalized_texcoord(pos: Vec2, texture_size: UVec2) -> U16Vec2 {
    u16vec2(
        ((pos.x / texture_size.x as f32) * 65535.0) as u16,
        ((pos.y / texture_size.y as f32) * 65535.0) as u16,
    )
}

/// Initializes the paint texture atlas.
///
/// Returns the atlas and the UV coordinate of a white pixel in the atlas.
fn init_atlas() -> (Atlas, U16Vec2) {
    let mut atlas = Atlas::new(1024, 1024);
    // Add a white pixel at (0,0) for drawing solid colors without needing additional logic in the
    // shaders
    let rect = atlas.write(1, 1, &[Srgba8::WHITE], 1, 1);
    let pos = texel_to_normalized_texcoord(
        vec2(rect.min.x as f32 + 0.5, rect.min.y as f32 + 0.5),
        uvec2(atlas.width, atlas.height),
    );
    (atlas, pos)
}

pub struct PaintRenderParams<'a> {
    pub camera: Camera,
    pub color_target: &'a gpu::Image,
    pub depth_target: Option<&'a gpu::Image>,
}

pub struct Painter {
    pipelines: Pipelines,
    texture_atlas: Atlas,
    white_pixel_uv: U16Vec2,
    glyph_cache: GlyphCache,
    sampler: gpu::Sampler,
    color_format: vk::Format,
    depth_format: Option<vk::Format>,
}

impl Painter {
    /// Creates a new painter.
    ///
    /// `target_color_format` and `target_depth_format` specify the formats of the render targets that will be used during rendering.
    pub fn new(target_color_format: gpu::Format, target_depth_format: Option<gpu::Format>) -> Painter {
        let (atlas, white_pixel_uv) = init_atlas();
        let sampler = Sampler::new(gpu::SamplerCreateInfo {
            mag_filter: vk::Filter::LINEAR,
            min_filter: vk::Filter::LINEAR,
            ..
        });

        Painter {
            pipelines: Pipelines::create(target_color_format, target_depth_format),
            color_format: target_color_format,
            depth_format: target_depth_format,
            glyph_cache: Default::default(),
            texture_atlas: atlas,
            white_pixel_uv,
            sampler,
        }
    }

    /// Returns a scene builder.
    pub fn build_scene(&mut self) -> PaintScene<'_> {
        PaintScene::new(self)
    }
}

pub struct Primitive {
    kind: PrimKind,
    clip: Rect,
    texture: gpu::TextureHandle,
}

enum PrimKind {
    Geometry(Mesh),
}

/// Options passed to `PaintScene::draw_glyph_run`.
#[derive(Clone, Debug, Default)]
pub struct DrawGlyphRunOptions {
    pub color: Srgba8,
    pub size: f32,
}

/// Draws shapes onto a target image.
pub struct PaintScene<'a> {
    painter: &'a mut Painter,
    pub(crate) tess: Tessellator,
    prims: Vec<Primitive>,
    clip_stack: Vec<Rect>,
}

impl<'a> PaintScene<'a> {
    pub(super) fn new(painter: &'a mut Painter) -> Self {
        Self {
            painter,
            tess: Tessellator::new(),
            prims: vec![],
            clip_stack: vec![Rect::INFINITE],
        }
    }

    fn end_prim(&mut self, texture: Option<gpu::TextureHandle>) {
        if !self.tess.is_empty() {
            let mesh = self.tess.finish_and_reset();
            let prim = Primitive {
                kind: PrimKind::Geometry(mesh),
                clip: self.clip_stack.last().cloned().unwrap(),
                texture: texture.unwrap_or(self.painter.texture_atlas.texture_handle()),
            };
            self.prims.push(prim);
        }
    }

    /// Draws a rounded rectangle at the specified position with the given size and corner radius.
    pub fn fill_rrect(&mut self, rect: Rect, radius: f32, color: impl Into<Srgba8>) {
        let color = color.into();
        self.tess.fill_rrect(
            RectShape {
                rect,
                radius,
                colors: [color; 4],
                feather: 0.0,
            },
            self.painter.white_pixel_uv,
        );
    }

    fn clip_rect(&self) -> Rect {
        *self.clip_stack.last().unwrap()
    }

    /// Pushes a clip rectangle onto the stack. All subsequent drawing operations will be clipped to this rectangle.
    pub fn push_clip(&mut self, rect: Rect) {
        self.end_prim(None);
        let clip = self.clip_rect().intersect(&rect).unwrap_or_default();
        self.clip_stack.push(clip);
    }

    /// Pops the last clip rectangle from the stack.
    pub fn pop_clip(&mut self) {
        self.end_prim(None);
        self.clip_stack.pop();
    }

    /// Draws a glyph run.
    pub fn draw_glyph_run(&mut self, position: Vec2, glyph_run: &GlyphRun<'_>, options: &DrawGlyphRunOptions) {
        self.end_prim(None);

        let format = glyph_run.format();
        let x = glyph_run.offset();
        let y = glyph_run.baseline();
        let mut advance = 0.0;
        for glyph in glyph_run.glyphs() {
            //eprintln!(
            //    "   glyph id={} advance={} (x={})",
            //    glyph.id.0,
            //    glyph.advance,
            //    x + advance
            //);

            let pos = position + vec2(x + advance, y) + glyph.offset;
            //debug!("glyph_offset = {:?}", glyph.offset);
            advance += glyph.advance;

            let (entry, quantized_pos) = self.painter.glyph_cache.rasterize_glyph(
                &mut self.painter.texture_atlas,
                &format.font,
                glyph.id,
                format.size as u32,
                pos,
            );

            if entry.px_bounds.is_null() {
                //eprintln!("    skipping empty glyph");
                continue;
            }

            let quad = entry.px_bounds.to_rect().translate(quantized_pos);
            let uv0 = entry.normalized_texcoords[0];
            let uv1 = entry.normalized_texcoords[1];
            //eprintln!("    glyph {:?} quad={:?} uv0={:?} uv1={:?} tex={:?}", glyph.id, quad, uv0, uv1, self.painter.glyph_cache.texture_handle());
            self.tess.quad(quad.min, quad.max, uv0, uv1, Srgba8::WHITE);
        }

        self.end_prim(None);
    }

    pub fn finish(mut self, cmd: &mut CommandBuffer, params: &PaintRenderParams) {
        self.end_prim(None);
        self.draw_inner(cmd, params);
    }

    fn draw_inner(mut self, cmd: &mut CommandBuffer, params: &PaintRenderParams) {
        assert_eq!(
            params.color_target.format(),
            self.painter.color_format,
            "mismatched color target format"
        );
        assert_eq!(
            params.depth_target.as_ref().map(|d| d.format()),
            self.painter.depth_format,
            "mismatched depth target format"
        );

        // prepare texture atlas
        let _atlas = self.painter.texture_atlas.prepare_texture(cmd);

        // setup encoder
        let mut encoder = cmd.begin_rendering(
            &[gpu::ColorAttachment {
                image: &params.color_target,
                clear_value: None,
            }],
            params.depth_target.as_ref().map(|d| gpu::DepthStencilAttachment {
                image: d,
                depth_clear_value: None,
                stencil_clear_value: None,
            }));

        let width = params.color_target.width();
        let height = params.color_target.height();
        encoder.set_viewport(0.0, 0.0, width as f32, height as f32, 0.0, 1.0);
        encoder.set_scissor(0, 0, width, height);
        encoder.bind_graphics_pipeline(&self.painter.pipelines.paint);

        for prim in self.prims.iter() {
            if prim.clip.is_null() {
                continue;
            }

            match &prim.kind {
                PrimKind::Geometry(mesh) => {
                    let root_params = encoder.upload(&PaintRootParams {
                        matrix: params.camera.view_projection(),
                        screen_size: [width as f32, height as f32],
                        line_width: 1.0,
                        texture: prim.texture,
                        sampler: self.painter.sampler.device_handle(),
                    });
                    draw_mesh(&mut encoder, params, mesh, prim.clip, root_params);
                }
            }
        }
        encoder.finish();
    }
}

fn set_scissor(encoder: &mut gpu::RenderEncoder, params: &PaintRenderParams, clip: Rect) {
    let width = params.color_target.width();
    let height = params.color_target.height();

    // Transform clip rect to physical pixels
    let pixels_per_point = 1.0;
    let clip_min_x = ((pixels_per_point * clip.min.x).round() as i32).clamp(0, width as i32);
    let clip_min_y = ((pixels_per_point * clip.min.y).round() as i32).clamp(0, height as i32);
    let clip_max_x = ((pixels_per_point * clip.max.x).round() as i32).clamp(clip_min_x, width as i32);
    let clip_max_y = ((pixels_per_point * clip.max.y).round() as i32).clamp(clip_min_y, height as i32);

    encoder.set_scissor(clip_min_x, clip_min_y, clip_max_x as u32, clip_max_y as u32);
}
/*
fn draw_glyph_run(
    encoder: &mut gpu::RenderEncoder,
    params: &PaintRenderParams,
    vertices: &[GlyphVertex],
    clip: Rect,
) {
    let vertex_buffer = encoder.device().upload_slice(gpu::BufferUsage::VERTEX, vertices);
    encoder.bind_vertex_buffer(0, vertex_buffer.as_bytes().into());
    set_scissor(encoder, params, clip);
    encoder.draw(0..vertices.len() as u32, 0..1);
}*/

fn draw_mesh(
    encoder: &mut gpu::RenderEncoder,
    params: &PaintRenderParams,
    mesh: &Mesh,
    clip: Rect,
    root_params: Ptr<PaintRootParams>,
) {
    if mesh.vertices.is_empty() || mesh.indices.is_empty() {
        return;
    }
    let vertex_buffer = gpu::Buffer::from_slice(&mesh.vertices);
    let index_buffer = gpu::Buffer::from_slice(&mesh.indices);
    set_scissor(encoder, params, clip);
    encoder.draw_indexed(
        TriangleList,
        &index_buffer,
        0..mesh.indices.len() as u32,
        Some(vertex_buffer.as_bytes()),
        0,
        0..1,
        RootParams::Ptr(root_params),
    );
}

#[cfg(test)]
mod test {}
