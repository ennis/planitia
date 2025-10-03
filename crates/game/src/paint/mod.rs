mod atlas;
mod color;
mod painter;
mod scene;
mod shape;
mod tessellation;
mod text;

use gpu::prelude::DeviceExt;
use gpu::{CommandStream, RenderPassInfo, Vertex as GpuVertex, vk};
use math::geom::Camera;
use math::{Mat4, Rect, U16Vec2, Vec2, Vec3};
use shader_bridge::ShaderLibrary;
use std::mem;

use crate::paint::shape::RectShape;
use crate::paint::tessellation::{Mesh, Tessellator};
use crate::paint::text::{Font, Glyph, GlyphCache};
pub use color::Srgba32;

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
    pub uv: U16Vec2,
    /// Color
    pub color: Srgba32,
    /// Feather factor
    pub feather: f32,
}

impl FeatherVertex {
    const SIZE_CHECK: () = assert!(size_of::<Self>() == 16);

    pub const fn new(p: Vec2, feather: f32, color: Srgba32) -> Self {
        Self {
            p,
            feather,
            color,
            uv: U16Vec2::new(0, 0),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, GpuVertex)]
pub struct GlyphVertex {
    pub p: Vec2,
    pub uv: U16Vec2,
    pub color: Srgba32,
}

impl GlyphVertex {
    const SIZE_CHECK: () = assert!(size_of::<Self>() == 20);

    pub const fn new(p: Vec2, uv: U16Vec2, color: Srgba32) -> Self {
        Self { p, uv, color }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct PushConstants {
    matrix: Mat4,
    screen_size: [f32; 2],
    line_width: f32,
    texture: gpu::ImageHandle = gpu::ImageHandle::INVALID,
    sampler: gpu::SamplerHandle = gpu::SamplerHandle::INVALID,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct GlyphPushConstants {
    atlas: gpu::ImageHandle,
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

pub struct PaintRenderParams<'a> {
    pub camera: Camera,
    pub color_target: &'a gpu::Image,
    pub depth_target: Option<&'a gpu::Image>,
}

pub struct Painter {
    pipelines: Pipelines,
    glyph_cache: GlyphCache,
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
            glyph_cache: Default::default(),
        }
    }

    /// Returns a scene builder.
    pub fn build_scene(&mut self) -> PaintScene<'_> {
        PaintScene::new(self)
    }

    // to draw text, we need the glyph cache and the font collection; the signature will look like this:
    //
    //      fn draw_text(&self, font_collection: &mut FontCollection, glyph_cache: &mut GlyphCache, &FormattedText)
    //
    // Alternatively, the painter could have its own glyph_cache.
    // Also, the font collection would not be necessary if formatted_text holds Arcs to the fonts instead of IDs in a collection.
    // -> do this instead
    // -> then `FontCollection` can be removed entirely (eventually it could be something that handles font resolution)
    // -> FormattedText can be measured without any dependency
    // -> the glyph cache can be made global (shared between multiple painters), but I'm not sure
    //    that there will ever be multiple painters in practice
}

pub struct Primitive {
    kind: PrimKind,
    clip: Rect,
}

enum PrimKind {
    Geometry(Mesh),
}

/// Options passed to `PaintScene::draw_glyph_run`.
#[derive(Clone)]
pub struct DrawGlyphRunOptions {
    pub color: Srgba32,
    pub size: f32,
    pub font: Font,
}

/// Draws shapes onto a target image.
pub struct PaintScene<'a> {
    painter: &'a mut Painter,
    pub(crate) tess: Tessellator,
    prims: Vec<Primitive>,
    clip_stack: Vec<Rect>,
}

fn textured_quad(p0: Vec2, p1: Vec2, uv0: U16Vec2, uv1: U16Vec2, color: Srgba32) -> [FeatherVertex; 6] {
    [
        FeatherVertex {
            p: Vec2::new(p0.x, p0.y),
            uv: U16Vec2::new(uv0.x, uv0.y),
            color,
            feather: 0.0,
        },
        FeatherVertex {
            p: Vec2::new(p1.x, p0.y),
            uv: U16Vec2::new(uv1.x, uv0.y),
            color,
            feather: 0.0,
        },
        FeatherVertex {
            p: Vec2::new(p1.x, p1.y),
            uv: U16Vec2::new(uv1.x, uv1.y),
            color,
            feather: 0.0,
        },
        FeatherVertex {
            p: Vec2::new(p0.x, p0.y),
            uv: U16Vec2::new(uv0.x, uv0.y),
            color,
            feather: 0.0,
        },
        FeatherVertex {
            p: Vec2::new(p1.x, p1.y),
            uv: U16Vec2::new(uv1.x, uv1.y),
            color,
            feather: 0.0,
        },
        FeatherVertex {
            p: Vec2::new(p0.x, p1.y),
            uv: U16Vec2::new(uv0.x, uv1.y),
            color,
            feather: 0.0,
        },
    ]
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

    fn end_prim(&mut self) {
        if !self.tess.is_empty() {
            let mesh = self.tess.finish_and_reset();
            let prim = Primitive {
                kind: PrimKind::Geometry(mesh),
                clip: self.clip_stack.pop().unwrap(),
            };
            self.prims.push(prim);
        }
    }

    /// Draws a rounded rectangle at the specified position with the given size and corner radius.
    pub fn fill_rrect(&mut self, rect: Rect, radius: f32, color: impl Into<Srgba32>) {
        let color = color.into();
        self.tess.fill_rrect(RectShape {
            rect,
            radius,
            colors: [color; 4],
            feather: 0.0,
        });
    }

    fn clip_rect(&self) -> Rect {
        *self.clip_stack.last().unwrap()
    }

    /// Pushes a clip rectangle onto the stack. All subsequent drawing operations will be clipped to this rectangle.
    pub fn push_clip(&mut self, rect: Rect) {
        self.end_prim();
        let clip = self.clip_rect().intersect(&rect).unwrap_or_default();
        self.clip_stack.push(clip);
    }

    /// Pops the last clip rectangle from the stack.
    pub fn pop_clip(&mut self) {
        self.end_prim();
        self.clip_stack.pop();
    }

    /// Draws a glyph run.
    pub fn draw_glyph_run(
        &mut self,
        position: Vec2,
        glyphs: impl Iterator<Item = Glyph>,
        options: &DrawGlyphRunOptions,
    ) {
        self.end_prim();

        let mut vertices = Vec::new();
        for glyph in glyphs {
            let entry = self
                .painter
                .glyph_cache
                .rasterize_glyph(&options.font, glyph.id, options.size as u32);
            if entry.px_bounds.is_null() {
                continue;
            }

            let pos = position + glyph.offset;
            let quad = entry.px_bounds.to_rect().translate(pos);
            let tex_rect = entry.texture_rect();
            let uv0 = U16Vec2::new(tex_rect.min.x as u16, tex_rect.min.y as u16);
            let uv1 = U16Vec2::new(tex_rect.max.x as u16, tex_rect.max.y as u16);
            vertices.extend(textured_quad(quad.min, quad.max, uv0, uv1, options.color));
        }

        /*self.prims.push(Primitive {
            kind: PrimKind::GlyphRun { mesh: vertices },
            clip: self.clip_rect(),
        });*/
    }

    pub fn finish(mut self, cmd: &mut CommandStream, params: &PaintRenderParams) {
        self.end_prim();
        self.draw_inner(cmd, params);
    }

    fn draw_inner(mut self, cmd: &mut CommandStream, params: &PaintRenderParams) {
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

        // setup encoder
        let mut encoder = cmd.begin_rendering(RenderPassInfo {
            color_attachments: &[gpu::ColorAttachment {
                image: &params.color_target,
                clear_value: None,
            }],
            depth_stencil_attachment: params.depth_target.as_ref().map(|d| gpu::DepthStencilAttachment {
                image: d,
                depth_clear_value: None,
                stencil_clear_value: None,
            }),
        });

        let width = params.color_target.width();
        let height = params.color_target.height();
        encoder.set_viewport(0.0, height as f32, width as f32, -(height as f32), 0.0, 1.0);
        encoder.set_scissor(0, 0, width, height);
        encoder.bind_graphics_pipeline(&self.painter.pipelines.paint);
        encoder.set_primitive_topology(gpu::vk::PrimitiveTopology::TRIANGLE_LIST);

        for prim in self.prims.iter() {
            if prim.clip.is_null() {
                return;
            }

            match &prim.kind {
                PrimKind::Geometry(mesh) => {
                    encoder.push_constants(&PushConstants {
                        matrix: params.camera.view_projection(),
                        screen_size: [width as f32, height as f32],
                        line_width: 1.0,
                        ..
                    });
                    draw_mesh(&mut encoder, params, mesh, prim.clip);
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

fn draw_mesh(encoder: &mut gpu::RenderEncoder, params: &PaintRenderParams, mesh: &Mesh, clip: Rect) {
    let vertex_buffer = encoder.device().upload_slice(gpu::BufferUsage::VERTEX, &mesh.vertices);
    let index_buffer = encoder.device().upload_slice(gpu::BufferUsage::INDEX, &mesh.indices);
    encoder.bind_vertex_buffer(0, vertex_buffer.as_bytes().into());
    encoder.bind_index_buffer(vk::IndexType::UINT32, index_buffer.as_bytes().into());
    set_scissor(encoder, params, clip);
    encoder.draw_indexed(0..mesh.indices.len() as u32, 0, 0..1);
}

#[cfg(test)]
mod test {}
