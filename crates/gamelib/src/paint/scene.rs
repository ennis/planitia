use crate::paint::fill::Fill;
use crate::paint::flatten::flatten_path;
use crate::paint::gradient::ColorStop;
use crate::paint::renderer::DrawCommand;
use crate::paint::tessellation::{Mesh, Tessellator};
use crate::paint::{AsPathSlice, GlyphRun, GradientIntegralSegment, GradientRampData, PaintRenderParams, PaintVertex, Painter, Path, PathBuilder, PathSlice, RRect, TextureFill, compute_gradient_integral, renderer, TextFormat, TextLayout};
use color::{Srgba8, srgba8};
use gpu::PrimitiveTopology::TriangleList;
use gpu::{CommandBuffer, Ptr, PushDataSource};
use log::{error, warn};
use math::{Mat3, Rect, Vec2, Vec3, Vec4, rect_transform, vec2, vec3};
use std::mem;
use ron::de::Position;

/// Options for drawing a glyph run.
#[derive(Clone, Debug, Default)]
pub struct DrawGlyphRunOptions {
    /// Glyph color.
    pub color: Srgba8,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[repr(u8)]
pub enum BlendMode {
    Normal,
    Multiply,
    Screen,
    Overlay,
    Darken,
    Lighten,
    ColorDodge,
    ColorBurn,
    HardLight,
    SoftLight,
    Difference,
    Exclusion,
    Mask,
}

#[derive(Clone, Copy, Debug)]
pub struct GroupOptions {
    pub blend_mode: BlendMode,
    pub opacity: f32,
}

/// Represents a gradient ramp (a list of color stops).
///
/// Gradient ramps are valid only while a scene is being built.
/// Storing a [`GradientRamp`] between scenes is not allowed and may lead to undefined behavior.
pub type GradientRamp = usize;

/// Representation of a painter scene being drawn.
pub struct PaintScene<'a> {
    painter: &'a mut Painter,
    clip_stack: Vec<Rect>,
    transform_stack: Vec<Mat3>,
    transform: Mat3,
    tmp_path: PathBuilder,
    rscene: renderer::Scene,
}

impl<'a> PaintScene<'a> {
    pub(super) fn new(painter: &'a mut Painter, clear_color: Srgba8) -> Self {
        Self {
            painter,
            clip_stack: vec![Rect::INFINITE],
            transform_stack: vec![],
            transform: Default::default(),
            rscene: renderer::Scene::new(clear_color),
            tmp_path: PathBuilder::new(),
        }
    }

    /// Creates a [`GradientRamp`] from the specified list of color stops.
    ///
    /// The returned [`GradientRamp`] is valid only for this scene.
    pub fn create_gradient_ramp(&mut self, stops: &[ColorStop]) -> GradientRamp {
        let ramp = compute_gradient_integral(stops, &mut self.rscene.gradient_integral_segments);
        self.rscene.gradient_ramps.push(ramp);
        self.rscene.gradient_ramps.len() - 1
    }

    /// Clears the drawing area with the specified color.
    pub fn clear(&mut self, color: Srgba8) {
        self.rscene.draw_commands.push(DrawCommand::Clear(color));
    }


    /// Draws a rounded rectangle at the specified position with the given size and corner radius.
    pub fn fill_rrect(&mut self, rect: Rect, radius: f32, fill: impl Into<Fill>) {
        self.tmp_path.rrect(&rect, vec2(radius, radius));
        self.fill_tmp_path(fill);
    }

    /// Fills a rectangle.
    pub fn fill_rect(&mut self, rect: Rect, fill: impl Into<Fill>) {
        self.tmp_path.rect(&rect);
        self.fill_tmp_path(fill);
    }

    /// Fills a circle.
    pub fn fill_circle(&mut self, center: Vec2, radius: f32, fill: impl Into<Fill>) {
        self.tmp_path.circle(center, radius);
        self.fill_tmp_path(fill);
    }

    fn fill_tmp_path(&mut self, fill: impl Into<Fill>) {
        let path = mem::take(&mut self.tmp_path);
        self.fill_path(&path, fill);
        self.tmp_path = path;
        self.tmp_path.clear();
    }

    /// Fills a path.
    pub fn fill_path(&mut self, path: &impl AsPathSlice, fill: impl Into<Fill>) {
        let path = path.as_path_slice();
        let base_vertex = self.rscene.path_points.len();
        let base_verb = self.rscene.path_verbs.len();
        self.rscene.path_points.extend_from_slice(&path.points);
        self.rscene.path_verbs.extend_from_slice(&path.verbs);

        let fill_index = self.rscene.register_fill(&fill.into(), &self.transform);
        self.rscene.draw_commands.push(DrawCommand::FillPath {
            verb_range: base_verb..self.rscene.path_verbs.len(),
            base_vertex,
            fill: fill_index,
        });
    }

    pub fn save(&mut self) {
        self.transform_stack.push(self.transform);
    }

    pub fn restore(&mut self) {
        if let Some(transform) = self.transform_stack.pop() {
            self.transform = transform;
        } else {
            warn!("restore() called without a matching save()");
        }
        self.rscene.draw_commands.push(DrawCommand::SetTransform(self.transform));
    }

    pub fn transform(&mut self, transform: Mat3) {
        self.transform = self.transform * transform;
        self.rscene.draw_commands.push(DrawCommand::SetTransform(self.transform));
    }

    pub fn draw_line(&mut self, p0: Vec2, p1: Vec2, width: f32, fill: impl Into<Fill>) {
        let mut builder = PathBuilder::new();
        builder.move_to(p0);
        builder.line_to(p1);
        self.fill_path(&builder.finish(), fill);
    }

    fn clip_rect(&self) -> Rect {
        *self.clip_stack.last().unwrap()
    }

    /// Pushes a clip rectangle onto the stack. All subsequent drawing operations will be clipped to this rectangle.
    pub fn push_clip(&mut self, rect: Rect) {
        let clip = self.clip_rect().intersect(&rect).unwrap_or_default();
        self.clip_stack.push(clip);
    }

    /// Pops the last clip rectangle from the stack.
    pub fn pop_clip(&mut self) {
        self.clip_stack.pop();
    }

    /// Begins a new group.
    pub fn push_group(&mut self, group_options: &GroupOptions) {
        //self.rscene.draw_commands.push(DrawCommand::BeginGroup);
    }

    ///
    pub fn pop_group(&mut self) {}

    /// Draws text at the specified position with the given format and color.
    pub fn draw_text(&mut self, position: Vec2, text: &str, format: &TextFormat, color: Srgba8) {
        let mut layout = TextLayout::new(format, text);
        layout.layout(1000.0);
        for glyph_run in layout.glyph_runs() {
            self.draw_glyph_run(position, &glyph_run, &DrawGlyphRunOptions { color });
        }
    }

    /// Draws a glyph run.
    pub fn draw_glyph_run(&mut self, position: Vec2, glyph_run: &GlyphRun<'_>, options: &DrawGlyphRunOptions) {
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
            self.fill_rect(
                quad,
                TextureFill {
                    texture: self.painter.texture_atlas.texture_handle(),
                    sampler: self.painter.sampler.device_handle(),
                    local_to_uv: rect_transform(quad, Rect::from_min_max(entry.uv[0], entry.uv[1])),
                    color: options.color,
                },
            );
        }
    }

    pub fn finish(mut self, cmd: &mut CommandBuffer, params: &PaintRenderParams) {
        let _atlas = self.painter.texture_atlas.prepare_texture(cmd);
        if let Err(err) = renderer::render_scene(cmd, self.painter, params, &self.rscene) {
            error!("failed to render scene: {err}");
        }
    }
}

//--------------------------------------------------------------------------------------------------

/*
struct DrawOp {
    mesh: Mesh,
    clip: Rect,
    device_to_local: Mat3,
    fill: Fill,
}

struct RenderScene {
    vertices: Vec<PaintVertex>,
    indices: Vec<u32>,
    ops: Vec<DrawOp>,
}

impl Default for RenderScene {
    fn default() -> Self {
        Self { vertices: vec![], indices: vec![], ops: vec![] }
    }
}

fn render_scene(cmd: &mut CommandBuffer, painter: &mut Painter, params: &PaintRenderParams, scene: &RenderScene) {
    assert_eq!(params.color_target.format(), painter.color_format, "mismatched color target format");
    assert_eq!(
        params.depth_target.as_ref().map(|d| d.format()),
        painter.depth_format,
        "mismatched depth target format"
    );

    // prepare texture atlas
    let _atlas = painter.texture_atlas.prepare_texture(cmd);

    // setup encoder
    let mut encoder = cmd.begin_rendering(
        &[gpu::ColorAttachment { image: &params.color_target, clear: None }],
        params.depth_target.as_ref().map(|d| gpu::DepthStencilAttachment {
            image: d,
            depth_clear: None,
            stencil_clear: None,
        }),
    );

    let width = params.color_target.width();
    let height = params.color_target.height();
    encoder.set_viewport(0.0, 0.0, width as f32, height as f32, 0.0, 1.0);
    encoder.set_scissor(0, 0, width, height);
    encoder.bind_graphics_pipeline(&painter.pipelines.paint);

    for prim in scene.ops.iter() {
        if prim.clip.is_null() {
            continue;
        }

        let texture;
        let device_to_uv_transform;
        let mut color = Srgba8::WHITE;

        match prim.fill {
            Fill::Solid(solid_color) => {
                texture = painter.texture_atlas.texture_handle();
                device_to_uv_transform = Mat3::from_cols(
                    Vec3::ZERO,
                    Vec3::ZERO,
                    vec3(painter.white_pixel_uv_f.x, painter.white_pixel_uv_f.y, 1.0),
                );
                color = solid_color;
            }
            Fill::Texture { texture: tex, sampler, local_to_uv: uv_transform } => {
                texture = tex;
                device_to_uv_transform = prim.device_to_local * uv_transform;
            }
            _ => {
                // TODO dummy
                texture = painter.texture_atlas.texture_handle();
                device_to_uv_transform = Mat3::from_cols(
                    Vec3::ZERO,
                    Vec3::ZERO,
                    vec3(painter.white_pixel_uv_f.x, painter.white_pixel_uv_f.y, 1.0),
                );
            }
        };

        let root_params = encoder.upload(&PaintRootParams {
            screen_size: [width as f32, height as f32],
            line_width: 1.0,
            device_to_uv_transform,
            texture,
            sampler: painter.sampler.device_handle(),
            color,
        });
        draw_mesh(&mut encoder, params, &prim.mesh, prim.clip, root_params);
    }
    encoder.finish();
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
        PushDataSource::Indirect(root_params),
    );
}
*/
