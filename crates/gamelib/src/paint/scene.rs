use crate::paint::fill::Fill;
use crate::paint::flatten::flatten_path;
use crate::paint::gradient::ColorStop;
use crate::paint::pipelines::PaintRootParams;
use crate::paint::renderer::DrawCommand;
use crate::paint::tessellation::{Mesh, Tessellator};
use crate::paint::{
    AsPathSlice, GlyphRun, GradientRampData, GradientIntegralSegment, PaintRenderParams, PaintVertex, Painter, Path,
    PathBuilder, PathSlice, RRect, compute_gradient_integral, renderer,
};
use color::{Srgba8, srgba8};
use gpu::PrimitiveTopology::TriangleList;
use gpu::{CommandBuffer, Ptr, PushDataSource};
use log::{error, warn};
use math::{Mat3, Rect, Vec2, Vec3, Vec4, rect_transform, vec2, vec3};

/// Options for drawing a glyph run.
#[derive(Clone, Debug, Default)]
pub struct DrawGlyphRunOptions {
    /// Glyph color.
    pub color: Srgba8,
    /// Glyph size in points.
    pub size: f32,
}

/// Represents a gradient ramp (a list of color stops).
///
/// Gradient ramps are valid only while a scene is being built.
/// Storing a [`GradientRamp`] between scenes is not allowed and may lead to undefined behavior.
pub type GradientRamp = usize;

/// Representation of a painter scene being drawn.
pub struct PaintScene<'a> {
    painter: &'a mut Painter,
    pub(crate) tess: Tessellator,
    old_scene: RenderScene,
    clip_stack: Vec<Rect>,
    transform_stack: Vec<Mat3>,
    transform: Mat3,
    scene: renderer::Scene,
}

impl<'a> PaintScene<'a> {
    pub(super) fn new(painter: &'a mut Painter) -> Self {
        Self {
            painter,
            tess: Tessellator::new(),
            old_scene: RenderScene::default(),
            clip_stack: vec![Rect::INFINITE],
            transform_stack: vec![],
            transform: Default::default(),
            scene: renderer::Scene::default(),
        }
    }

    fn push_draw_op(&mut self, fill: impl Into<Fill>) {
        let fill = fill.into();
        let device_to_local = Mat3::IDENTITY; // TODO transform stack
        self.old_scene.ops.push(DrawOp {
            mesh: self.tess.finish_and_reset(),
            clip: self.clip_rect(),
            device_to_local,
            fill,
        });
    }

    /// Creates a [`GradientRamp`] from the specified list of color stops.
    ///
    /// The returned [`GradientRamp`] is valid only for this scene.
    pub fn create_gradient_ramp(&mut self, stops: &[ColorStop]) -> GradientRamp {
        let ramp = compute_gradient_integral(stops, &mut self.scene.gradient_integral_segments);
        self.scene.gradient_ramps.push(ramp);
        self.scene.gradient_ramps.len() - 1
    }

    /// Draws a rounded rectangle at the specified position with the given size and corner radius.
    pub fn fill_rrect(&mut self, rect: Rect, radius: f32, fill: impl Into<Fill>) {
        let fill = fill.into().transform(&self.transform);
        self.tess.fill_rrect(RRect { rect, radius });
        self.push_draw_op(fill);
    }

    /// Fills a circle.
    ///
    /// # Arguments
    ///
    /// - `center` the center of the circle
    /// - `radius` the radius of the circle
    /// - `fill` how to fill the circle
    pub fn fill_circle(&mut self, center: Vec2, radius: f32, fill: impl Into<Fill>) {
        // Approximate the circle with cubic Bézier curves.
        // Four arcs of 90°, each controlled by the "magic" constant k = 4*(sqrt(2)-1)/3.
        const K: f32 = 0.5522847498;
        let r = radius;
        let k = K * r;
        let mut b = PathBuilder::new();
        b.move_to(center + vec2(r, 0.0));
        b.cubic_to(center + vec2(r, k), center + vec2(k, r), center + vec2(0.0, r));
        b.cubic_to(center + vec2(-k, r), center + vec2(-r, k), center + vec2(-r, 0.0));
        b.cubic_to(center + vec2(-r, -k), center + vec2(-k, -r), center + vec2(0.0, -r));
        b.cubic_to(center + vec2(k, -r), center + vec2(r, -k), center + vec2(r, 0.0));
        b.close();
        self.fill_path(&b.finish(), fill);
    }

    /// Fills a path.
    ///
    /// # Arguments
    ///
    /// - `path` the path to fill
    /// - `fill` how to fill the path
    pub fn fill_path(&mut self, path: &impl AsPathSlice, fill: impl Into<Fill>) {
        let path = path.as_path_slice();
        let base_vertex = self.scene.path_points.len();
        let base_verb = self.scene.path_verbs.len();
        self.scene.path_points.extend_from_slice(&path.points);
        self.scene.path_verbs.extend_from_slice(&path.verbs);

        let fill_index = self.scene.fills.len();
        // Fills may contain local coordinates (e.g. gradient lines). Transform them to screen space
        // now so that we don't have to worry about it later.
        let fill = fill.into().transform(&self.transform);
        self.scene.fills.push(fill);

        self.scene.draw_commands.push(DrawCommand::FillPath {
            verb_range: base_verb..self.scene.path_verbs.len(),
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
        self.scene.draw_commands.push(DrawCommand::SetTransform(self.transform));
    }

    pub fn transform(&mut self, transform: Mat3) {
        self.transform = self.transform * transform;
        self.scene.draw_commands.push(DrawCommand::SetTransform(self.transform));
    }

    pub fn draw_line(&mut self, p0: Vec2, p1: Vec2, width: f32, fill: impl Into<Fill>) {
        let fill = fill.into();
        self.tess.stroke_line(p0, p1, width);
        self.push_draw_op(fill);
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
            //let uv0 = entry.normalized_texcoords[0];
            //let uv1 = entry.normalized_texcoords[1];
            //eprintln!("    glyph {:?} quad={:?} uv0={:?} uv1={:?} tex={:?}", glyph.id, quad, uv0, uv1, self.painter.glyph_cache.texture_handle());
            self.tess.quad(quad.min, quad.max);
            self.push_draw_op(Fill::Texture {
                texture: self.painter.texture_atlas.texture_handle(),
                sampler: self.painter.sampler.device_handle(),
                local_to_uv: rect_transform(quad, Rect::from_min_max(entry.uv[0], entry.uv[1])),
            });
        }
    }

    pub fn finish(mut self, cmd: &mut CommandBuffer, params: &PaintRenderParams) {
        if let Err(err) = renderer::render_scene(cmd, self.painter, params, &self.scene) {
            error!("failed to render scene: {err}");
        }
        render_scene(cmd, self.painter, params, &self.old_scene);
    }
}

//--------------------------------------------------------------------------------------------------

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
