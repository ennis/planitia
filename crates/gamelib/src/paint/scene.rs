use crate::app::get_context;
use crate::paint::fill::Fill;
use crate::paint::flatten::flatten_path;
use crate::paint::gradient::ColorStop;
use crate::paint::renderer::DrawCommand;
use crate::paint::text::Glyph;
use crate::paint::{
    AsPathSlice, BlendMode, GlyphRun, GradientIntegralSegment, GradientRampData, PaintVertex, Painter, Path,
    PathBuilder, PathSlice, RRect, StrokeLocation, TextFormat, TextLayout, TextureFill, compute_gradient_integral,
    renderer,
};
use color::{Srgba8, srgba8};
use gpu::PrimitiveTopology::TriangleList;
use gpu::{CommandBuffer, Format, Ptr, PushDataSource};
use log::{error, warn};
use math::{Mat3, Rect, Vec2, Vec3, Vec4, rect_transform, vec2, vec3};
use std::mem;

/// Options for drawing a glyph run.
#[derive(Clone, Debug, Default)]
pub struct DrawGlyphRunOptions {
    /// Glyph color.
    pub color: Srgba8,
}

#[derive(Clone, Copy, Debug)]
pub struct GroupOptions {
    pub blend_mode: BlendMode,
    pub opacity: f32,
}

/// Path stroking options.
#[derive(Clone, Copy, Debug)]
pub struct StrokeOptions {
    /// Stroke width in pixels.
    pub width: f32 = 1.0,
    /// Stroke location relative to the path.
    pub location: StrokeLocation = StrokeLocation::Center,
}

/// Represents a gradient ramp (a list of color stops).
///
/// Gradient ramps are valid only while a scene is being built.
/// Storing a [`GradientRamp`] between scenes is not allowed and may lead to undefined behavior.
pub type GradientRamp = usize;

/// Representation of a painter scene being drawn.
// NOTE: eventually we may want to submit draw commands directly from the PaintScene methods instead of submitting everything in `render()`,
//       which would require holding a reference to a `CommandBuffer`.
pub struct PaintScene {
    clip_stack: Vec<Rect>,
    transform_stack: Vec<Mat3>,
    transform: Mat3,
    tmp_path: PathBuilder,
    rscene: renderer::Scene,
}

impl PaintScene {
    /// Creates a new [`PaintScene`] initially cleared with the specified color.
    pub fn new(clear_color: Srgba8) -> Self {
        Self {
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

    /// Draws a rectangle outline with the specified width and fill.
    pub fn stroke_rect(&mut self, rect: impl Into<Rect>, fill: impl Into<Fill>, stroke_options: &StrokeOptions) {
        let rect = rect.into();
        let (inner, outer) = match stroke_options.location {
            StrokeLocation::Center => {
                (rect.expand(-stroke_options.width * 0.5), rect.expand(stroke_options.width * 0.5))
            }
            StrokeLocation::Inside => (rect.expand(-stroke_options.width), rect),
            StrokeLocation::Outside => (rect, rect.expand(stroke_options.width)),
        };
        self.tmp_path.rect(&inner);
        self.tmp_path.rect_ccw(&outer);
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

    /// Pushes the current transform onto the stack. Subsequent calls to [`restore()`](Self::restore) will restore this transform.
    pub fn save(&mut self) {
        self.transform_stack.push(self.transform);
    }

    /// Pops the last transform from the stack and sets it as the current transform.
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

    /// Draws a line between two points with the specified width and fill.
    pub fn draw_line(&mut self, p0: Vec2, p1: Vec2, width: f32, fill: impl Into<Fill>) {
        // TODO: width?
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
    pub fn push_group(&mut self, _group_options: &GroupOptions) {
        //self.rscene.draw_commands.push(DrawCommand::BeginGroup);
    }

    ///
    pub fn pop_group(&mut self) {}

    /// Draws text at the specified position with the given format and color.
    ///
    /// # Arguments
    /// * `position` where the text will be drawn (top-left corner).
    /// * `text` the text to draw.
    /// * `format` the text format to use.
    /// * `color` text color.
    pub fn draw_text(&mut self, position: Vec2, text: &str, format: &TextFormat, color: Srgba8) {
        let _span = crate::span!("draw_text");
        let mut layout = TextLayout::new(format, text);
        layout.layout(1000.0);
        for glyph_run in layout.glyph_runs() {
            self.draw_glyph_run(position, &glyph_run, &DrawGlyphRunOptions { color });
        }
    }

    /// Draws a TextLayout object.
    pub fn draw_text_layout(&mut self, position: Vec2, layout: &TextLayout, color: Srgba8) {
        for glyph_run in layout.glyph_runs() {
            self.draw_glyph_run(position, &glyph_run, &DrawGlyphRunOptions { color });
        }
    }

    /// Draws a glyph run.
    pub fn draw_glyph_run(&mut self, position: Vec2, glyph_run: &GlyphRun<'_>, options: &DrawGlyphRunOptions) {
        let _span = crate::span!("draw_glyph_run");
        let format = glyph_run.format();
        let x = glyph_run.offset();
        let y = glyph_run.baseline();
        self.draw_glyphs(position + vec2(x, y), glyph_run.glyphs(), format, options);
    }

    /// Draws a sequence of positioned glyphs.
    ///
    /// # Arguments
    /// * baseline_position starting position of the glyphs. It positions *the baseline* of the first glyph.
    pub fn draw_glyphs(
        &mut self,
        baseline_position: Vec2,
        glyphs: impl IntoIterator<Item = Glyph>,
        format: &TextFormat,
        options: &DrawGlyphRunOptions,
    ) {
        let mut painter = get_context().painter.borrow_mut();
        let mut advance = 0.0;
        for glyph in glyphs {
            //eprintln!(
            //    "   glyph id={} advance={} (x={})",
            //    glyph.id.0,
            //    glyph.advance,
            //    x + advance
            //);

            let pos = baseline_position + vec2(advance, 0.0) + glyph.offset;
            //debug!("glyph_offset = {:?}", glyph.offset);
            advance += glyph.advance;

            // Rasterize the glyph in the texture atlas.
            let (entry, quantized_pos) = painter.rasterize_glyph(&format.font, glyph.id, format.size as u32, pos);

            if entry.px_bounds.is_null() {
                // Nothing to draw for this glyph (whitespace).
                continue;
            }

            let quad = entry.px_bounds.to_rect().translate(quantized_pos);
            self.fill_rect(
                quad,
                TextureFill {
                    texture: painter.texture_atlas.texture_handle(),
                    sampler: painter.sampler.device_handle(),
                    local_to_uv: rect_transform(quad, Rect::from_min_max(entry.uv[0], entry.uv[1])),
                    color: options.color,
                },
            );
        }
    }

    /// Alias for [`render_scene(cmd, render_target, self)`](render_scene).
    pub fn render(self, render_target: &gpu::Image) {
        render_scene(render_target, self);
    }
}

/// Renders a [`PaintScene`] to the specified render target.
///
/// This consumes the `PaintScene` object.
///
/// # Arguments
///
pub fn render_scene(render_target: &gpu::Image, scene: PaintScene) {
    let _span = crate::span!("render_scene");
    if !scene.transform_stack.is_empty() {
        warn!("finish() called with unbalanced save/restore calls");
    }

    // Check target format.
    assert!(
        matches!(render_target.format(), Format::R8G8B8A8_UNORM),
        "unsupported color target format: {:?}",
        render_target.format()
    );

    // Upload pending changes to texture atlas.
    let mut painter = get_context().painter.borrow_mut();
    painter.update_texture_atlas();

    // Render the scene.
    if let Err(err) = renderer::render_scene(&mut painter, render_target, &scene.rscene) {
        error!("failed to render scene: {err}");
    }
}
