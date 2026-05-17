use log::warn;
use crate::paint::fill::{Fill, rect_transform};
use crate::paint::flatten::flatten_path;
use crate::paint::render::{DrawOp, RenderScene, RenderScene2, render_scene, render_scene_2};
use crate::paint::tessellation::Tessellator;
use crate::paint::{GlyphRun, PaintRenderParams, Painter, Path, PathBuilder, PathSlice, RRect};
use color::{srgba8, Srgba8};
use gpu::CommandBuffer;
use math::{Mat3, Rect, Vec2, vec2};

/// Options for drawing a glyph run.
#[derive(Clone, Debug, Default)]
pub struct DrawGlyphRunOptions {
    /// Glyph color.
    pub color: Srgba8,
    /// Glyph size in points.
    pub size: f32,
}

/// Representation of a painter scene being drawn.
pub struct PaintScene<'a> {
    painter: &'a mut Painter,
    pub(crate) tess: Tessellator,
    scene: RenderScene,
    clip_stack: Vec<Rect>,
    transform_stack: Vec<Mat3>,
    transform: Mat3,
    scene_2: RenderScene2,
}

impl<'a> PaintScene<'a> {
    pub(super) fn new(painter: &'a mut Painter) -> Self {
        Self {
            painter,
            tess: Tessellator::new(),
            scene: RenderScene::default(),
            clip_stack: vec![Rect::INFINITE],
            transform_stack: vec![],
            transform: Default::default(),
            scene_2: RenderScene2::default(),
        }
    }

    fn push_draw_op(&mut self, fill: impl Into<Fill>) {
        let fill = fill.into();
        let device_to_local = Mat3::IDENTITY; // TODO transform stack
        self.scene.ops.push(DrawOp {
            mesh: self.tess.finish_and_reset(),
            clip: self.clip_rect(),
            device_to_local,
            fill,
        });
    }

    /// Draws a rounded rectangle at the specified position with the given size and corner radius.
    pub fn fill_rrect(&mut self, rect: Rect, radius: f32, fill: impl Into<Fill>) {
        let fill = fill.into();
        self.tess.fill_rrect(RRect { rect, radius });
        self.push_draw_op(fill);

     //  //v2 scene
     //  let mut path = PathBuilder::new();
     //  path.move_to(rect.min);
     //  path.line_to(vec2(rect.max.x, rect.min.y - 233.0));
     //  path.line_to(rect.max);
     //  path.line_to(vec2(rect.min.x + 20.0, rect.max.y));
     //  path.close();

     // self.tess.stroke_path(PathSlice::from(&path), 1.0);
     // self.push_draw_op(Fill::Solid(srgba8(255, 255, 0, 255)));

     //  flatten_path(
     //      PathSlice::from(&path),
     //      &Mat3::IDENTITY,
     //      1.0,
     //      self.scene_2.path_index,
     //      &mut self.scene_2.vertices,
     //      &mut self.scene_2.contours,
     //  );
     //  self.scene_2.path_index += 1;
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
    }

    pub fn transform(&mut self, transform: Mat3) {
        self.transform = self.transform * transform;
    }

    pub fn fill_path(&mut self, path: &Path, fill: impl Into<Fill>) {
        let (vertices, contours) = flatten_path(
            PathSlice::from(path),
            &self.transform,
            1.0,
            self.scene_2.path_index,
            &mut self.scene_2.vertices,
            &mut self.scene_2.contours,
        );
        self.scene_2.paths.push(contours);
        self.scene_2.fills.push(fill.into());
        self.scene_2.path_index += 1;

       // self.tess.stroke_path(PathSlice::from(path), 1.0);
       // self.push_draw_op(Fill::Solid(srgba8(255, 255, 0, 255)));
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
                uv_transform: rect_transform(quad, Rect::from_min_max(entry.uv[0], entry.uv[1])),
            });
        }
    }

    pub fn finish(mut self, cmd: &mut CommandBuffer, params: &PaintRenderParams) {
        render_scene_2(cmd, self.painter, params, &self.scene_2);
        render_scene(cmd, self.painter, params, &self.scene);
    }
}
