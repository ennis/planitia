mod font;
mod format;
pub(crate) mod glyph_cache;
mod layout;
mod text_run;

use crate::{Draft, DrawKind, Fill, Vertex, thread_local_resources};
use color::Srgba8;
use gpu::vulkan::*;
use math::{IRect, Vec2, u16vec2, vec2};
use std::ops::Range;

pub use font::{Font, FontId};
pub use format::{TextFormat, FormatProperty};
pub use layout::{GlyphRun, TextLayout};

impl<'a> Draft<'a> {
    /// Draws text at the specified position with the given format and color.
    ///
    /// # Arguments
    /// * `position` where the text will be drawn (top-left corner).
    /// * `text` the text to draw.
    /// * `format` the text format to use.
    /// * `color` text color.
    pub fn draw_text(&mut self, position: Vec2, text: &str, format: &TextFormat, color: Srgba8) {
        let mut layout = TextLayout::new(format, text);
        layout.layout(f32::MAX);
        for glyph_run in layout.glyph_runs() {
            self.draw_glyph_run(position, &glyph_run, &DrawGlyphRunOptions { color });
        }
    }

    /// Draws a text layout object.
    pub fn draw_text_layout(&mut self, position: Vec2, layout: &TextLayout, color: Srgba8) {
        for glyph_run in layout.glyph_runs() {
            self.draw_glyph_run(position, &glyph_run, &DrawGlyphRunOptions { color });
        }
    }

    /// Draws a glyph run.
    pub fn draw_glyph_run(&mut self, position: Vec2, glyph_run: &GlyphRun<'_>, options: &DrawGlyphRunOptions) {
        let format = glyph_run.format();
        let x = glyph_run.offset();
        let y = glyph_run.baseline();
        self.draw_glyphs(position + vec2(x, y), glyph_run.glyphs(), format, options);
    }

    pub fn draw_glyphs(
        &mut self,
        baseline_position: Vec2,
        glyphs: impl IntoIterator<Item = Glyph>,
        format: &TextFormat,
        options: &DrawGlyphRunOptions,
    ) -> f32 {
        let mut glyph_cache = thread_local_resources().glyph_cache.borrow_mut();
        let mut texture_atlas = thread_local_resources().texture_atlas.borrow_mut();
        let mut advance = 0.0;
        let texture = texture_atlas.texture_handle();
        let sampler = gpu::register_sampler(&gpu::SamplerParams {
            mag_filter: VK_FILTER_LINEAR,
            min_filter: VK_FILTER_LINEAR,
            ..
        });
        self.new_draw(DrawKind::Draw, Fill::Texture { texture, sampler });
        for glyph in glyphs {
            let pos = baseline_position + vec2(advance, 0.0) + glyph.offset;
            advance += glyph.advance;
            let (entry, quantized_pos) =
                glyph_cache.rasterize_glyph(&mut texture_atlas, &format.font, glyph.id, format.size as u32, pos);
            if entry.px_bounds.is_null() {
                continue;
            }
            let quad = entry.px_bounds.to_rect().translate(quantized_pos);
            let uv16_min = u16vec2((entry.uv[0].x * u16::MAX as f32) as u16, (entry.uv[0].y * u16::MAX as f32) as u16);
            let uv16_max = u16vec2((entry.uv[1].x * u16::MAX as f32) as u16, (entry.uv[1].y * u16::MAX as f32) as u16);
            self.append_rect(
                Vertex { pos: quad.min.extend(0.0), uv: uv16_min, color: options.color },
                Vertex { pos: quad.max.extend(0.0), uv: uv16_max, color: options.color },
            );
        }
        advance
    }
}

/// Options for drawing a glyph run.
#[derive(Clone, Debug, Default)]
pub struct DrawGlyphRunOptions {
    /// Glyph color.
    pub color: Srgba8,
}

//--------------------------------------------------------------------------------------------------

/// Glyph identifier.
pub type GlyphId = ab_glyph::GlyphId;

/// Glyph.
#[derive(Clone, Copy, Debug)]
pub struct Glyph {
    pub id: GlyphId,
    /// Offset (relative to the origin of the cluster).
    pub offset: Vec2,
    /// Advance.
    pub advance: f32,
}

/// glyph cluster.
#[derive(Clone)]
pub struct GlyphCluster<'a> {
    /// Text source range corresponding to this cluster.
    pub source_range: Range<usize>,
    pub glyphs: &'a [Glyph],
    /// Total advance of the glyph cluster.
    pub advance: f32,
    pub bounds: IRect,
    /// Indicates whether this cluster is a possible line break point.
    pub possible_line_break: bool,
    /// Indicates whether this cluster is a mandatory line break point.
    pub mandatory_line_break: bool,
    /// Indicates that this cluster is white space.
    pub whitespace: bool,
}

///
pub struct ShapingParams {
    pub font: Font,
    pub size: f32,
}

/// Performs text analysis and shape the given text run into a sequence of glyph clusters.
///
/// NOTE: currently it doesn't do any CTL shaping or ligatures.
pub fn shape_text(format: &ShapingParams, text: &str, callback: &mut dyn FnMut(&GlyphCluster)) {
    //let _span = crate::span!("shape_text");
    let font = &format.font;
    let size = format.size;
    let mut prev_glyph_id: Option<GlyphId> = None;

    for (ch_pos, ch) in text.char_indices() {
        let ch_len_utf8 = ch.len_utf8();
        let src_range = ch_pos..(ch_pos + ch_len_utf8);

        // determine boundary type
        // TODO: more sophisticated boundary detection
        let whitespace = ch.is_whitespace();
        let possible_line_break = whitespace || ch == '-';
        let mandatory_line_break = ch == '\n' || ch == '\r';

        let glyph_id = font.glyph_id(ch);

        if glyph_id.0 != 0 {
            //debug!("Shaping char '{ch}' (id={glyph_id:?}) at {src_range:?}");

            let mut advance = font.h_advance(glyph_id, size);
            // adjust for kerning
            // note that this could also be an offset on the glyph position
            let next_glyph_id = text[src_range.end..].chars().next().map(|c| font.glyph_id(c));
            if let Some(next_glyph_id) = next_glyph_id {
                let kern = font.kern(glyph_id, next_glyph_id, size);
                advance += kern;
            }

            let glyph = Glyph {
                id: glyph_id,
                // for simple shaping the offset is always zero
                offset: vec2(0.0, 0.0),
                advance,
            };

            callback(&GlyphCluster {
                source_range: src_range,
                glyphs: std::slice::from_ref(&glyph),
                advance,
                bounds: Default::default(),
                possible_line_break,
                mandatory_line_break,
                whitespace,
            });
            prev_glyph_id = Some(glyph_id);
        } else {
            // cluster without associated glyph
            callback(&GlyphCluster {
                source_range: src_range,
                glyphs: &[],
                advance: 0.0,
                bounds: Default::default(),
                possible_line_break,
                mandatory_line_break,
                whitespace,
            });
            prev_glyph_id = None;
        }
    }
}
