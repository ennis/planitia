use ab_glyph::GlyphId;
use crate::paint::text::format::TextFormat;
use crate::paint::text::{Glyph, ShapingParams, shape_text};
use math::{vec2, Vec2};
use std::ops::Range;

#[derive(Default, Copy, Clone, Debug, Eq, PartialEq)]
pub struct ClusterInfo(u16);

const CLUSTER_FLAG_LINE_BREAK_BEFORE: u16 = 0x01;
const CLUSTER_FLAG_LINE_BREAK_AFTER: u16 = 0x02;
const CLUSTER_FLAG_MANDATORY_BREAK_BEFORE: u16 = 0x04;
const CLUSTER_FLAG_MANDATORY_BREAK_AFTER: u16 = 0x08;
const CLUSTER_FLAG_WHITESPACE: u16 = 0x10;

impl ClusterInfo {
    pub fn set_line_break_before(&mut self) {
        self.0 |= CLUSTER_FLAG_LINE_BREAK_BEFORE;
    }

    pub fn is_line_break_before(&self) -> bool {
        (self.0 & CLUSTER_FLAG_LINE_BREAK_BEFORE) != 0
    }

    pub fn set_line_break_after(&mut self) {
        self.0 |= CLUSTER_FLAG_LINE_BREAK_AFTER;
    }

    pub fn is_line_break_after(&self) -> bool {
        (self.0 & CLUSTER_FLAG_LINE_BREAK_AFTER) != 0
    }

    pub fn set_mandatory_break_before(&mut self) {
        self.0 |= CLUSTER_FLAG_MANDATORY_BREAK_BEFORE;
    }

    pub fn is_mandatory_break_before(&self) -> bool {
        (self.0 & CLUSTER_FLAG_MANDATORY_BREAK_BEFORE) != 0
    }

    pub fn set_mandatory_break_after(&mut self) {
        self.0 |= CLUSTER_FLAG_MANDATORY_BREAK_AFTER;
    }

    pub fn is_mandatory_break_after(&self) -> bool {
        (self.0 & CLUSTER_FLAG_MANDATORY_BREAK_AFTER) != 0
    }

    pub fn is_mandatory_break(&self) -> bool {
        self.is_mandatory_break_before() || self.is_mandatory_break_after()
    }

    pub fn is_line_break(&self) -> bool {
        self.is_line_break_before() || self.is_line_break_after()
    }

    pub fn set_whitespace(&mut self) {
        self.0 |= CLUSTER_FLAG_WHITESPACE;
    }

    pub fn is_whitespace(&self) -> bool {
        (self.0 & CLUSTER_FLAG_WHITESPACE) != 0
    }
}

#[derive(Copy, Clone, Debug)]
pub struct LineMetrics {
    /// Baseline relative to the top of the line.
    pub baseline: f32,
    /// Max ascender distance above the baseline.
    pub ascent: f32,
    /// Max descender distance below the baseline (positive value).
    pub descent: f32,
    /// Spacing after this line.
    pub leading: f32,
    /// Total line height (ascent + descent + leading).
    pub height: f32,
}

/// Line within a text layout.
#[derive(Clone, Debug)]
pub struct LineData {
    /// Line metrics.
    pub metrics: LineMetrics,
    /// Range within the original text.
    pub range: Range<usize>,
    /// Glyph cluster range.
    /// TODO: we could only store the end index, since the start is the end of the previous line.
    pub cluster_range: Range<usize>,
    /// Position of the line (baseline).
    pub position: Vec2,
}

impl LineData {}

/// A fragment of text with the same format.
pub struct TextFragment<'a> {
    format: &'a TextFormat,
    range: Range<usize>,
    glyphs: &'a [Glyph],
}

//-------------------------------------------------------------------------------------------------

#[derive(Copy, Clone, Debug)]
pub struct GlyphClusterData {
    pub info: ClusterInfo,
    /// Number of glyphs in the cluster, or 0xFF if the cluster is a single glyph and `glyph_offset_or_id` contains the glyph id directly.
    pub glyph_len: u8,
    pub text_len: u8,
    /// Offset of glyph cluster from the start of the fragment OR direct glyph id if glyph_count = 0xFF.
    pub glyph_offset_or_id: u16,
    /// Text source range relative to the start of the fragment.
    pub text_offset: u16,
    /// Total advance of the glyph cluster.
    pub advance: f32,
}

#[derive(Default, Copy, Clone, Debug, Eq, PartialOrd, PartialEq)]
pub struct CursorData {
    text_pos: usize,
    cluster: usize,
    fragment: usize,
}

impl CursorData {
    pub const END: CursorData = CursorData {
        text_pos: usize::MAX,
        cluster: usize::MAX,
        fragment: usize::MAX,
    };
}

#[derive(Clone, Debug)]
struct FragmentData {
    format: TextFormat,
    text_range: Range<usize>,
    cluster_range: Range<usize>,
}

/// An iterator over glyph runs in a text layout.
pub struct GlyphRunIter<'a> {
    layout: &'a TextLayout,
    /// Current line.
    line: usize,
    fragment: usize,
    cluster: usize,
    /// Last cluster index.
    ///
    /// It should be either at the end of a fragment, or at a line boundary.
    last_cluster: usize,
    x: f32,
}

impl<'a> Iterator for GlyphRunIter<'a> {
    type Item = GlyphRun<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        // Glyph runs approximately correspond to fragments, except that
        // glyph runs don't cross line boundaries. So we iterate over fragments, and split them in
        // multiple runs if they cross line boundaries.

        if self.cluster >= self.last_cluster {
            return None;
        }

        let x = self.x;
        let cluster = self.cluster;
        let i_fragment = self.fragment;
        let i_line = self.line;
        let line = &self.layout.lines[i_line];
        let fragment = &self.layout.fragments[i_fragment];

        // go to next fragment|line boundary
        let mut p = fragment.cluster_range.end;
        // exceeding last cluster
        if p > self.last_cluster {
            p = self.last_cluster;
        }

        // crossing a line boundary?
        if p > line.cluster_range.end {
            p = line.cluster_range.end;
            // go to next line
            while !self.layout.lines[self.line].cluster_range.contains(&p) {
                self.line += 1;
            }
            self.x = self.layout.lines[self.line].position.x;
        } else {
            // add advances of clusters in this run
            while self.cluster < p {
                self.x += self.layout.glyph_clusters[self.cluster].advance;
                self.cluster += 1;
            }
        }
        self.cluster = p;

        // next fragment
        while self.fragment < self.layout.fragments.len() && !self.layout.fragments[self.fragment].cluster_range.contains(&p) {
            self.fragment += 1;
        }

        Some(GlyphRun {
            layout: self.layout,
            fragment,
            line,
            line_index: i_line,
            x,
            cluster_range: cluster..p,
        })

    }
}

pub struct GlyphRun<'a> {
    layout: &'a TextLayout,
    fragment: &'a FragmentData,
    line: &'a LineData,
    line_index: usize,
    x: f32,
    cluster_range: Range<usize>,
}

impl<'a> GlyphRun<'a> {
    /// Returns the text format of this run.
    pub fn format(&self) -> &'a TextFormat {
        &self.fragment.format
    }

    /// Offset of the run along the baseline.
    pub fn offset(&self) -> f32 {
        self.x
    }

    /// Position of the run baseline within the parent layout.
    pub fn baseline(&self) -> f32 {
        self.line.position.y + self.line.metrics.baseline
    }

    /// Offset (x-offset, baseline) of this run's baseline in the layout.
    ///
    /// Equivalent to `vec2(self.offset(), self.baseline())`.
    pub fn origin(&self) -> Vec2 {
        vec2(self.offset(), self.baseline())
    }

    /// Returns the range of original text covered by this run.
    pub fn text_range(&self) -> Range<usize> {
        let start_cluster = &self.layout.glyph_clusters[self.cluster_range.start];
        let end_cluster = &self.layout.glyph_clusters[self.cluster_range.end - 1];
        let start = self.fragment.text_range.start + start_cluster.text_offset as usize;
        let end = self.fragment.text_range.start + end_cluster.text_offset as usize + end_cluster.text_len as usize;
        start..end
    }

    /// Returns the line index of this run.
    pub fn line_index(&self) -> usize {
        self.line_index
    }

    /// Iterates over the glyphs in this run.
    pub fn glyphs(&self) -> impl Iterator<Item = Glyph> + 'a {
        //let y_baseline = self.line.position.y + self.line.metrics.baseline;
        self.layout.glyph_clusters[self.cluster_range.clone()]
            .iter()
            .flat_map(move |cluster| {
                if cluster.glyph_len == 0xFF {
                    // simple cluster
                    let glyph = Glyph {
                        id: GlyphId(cluster.glyph_offset_or_id),
                        offset: Vec2::ZERO,    // zero for simple clusters
                        advance: cluster.advance,
                    };
                    Some(glyph)
                } else {
                    // complex cluster
                    None // TODO
                }
            })
    }
}

/// A laid-out block of text.
#[derive(Clone, Debug)]
pub struct TextLayout {
    width: f32,
    height: f32,
    fragments: Vec<FragmentData>,
    lines: Vec<LineData>,
    glyph_clusters: Vec<GlyphClusterData>,
    glyph_data: Vec<Glyph>,
}

impl TextLayout {
    /// Constructs a new text layout from a default text style and attributed text runs.
    pub fn new(format: &TextFormat, text: &str) -> TextLayout {
        let mut glyph_data = Vec::new();
        // map from
        let mut glyph_cluster_data = Vec::new();

        shape_text(
            &ShapingParams {
                font: format.font.clone(),
                size: format.size,
            },
            text,
            &mut |cluster| {
                //glyph_cluster_map.push((glyphs.len(), cluster.source_range.clone()));

                let info = {
                    let mut info = ClusterInfo::default();
                    if cluster.possible_line_break {
                        info.set_line_break_after();
                    }
                    if cluster.mandatory_line_break {
                        info.set_mandatory_break_after();
                    }
                    if cluster.whitespace {
                        info.set_whitespace();
                    }
                    info
                };

                if cluster.glyphs.len() == 1 && cluster.glyphs[0].offset == Vec2::ZERO {
                    // single-glyph cluster at zero offset, store glyph id directly
                    let source_range = cluster.source_range.clone();
                    glyph_cluster_data.push(GlyphClusterData {
                        info,
                        glyph_len: 0xFF,
                        text_len: source_range.len() as u8,
                        glyph_offset_or_id: cluster.glyphs[0].id.0,
                        // TODO split in multiple runs if too long
                        text_offset: source_range.start as u16,
                        advance: cluster.advance,
                    });
                } else {
                    todo!("complex clusters");
                    //glyph_data.extend_from_slice(cluster.glyphs);
                }
            },
        );

        // for now there's only one fragment since we don't support heterogeneous text formats yet.
        let fragment = FragmentData {
            format: format.clone(),
            text_range: 0..text.len(),
            cluster_range: 0..glyph_cluster_data.len(),
        };

        let layout = TextLayout {
            width: 0.0,
            height: 0.0,
            fragments: vec![fragment],
            lines: Vec::new(),
            glyph_clusters: glyph_cluster_data,
            glyph_data,
        };
        //dbg!(&layout);
        layout
    }

    /// Recomputes the layout of the text given the specified available width.
    pub fn layout(&mut self, width: f32) {
        let mut y = 0.0f32;
        let mut cursor = CursorData::default(); // beginning of text
        while let Some(line) = self.layout_line(&mut cursor, width as f32) {
            // continue laying out lines until we run out of text
            line.position.x = 0.0;
            line.position.y = y;
            y += line.metrics.height;
        }
        self.width = width;
        self.height = y;
    }

    /// Returns the height of the text layout.
    ///
    /// Only valid after calling `layout()`.
    pub fn height(&self) -> f32 {
        self.height
    }

    /// Returns the size of the text layout in pixels.
    ///
    /// Equivalent to `vec2(self.width(), self.height())`.
    pub fn size(&self) -> Vec2 {
        vec2(self.width, self.height)
    }

    /// Returns the baseline of the first line of text.
    ///
    /// Only valid after calling `layout()`.
    pub fn baseline(&self) -> f32 {
        if let Some(line) = self.lines.first() {
            line.metrics.baseline
        } else {
            0.0
        }
    }

    fn cursor_at_end(&self, cursor: &CursorData) -> bool {
        cursor.cluster >= self.glyph_clusters.len()
    }

    fn next_glyph_cluster(&self, cur: &mut CursorData) -> bool {
        if cur.cluster >= self.glyph_clusters.len() {
            return false;
        }

        cur.cluster = cur.cluster + 1;

        if cur.cluster >= self.glyph_clusters.len() {
            // dummy fragment index
            cur.fragment = self.fragments.len();
            // end of source text
            cur.text_pos = self.fragments.last().map(|f| f.text_range.end).unwrap_or(0);
            return false;
        }

        // the next cluster might be in the next fragment,
        // iterate forward to find the fragment containing it
        // (normally it should be the next fragment, unless there are empty fragments between)
        while cur.fragment < self.fragments.len() && !self.fragments[cur.fragment].cluster_range.contains(&cur.cluster) {
            cur.fragment += 1;
        }

        //eprintln!("cur={:?}, fragments={:?}", cur, self.fragments.len());
        assert!(cur.fragment < self.fragments.len());

        // update source text position
        cur.text_pos =
            self.fragments[cur.fragment].text_range.start + self.glyph_clusters[cur.cluster].text_offset as usize;
        true
    }

    pub fn layout_line(&mut self, pos: &mut CursorData, available_width: f32) -> Option<&mut LineData> {

        // FIXME: should delete all lines after pos

        if pos.cluster >= self.glyph_clusters.len() {
            // no more text to layout
            return None;
        }

        let mut x = 0.0f32;
        let start = *pos;
        let mut break_opportunity = None;

        // place glyph clusters in the line until we run out of space or text
        loop {
            let c = &self.glyph_clusters[pos.cluster];

            if c.info.is_line_break_before() {
                // possible line break opportunity before this character
                break_opportunity = Some(*pos);
            }

            // advance to next cluster
            self.next_glyph_cluster(pos);

            // explicit new line
            if c.info.is_mandatory_break() {
                break;
            }

            x += c.advance;

            if x <= available_width {
                // we fit, continue
                if c.info.is_line_break_after() {
                    // possible line break opportunity after this character
                    break_opportunity = Some(*pos);
                }
            } else {
                // line break, revert to last opportunity if any
                if let Some(break_pos) = break_opportunity {
                    *pos = break_pos;

                    // eat non-newline trailing whitespace
                    loop {
                        let info = &self.glyph_clusters[pos.cluster].info;
                        if info.is_mandatory_break() || !info.is_whitespace() {
                            break;
                        }
                        if !self.next_glyph_cluster(pos) {
                            break;
                        }
                    }
                }
                break
            }

            if pos.cluster >= self.glyph_clusters.len() {
                // no more text to layout
                break
            }
        }

        // we produced an empty line, because

        // calculate line metrics
        let mut max_leading = 0.0f32;
        let mut max_ascent = 0.0f32;
        let mut max_descent = 0.0f32;

        {
            let mut p = start;
            while p.cluster < pos.cluster {
                //let c = &self.glyph_clusters[p.cluster];
                let fragment = &self.fragments[p.fragment];

                let font_size = fragment.format.size;
                let ascent = fragment.format.font.ascent(font_size);
                let descent = -fragment.format.font.descent(font_size);
                let line_gap = fragment.format.font.line_gap(font_size);

                max_leading = max_leading.max(line_gap);
                max_ascent = max_ascent.max(ascent);
                max_descent = max_descent.max(descent);

                self.next_glyph_cluster(&mut p);
            }
        }

        let mut baseline = max_ascent + 0.5 * max_leading;

        // add line
        self.lines.push(LineData {
            metrics: LineMetrics {
                baseline,
                ascent: max_ascent,
                descent: max_descent,
                leading: max_leading,
                height: max_ascent + max_descent + max_leading,
            },
            range: start.text_pos..pos.text_pos,
            cluster_range: start.cluster..pos.cluster,
            position: Vec2::ZERO,
        });
        self.lines.last_mut()
    }

    /// Iterates over the glyph runs in this layout.
    pub fn glyph_runs(&self) -> GlyphRunIter<'_> {
        GlyphRunIter {
            layout: self,
            line: 0,
            fragment: 0,
            cluster: 0,
            last_cluster: self.glyph_clusters.len(),
            x: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_layout() {
        // Example test case for text layout
        let format = TextFormat::default();
        let text = "This is a sample                          text to be laid out.";
        let mut layout = TextLayout::new(&format, text);

        let available_width = 100.0;

        // current y-position
        let mut y = 0.0;
        let mut cursor = CursorData::default(); // beginning of text

        let mut line_count = 0;
        while let Some(line) = layout.layout_line(&mut cursor, available_width) {
            line.position.x = 0.0;
            line.position.y = y;
            y += line.metrics.height;

            let line_text = &text[line.range.clone()];
            let ascender = line.metrics.ascent;
            let descender = line.metrics.descent;
            let height = line.metrics.height;
            let line_number = line_count + 1;
            let baseline = line.metrics.baseline;
            let y_offset = line.position.y;
            eprintln!(
                "line {line_number}, text {line_text:?}, height={height}, ascender={ascender}, descender={descender}, baseline={baseline}, y_offset={y_offset}"
            );
            line_count += 1;
        }

        for glyph_run in layout.glyph_runs() {
            let origin = glyph_run.origin();
            let text = &text[glyph_run.text_range()];
            let line = glyph_run.line_index();
            eprintln!("Run {text:?} line {line} at {origin:?}");

            let x = glyph_run.offset();
            let y = glyph_run.baseline();
            let mut advance  = 0.0;
            for glyph in glyph_run.glyphs() {
                eprintln!("   glyph id={} advance={} (x={})", glyph.id.0, glyph.advance, x+advance);
                advance += glyph.advance;
            }
        }
    }
}

/*

Text Layout:

It's often the case that the same text need to be broken into lines multiple times, because
the available width has changed. So the TextLayout object is mutable.

The line-breaking and layout process is interactive.
The caller specifies the available width of a line, and alignment constraints,
then as much text as will fit is placed on the line and aligned. The function returns information
about the line such as the total line advance, and the line height.

It's also possible to relayout part of the text, keeping previous lines intact.

 */
