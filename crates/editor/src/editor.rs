//! Implementation of the projectional editor.

use crate::layout::{LCell, LChildCollection, LCollection, LTextCell, LayoutMode};
use crate::model::{Node, NodeData};
use gamelib::color::{Srgba8, srgba8};
use gamelib::math::{IRect, IVec2};
use gamelib::paint::{Font, PaintScene, StrokeLocation, StrokeOptions, TextFormat, TextLayout};
use log::{debug, info};

#[derive(Clone, Debug)]
pub struct EditorColors {
    pub background: Srgba8 = srgba8(255, 255, 255, 255),
    pub text: Srgba8 = srgba8(0, 0, 0, 255),
    pub cell_borders: Srgba8 = srgba8(200, 200, 200, 255),
    pub selection: Srgba8 = srgba8(0, 120, 215, 100),
    pub cursor: Srgba8 = srgba8(0, 0, 0, 255),
}

impl Default for EditorColors {
    fn default() -> Self {
        Self { .. }
    }
}

#[derive(Clone, Debug)]
pub struct EditorConfig {
    /// Default font.
    pub font: Font,
    /// Indent size in basic width units.
    pub indent: u32,
    pub colors: EditorColors = EditorColors { .. },
}

#[derive(Copy, Clone, Debug)]
struct EditorMetrics {
    font_size: i32,
    /// Atomic unit of width, in pixels.
    ///
    /// Corresponds to the width of a space character in the current font and size.
    width_unit: i32,
    /// Indent size in basic width units.
    indent: u32,
    /// Line height in pixels.
    line_height: i32,
}

/// Document cursor.
#[derive(Copy, Clone, Debug)]
pub struct Cursor {
    /// Index of the visual cell that the cursor is in.
    cell_index: usize,
    /// Text position within the cell.
    text_offset: usize,
}

/// Describes a cursor movement operation.
#[derive(Copy, Clone, Debug)]
pub enum CursorMove {
    /// Move the cursor to the next character, or to the next selectable visual cell if at the end of the current cell.
    Next,
    /// Move the cursor to the previous character, or to the previous selectable visual cell if at the start of the current cell.
    Previous,
    /// Move the cursor to the start of the current visual cell.
    CellStart,
    /// Move the cursor to the end of the current visual cell.
    CellEnd,
    /// Move the cursor to the start of the current line.
    StartOfLine,
    /// Move the cursor to the end of the current line.
    EndOfLine,
    /// Move the cursor to the start of the parent logical node.
    ParentNode,
}

/// An instance of the projectional editor.
///
/// Handles text input, layout, and rendering of a document.
pub(crate) struct Editor {
    document: Node,
    config: EditorConfig,
    metrics: EditorMetrics,
    cells: Vec<VCell> = vec![],
    text_layouts: Vec<TextLayout> = vec![],
}

impl Editor {
    pub fn new(document: Node, config: &EditorConfig) -> Self {
        let mut this = Self {
            document,
            config: config.clone(),
            metrics: EditorMetrics { font_size: 0, width_unit: 0, indent: 2, line_height: 0 },
            ..
        };
        // Set default font size, which also computes the metrics.
        this.set_font_size(24);
        this
    }

    pub fn set_font_size(&mut self, font_size: i32) {
        self.metrics.font_size = font_size;
        let font_size_f32 = font_size as f32;
        self.metrics.width_unit = self.config.font.h_advance_char(' ', font_size_f32).ceil() as i32;
        self.metrics.line_height = (font_size_f32 + self.config.font.line_gap(font_size_f32)).ceil() as i32;
    }

    pub fn move_cursor(&mut self, )

    fn text_format(&self) -> TextFormat {
        TextFormat {
            font: self.config.font.clone(),
            size: self.metrics.font_size as f32,
            color: self.config.colors.text,
            ..
        }
    }

    pub fn layout(&mut self) {
        let arena = bumpalo::Bump::new();
        let text_format = self.text_format();
        let mut state = LayoutState::new(&arena, text_format, &self.metrics);
        state.layout_node(&self.document);
        self.cells = state.vcells;
        self.text_layouts = state.layouts;
    }

    pub fn paint(&mut self, scene: &mut PaintScene) {
        scene.clear(self.config.colors.background);
        for cell in self.cells.iter() {
            if let Some(idx) = cell.text_layout_idx {
                let text_layout = &self.text_layouts[idx];
                scene.draw_text_layout(cell.rect.min.as_vec2(), text_layout, self.config.colors.text);
            }

            let mut cell_rect = cell.rect;
            // inflate so that the border overlaps with adjacent cells
            cell_rect.max.x += 1;
            cell_rect.max.y += 1;
            scene.stroke_rect(
                cell_rect.to_rect(),
                self.config.colors.cell_borders,
                &StrokeOptions { width: 1.0, location: StrokeLocation::Inside },
            );
        }
    }
}

/// Visual cell.
struct VCell {
    /// Computed bounds.
    rect: IRect,
    /// Index of the TextLayout for this cell in the layout vector.
    text_layout_idx: Option<usize>,
    /// Whether this cell is selectable (i.e., can receive the cursor).
    /// If false, the cursor will skip over this cell when moving.
    selectable: bool,
}

impl VCell {
    fn empty() -> Self {
        Self { rect: IRect::ZERO, text_layout_idx: None, selectable: false }
    }
}

#[derive(Clone, Copy, Debug)]
enum OuterLayoutMode {
    /// Contents are laid out inline within the parent flow.
    /// This doesn't define a separate block layout context.
    Inline,
    /// Start a new block layout context, with its own flow and bounds.
    Block {
        /// Whether to add a line break before this block.
        break_before: bool,
        /// Whether to add a line break after this block.
        break_after: bool,
    },
}

#[derive(Clone, Copy, Debug)]
struct LayoutCtx {
    /// How the contents should be laid out.
    content_layout: LayoutMode,
    self_layout: OuterLayoutMode,
    cursor: IVec2,
    rect: IRect,
}

struct LayoutState<'a> {
    arena: &'a bumpalo::Bump,
    metrics: &'a EditorMetrics,
    pos: IVec2,
    block_start: bool,
    /// Current text format.
    text_format: TextFormat,
    layouts: Vec<TextLayout>,
    vcells: Vec<VCell>,
    ctx: LayoutCtx,
    ctx_stack: Vec<LayoutCtx>,
}

/*
 
 */

impl<'a> LayoutState<'a> {
    fn new(arena: &'a bumpalo::Bump, text_format: TextFormat, metrics: &'a EditorMetrics) -> Self {
        Self {
            arena,
            pos: IVec2::ZERO,
            metrics,
            block_start: false,
            text_format,
            layouts: vec![],
            vcells: vec![],
            ctx: LayoutCtx {
                content_layout: LayoutMode::Inline,
                self_layout: OuterLayoutMode::Inline,
                cursor: IVec2::ZERO,
                rect: IRect::ZERO,
            },
            ctx_stack: vec![],
        }
    }

    fn layout_node(&mut self, node: &Node) {
        let Some(lc) = node.layout() else {
            // Nothing to layout
            return;
        };
        self.layout_cell(lc)
    }

    fn layout_cell(&mut self, l_cell: &LCell) {
        match l_cell {
            LCell::Text(l_text) => {
                self.layout_text_cell(l_text);
            }
            LCell::Child(l_ch) => {}
            LCell::Collection(l_coll) => {
                self.layout_collection_cell(l_coll);
            }
            LCell::ChildCollection(l_chcoll) => {

            }
        }
    }

    fn alloc_rect(&mut self, size: IVec2) -> IRect {
        let rect = IRect::from_origin_size(self.pos, size);
        self.pos.x += size.x;
        rect
    }

    fn layout_child_collection_cell(&mut self, node: &NodeData, l_chcoll: &LChildCollection) {
        let direction = l_chcoll.direction;
        let pos = self.pos;
        //let collection = node.
    }

    fn layout_collection_cell(&mut self, l_coll: &LCollection) {
        let direction = l_coll.direction;
        let pos = self.pos;
        for child in l_coll.children {
            self.layout_cell(child);
            match direction {
                LayoutMode::Horizontal => {
                    self.pos.y = pos.y;
                }
                LayoutMode::Vertical => {
                    self.pos.x = pos.x;
                    self.pos.y += self.metrics.line_height;
                }
                LayoutMode::Inline => {}
            }
        }
    }

    fn layout_text(&mut self, text: &str) {
        let mut text_layout = TextLayout::new(&self.text_format, text);
        text_layout.layout(1000.0); // TODO: Use actual width limit.
        let advance = text_layout.max_line_advance().ceil() as i32;
        let height = self.metrics.line_height;
        let size = IVec2::new(advance, height);
        let rect = self.alloc_rect(size);

        self.layouts.push(text_layout);
        let index = self.layouts.len() - 1;

        self.vcells.push(VCell { rect, text_layout_idx: Some(index) });
    }

    /// Layouts a text cell.
    fn layout_text_cell(&mut self, l_text: &LTextCell) {
        self.layout_text(l_text.text)
    }

    /// Enters a new block.
    fn begin_group(&mut self, group_layout: OuterLayoutMode, layout_mode: LayoutMode) {
        // Save current layout context.
        self.ctx_stack.push(self.ctx);

        match group_layout {
            OuterLayoutMode::Block { break_before: true, .. } if self.ctx.content_layout == LayoutMode::Inline => {
                // The group produces an out-of-line block, breaking the inline flow.
                // Put cursor at the start of the next line.
                self.pos.x = self.ctx.rect.min.x;
                self.pos.y += self.metrics.line_height;
            }
            _ => {}
        }

        // Set the new layout context.
        self.ctx.self_layout = group_layout;
        self.ctx.content_layout = layout_mode;
    }

    fn end_group(&mut self) {
        let mut prev_ctx = self.ctx_stack.pop().expect("No previous context to restore");

        // Finishing a group.
        match self.ctx.self_layout {
            OuterLayoutMode::Block { break_after: true, .. } => {
                // Go to next line after this block.
                prev_ctx.cursor.y += self.metrics.line_height;
            }
            OuterLayoutMode::Block { break_after: false, .. } => {
                // Stay on the same line after this block.
                prev_ctx.cursor.x = self.ctx.rect.width();
            }
            _ => {}
        }

        self.ctx = prev_ctx;
    }

    fn push_cell(&mut self, vcell: VCell) {
        let rect = vcell.rect;
        self.vcells.push(vcell);

        // Advance the cursor based on the current layout mode.
        match self.ctx.content_layout {
            LayoutMode::Horizontal => {
                self.pos.x += rect.width();
            }
            LayoutMode::Vertical => {
                self.pos.y += rect.height();
            }
            LayoutMode::Inline => {
                self.pos.x += rect.width();
            }
        }
    }

    /// Advances current line position.
    fn advance(&mut self, advance: i32) {
        self.pos.x += advance;
        self.block_start = false;
    }
}
