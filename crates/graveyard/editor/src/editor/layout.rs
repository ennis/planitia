//! Editor layout algorithm.
//!
//! This module contains the algorithm that produces a laid-out [visual tree](crate::editor::VisualTree)
//! from a [Document](crate::node::Document) and their associated [LayoutItems](crate::layout::LayoutItem).

use crate::editor::{Block, EditorMetrics, VisualCell, VisualTree};
use crate::layout::{LayoutItem, LayoutItemKind, LayoutMode};
use crate::node::Node;
use gamelib::math::{IRect, IVec2, ivec2};
use gamelib::paint::{TextFormat, TextLayout};

/// Layout context for a group of cells.
#[derive(Clone, Copy, Debug)]
struct BlockCtx {
    idx: usize,
    pos: IVec2 = IVec2::ZERO,
    line_height: i32 = 0,
    size: IVec2 = IVec2::ZERO,
}

struct LayoutState<'a> {
    metrics: &'a EditorMetrics,
    visual_tree: VisualTree,
    /// Current text format.
    text_format: TextFormat,
}

/*
struct Measure {
    size: IVec2,
    baseline: i32,
    mode: LayoutMode,
}*/

impl<'a> LayoutState<'a> {
    /// Allocates a rectangle in the block context and advances the position horizontally.
    fn alloc_inline_rect(&mut self, block: &mut BlockCtx, size: IVec2) -> IRect {
        let rect = IRect::from_origin_size(block.pos, size);
        block.pos.x += size.x;
        block.line_height = block.line_height.max(size.y);
        block.size.x = block.pos.x;
        block.size.y = block.size.y.max(block.pos.y + size.y);
        rect
    }

    /// Layouts a text cell.
    fn layout_text(&mut self, block: &mut BlockCtx, text: &str) {
        let mut text_layout = TextLayout::new(&self.text_format, text);
        text_layout.layout(1000.0); // TODO: Use actual width limit.
        let advance = text_layout.max_line_advance().ceil() as i32;
        let size = IVec2::new(advance, self.metrics.line_height);
        let rect = self.alloc_inline_rect(block, size);

        self.visual_tree.text_layouts.push(text_layout);
        self.visual_tree.cells.push(VisualCell {
            block_idx: block.idx,
            rect,
            text_layout_idx: Some(self.visual_tree.text_layouts.len() - 1),
            selectable: true,
        });
    }

    /// Layouts a node and its fields.
    fn layout_node(&mut self, block: &mut BlockCtx, node: &Node) {
        if let Some(item) = node.layout() {
            match item.mode {
                LayoutMode::Block => {
                    self.layout_block_item(block, node, item);
                }
                LayoutMode::Inline => {
                    self.layout_item(block, node, item);
                }
            }
        }
    }

    fn layout_block_item(&mut self, block: &mut BlockCtx, node: &Node, item: &LayoutItem) {
        // Start a new line for block-level elements.
        if block.line_height != 0 {
            block.pos.x = 0;
            block.pos.y += block.line_height;
            block.line_height = 0;
        }

        let cell_span_start = self.visual_tree.cells.len();

        // Create the new block in the visual tree.
        self.visual_tree.blocks.push(Block {
            parent: Some(block.idx),
            range: 0..0,
            rect: IRect::from_origin_size(block.pos, IVec2::ZERO),
        });
        let mut child_block_ctx = BlockCtx { idx: self.visual_tree.blocks.len() - 1, .. };

        // Layout the item inside the newly created block.
        self.layout_item(&mut child_block_ctx, node, item);

        // Update spanned cell range.
        let cell_span_end = self.visual_tree.cells.len();
        self.visual_tree.blocks[child_block_ctx.idx].range = cell_span_start..cell_span_end;

        // Move parent cursor on the next line after the block.
        block.pos.x = 0;
        block.pos.y += child_block_ctx.size.y;
        block.line_height = 0;
    }

    /// Processes a layout cell.
    fn layout_item(&mut self, block: &mut BlockCtx, node: &Node, layout: &LayoutItem) {
        // Layout children.
        match layout.kind {
            LayoutItemKind::Field { link, style: _ } => {
                let node_child = node.field(link.index).expect("field index out of bounds");
                if let Some(node) = node_child.as_node() {
                    self.layout_node(block, node);
                } else if let Some(node_collection) = node_child.as_collection() {
                    for child_node in node_collection.iter() {
                        self.layout_node(block, child_node);
                    }
                } else if let Some(_value) = node_child.as_value() {
                    // TODO property values
                    self.layout_text(block, "<...>");
                }
            }
            LayoutItemKind::Text { text, style } => {
                self.layout_text(block, text);
            }
            LayoutItemKind::Collection(layouts) => {
                for layout in layouts {
                    self.layout_item(block, node, layout);
                }
            }
        }
    }

    fn compute_cell_positions(&mut self) {
        for block in self.visual_tree.blocks.iter() {
            // For each cell, add the offsets of all enclosing blocks.
            for cell in self.visual_tree.cells[block.range.clone()].iter_mut() {
                cell.rect = cell.rect.translate(block.rect.min);
            }
        }
    }
}

/// Lays out a node tree into a visual tree.
pub(super) fn layout(root_node: &Node, metrics: &EditorMetrics, text_format: &TextFormat) -> VisualTree {
    let mut state = LayoutState {
        metrics,
        visual_tree: VisualTree { blocks: Vec::new(), cells: Vec::new(), text_layouts: Vec::new(), root_box: 0 },
        text_format: text_format.clone(),
    };

    // Start with a root block.
    state.visual_tree.blocks.push(Block {
        parent: None,
        range: 0..0,
        rect: IRect::from_origin_size(IVec2::ZERO, IVec2::ZERO),
    });
    let mut root_block = BlockCtx { idx: 0, .. };
    state.layout_node(&mut root_block, root_node);

    // Update the root box range.
    state.visual_tree.blocks[0].range.end = state.visual_tree.cells.len();

    // Compute final cell positions.
    state.compute_cell_positions();

    state.visual_tree
}
