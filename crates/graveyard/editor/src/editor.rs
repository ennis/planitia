//! Implementation of the projectional editor.

use std::ops::Range;
use crate::node::{Document, DocumentBox};
use gamelib::color::{srgba8, Srgba8};
use gamelib::input::{KeyboardEvent, NamedKey};
use gamelib::paint::{Font, PaintScene, StrokeLocation, StrokeOptions, TextFormat, TextLayout};
use gamelib::InputEvent;
use gamelib::math::{ivec2, IRect};

mod layout;

/// Editor color scheme.
#[derive(Clone, Debug)]
pub struct EditorColors {
    /// Background color.
    pub background: Srgba8 = srgba8(255, 255, 255, 255),
    /// Default text color.
    pub text: Srgba8 = srgba8(0, 0, 0, 255),
    /// Default cell border color.
    pub cell_borders: Srgba8 = srgba8(200, 200, 200, 255),
    /// Selection highlight color.
    pub selection: Srgba8 = srgba8(0, 120, 215, 100),
    /// Cursor color.
    pub cursor: Srgba8 = srgba8(0, 0, 0, 255),
}

impl Default for EditorColors {
    fn default() -> Self {
        Self { .. }
    }
}

/// Editor configuration options.
#[derive(Clone, Debug)]
pub struct EditorConfig {
    /// Default font.
    pub font: Font,
    /// Indent size in basic width units (spaces).
    pub indent: u32,
    /// Color scheme.
    pub colors: EditorColors = EditorColors { .. },
}

/// Computed editor metrics derived from the font.
#[derive(Copy, Clone, Debug)]
pub(crate) struct EditorMetrics {
    /// Font size in pixels.
    pub(crate) font_size: i32,
    /// Atomic unit of width, in pixels.
    ///
    /// Corresponds to the width of a space character in the current font and size.
    pub(crate) width_unit: i32,
    /// Indent size in basic width units.
    pub(crate) indent: u32,
    /// Line height in pixels.
    pub(crate) line_height: i32,
}

/// Editor cursor.
#[derive(Copy, Clone, Debug, Default)]
pub struct EditCursor {
    /// Index of the [visual cell](crate::editor::VisualCell) that the cursor is in.
    pos: usize = 0,
    /// Text position within the cell.
    text_offset: usize = 0,
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

enum CursorMoveDirection {
    Left,
    Right,
    Up,
    Down,
}

enum VerticalMoveDirection {
    Up,
    Down,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum LogicalCursorDirection {
    Forward,
    Backward,
}

// The editing cursor points to a document node, and possibly a text offset within that node.
// When the document is modified somehow, the underlying node may be deleted, which invalidates the editing cursor.
// To move the cursor, say, to the next node:
// - Retrieve the current node from the cursor
// ...
//
// Issue: the editing cursor *does not* point directly to a document node.
// This is because document nodes don't necessarily expand to one or more visual cells.
// The cursor should move between visual cells.
//
// To move the cursor to the next visual cell, increment the cell index.

//
// Editing:
// The cursor is positioned in a visual cell. We need to know the available editing operations for the cell.
// These are provided by the logical node that the cell belongs to.

pub(crate) struct VisualCell {
    /// Innermost block containing this cell.
    pub(crate) block_idx: usize,
    /// Computed bounds.
    pub(crate) rect: IRect,
    /// Index of the TextLayout for this cell in the layout vector.
    pub(crate) text_layout_idx: Option<usize>,
    /// Whether this cell is selectable (i.e., can receive the cursor).
    /// If false, the cursor will skip over this cell when moving.
    pub(crate) selectable: bool,
}

/// Rectangular region holding a group of visual cells.
struct Block {
    /// Index of the parent box.
    parent: Option<usize>,
    /// Visual cell range spanned by this visual box.
    range: Range<usize>,
    /// Computed box bounds.
    rect: IRect,
}

/// Visual tree.
///
/// Holds the hierarchy of blocks and visual cells that make up the document visually.
/// Visual trees are the product of the layout process, and are used for rendering, hit-testing,
/// and navigation.
#[derive(Default)]
struct VisualTree {
    blocks: Vec<Block>,
    cells: Vec<VisualCell>,
    text_layouts: Vec<TextLayout>,
    root_box: usize,
}

/// An instance of the projectional editor.
///
/// Handles text input, layout, and rendering of a document.
pub(crate) struct Editor {
    document: DocumentBox,
    config: EditorConfig,
    metrics: EditorMetrics,
    visual_tree: VisualTree,
    /// Maps line index to first visual cell index on that line.
    line_starts: Vec<usize> = vec![],
    /// Current editing cursor.
    cursor: EditCursor = EditCursor { .. },
    /// Current selection anchor.
    selection_anchor: Option<EditCursor> = None,
}

impl Editor {
    pub fn new(document: DocumentBox, config: &EditorConfig) -> Self {
        let mut this = Self {
            document,
            config: config.clone(),
            metrics: EditorMetrics { font_size: 0, width_unit: 0, indent: 2, line_height: 0 },
            visual_tree: VisualTree::default(),
            ..
        };
        // Set default font size, which also computes the metrics.
        this.set_font_size(24);
        this
    }

    pub fn document_mut(&mut self) -> &mut Document {
        &mut self.document
    }

    pub fn set_font_size(&mut self, font_size: i32) {
        self.metrics.font_size = font_size;
        let font_size_f32 = font_size as f32;
        self.metrics.width_unit = self.config.font.h_advance_char(' ', font_size_f32).ceil() as i32;
        self.metrics.line_height = (font_size_f32 + self.config.font.line_gap(font_size_f32)).ceil() as i32;
    }

    /*/// Updates line starts.
    fn update_line_starts(&mut self) {

        // Visual cells aren't necessarily ordered like text.
        // The next visual cell in logical order may be on the same line, or on the prev or next line.

        self.line_starts.clear();
        let mut current_line_start = 0;
        let mut current_line_y = 0;
        for (i, cell) in self.cells.iter().enumerate() {
            if cell.rect.min.y > current_line_y {
                // New line detected.
                self.line_starts.push(current_line_start);
                current_line_start = i;
                current_line_y = cell.rect.min.y;
            }
        }
        // Add the last line start.
        self.line_starts.push(current_line_start);
    }*/

    /// Returns the default text format for the editor.
    fn text_format(&self) -> TextFormat {
        TextFormat {
            font: self.config.font.clone(),
            size: self.metrics.font_size as f32,
            color: self.config.colors.text,
            ..
        }
    }

    /*/// Returns the vertical neighbor cell.
    fn move_to_vertical(&self, direction: VerticalMoveDirection) -> Option<usize> {

        let ref_pos = &self.cells[self.cursor.cell_index].rect.min;

        let ldir = match direction {
            VerticalMoveDirection::Up => -1,
            VerticalMoveDirection::Down => 1,
        };

        let ncells = self.cells.len() as isize;

        // Move in the logical direction until there is a baseline change.
        let mut index = self.cursor.cell_index as isize;

        let target_y = loop {
            index += ldir;
            if index < 0 || index >= ncells {
                return None;
            }
            let y = self.cells[index as usize].rect.min.y;
            if y != ref_pos.y {
                // the baseline changed.
                break y;
            }
        };

        // Continue moving in the logical direction, staying on the same target baseline (target_y),
        // looking for a cell that minimizes the horizontal distance to the current cell.
        // Stop when the baseline changes again.

        let mut closest = index;
        let mut min_x_diff = i32::MAX;
        loop {
            let pos = self.cells[index as usize].rect.min;
            if pos.y != target_y {
                // we moved again in the inline direction
                break;
            }
            let x_diff = i32::abs(pos.x - ref_pos.x);
            if x_diff < min_x_diff {
                min_x_diff = x_diff;
                closest = index;
            }


            index += ldir;
            if index < 0 || index >= ncells {
                return None;
            }
            let (cell_inline, cell_cross) = to_inline_cross(cell.rect.min);
            if cell_inline != cur_inline {
                // we moved in the inline direction.
                break;
            }
            if (cell_cross - cur_cross).abs() < (cur_cross - cur_cross).abs() {
                return Some(index as usize);
            }
        }
    }*/


    /// Moves the cursor in the inline direction.
    pub fn move_to_logical(&mut self, logical_direction: LogicalCursorDirection) {
        let direction = match logical_direction {
            LogicalCursorDirection::Forward => 1,
            LogicalCursorDirection::Backward => -1,
        };
        let ncells = self.visual_tree.cells.len() as isize;

        let mut index = self.cursor.pos as isize;

        index += direction;

        if index < 0 || index >= ncells {
            match logical_direction {
                LogicalCursorDirection::Forward => debug!("cursor at end of document"),
                LogicalCursorDirection::Backward => debug!("cursor at start of document"),
            }
            return;
        }

        self.cursor.pos = index as usize;
        self.cursor.text_offset = 0; // Reset text offset when moving to a new cell.
    }

    /// Handles input events directed at the editor.
    pub fn handle_input(&mut self, input: &InputEvent) {

        // Handle cursor movement keys.
        if input.is_shortcut(NamedKey::ArrowLeft) {
            self.move_to_logical(LogicalCursorDirection::Backward);
        } else if input.is_shortcut(NamedKey::ArrowRight) {
            self.move_to_logical(LogicalCursorDirection::Forward);
        }
        else if input.is_shortcut(NamedKey::ArrowUp) {

        }
        else if input.is_shortcut(NamedKey::ArrowDown) {

        }

    }

    fn handle_key_event(&mut self, key: &KeyboardEvent) {
        //if key.
    }

    pub fn layout(&mut self) {
        self.visual_tree = layout::layout(&self.document.root(), &self.metrics, &self.text_format());
    }

    /// Paints the editor.
    pub fn paint(&mut self, scene: &mut PaintScene) {
        scene.clear(self.config.colors.background);
        for cell in self.visual_tree.cells.iter() {
            if let Some(idx) = cell.text_layout_idx {
                let text_layout = &self.visual_tree.text_layouts[idx];
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
        // Draw cursor
        if let Some(cursor_cell) = self.visual_tree.cells.get(self.cursor.pos) {
            let cursor_x = cursor_cell.rect.min.x + self.cursor.text_offset as i32 * self.metrics.width_unit;
            let cursor_rect = IRect::from_origin_size(
                ivec2(cursor_x, cursor_cell.rect.min.y),
                ivec2(1, self.metrics.line_height),
            );
            scene.fill_rect(cursor_rect.to_rect(), self.config.colors.cursor);
        }
    }
}
