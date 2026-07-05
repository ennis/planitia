use gamelib::color::Srgba8;
use gamelib::math::{Vec2, vec2};
use gamelib::paint::{Font, PaintScene, TextFormat, TextLayout};
use log::debug;
use std::alloc::Layout;
use std::ops::Range;
use std::ptr;
use std::rc::Rc;
use std::sync::OnceLock;

pub trait Projectable {}

enum Token {
    Number(f64),
    String(String),
}
//---------------------------------------------------------------------

/*
impl<'a> crate::parser::LayoutCollection<'a> {
    pub(crate) fn layout(count: usize) -> (Layout, usize) {
        let (layout, array_offset) =
            Layout::new::<LayoutDirection>().extend(Layout::array::<LayoutNode>(count).unwrap()).unwrap();
        (layout.pad_to_align(), array_offset)
    }
}*/



#[derive(Debug, Clone, Copy)]
pub struct LayoutTextNode<'a> {
    pub text: &'a str,
    pub style: &'a Style,
}

#[derive(Debug, Clone, Copy)]
pub enum LayoutNode<'a> {
    Text(&'a LayoutTextNode<'a>),
    Collection(&'a LayoutCollection<'a>),
}

//---------------------------------------------------------------------


#[derive(Debug, Clone)]
struct DocCell {
    text: Range<usize>,
    style: Style,
    /// Cell layout properties.
    layout: CellLayout,
    /// Computed position after layout.
    position: Vec2,
    /// Text layout.
    text_layout: Option<TextLayout>,
}

pub struct LayoutBuilder<'a> {
    arena: &'a bumpalo::Bump,
    text: String,
    font: Font,
    cells: Vec<DocCell>,
    font_size: f32,
    width_unit: f32,
}

fn editor_font() -> Font {
    static DATA: &[u8] = include_bytes!("../assets/fonts/TX-02-Medium.otf");
    static FONT: OnceLock<Font> = OnceLock::new();
    FONT.get_or_init(|| Font::load_static_font_from_bytes(DATA)).clone()
}

impl<'a> LayoutBuilder<'a> {
    fn new(arena: &'a bumpalo::Bump) -> Self {
        let font_size = 16.0;
        let font = editor_font();
        // The width of the space character in the current font and size gives us the base
        // unit of width for horizontal sizes.
        let width_unit = font.h_advance(font.glyph_id(' '), font_size);
        debug!("DocumentBuilder: width_unit = {width_unit}");
        LayoutBuilder { text: String::new(), font, cells: vec![], font_size: 16.0, width_unit, arena }
    }

    /// Emits a text layout node.
    pub fn add_text(&mut self, text: &str) -> &'a LayoutTextNode {
        let text = self.arena.alloc_str(text);
        let style = self.arena.alloc(Style::default());
        let cell = LayoutTextNode { text, style };
        self.arena.alloc(cell)
    }

    /// Emits a collection of layout nodes.
    pub fn add_collection(
        &mut self,
        direction: LayoutDirection,
        children: &[LayoutNode<'a>],
    ) -> &'a LayoutCollection<'a> {
        let (layout, array_offset) = LayoutCollection::layout(children.len());

        // Allocate space for LayoutCollection and its trailing children.
        let ptr = self.arena.alloc_layout(layout);

        // Write child layout nodes.
        unsafe {
            let array_ptr = ptr.as_ptr().add(array_offset) as *mut LayoutNode;
            ptr::copy_nonoverlapping(children.as_ptr(), array_ptr, children.len());
        }

        // Write LayoutCollection header.
        unsafe {
            *(ptr.as_ptr() as *mut LayoutDirection) = direction;
        }

        // We can't yet construct a fat pointer like `*const LayoutCollection` from directly
        // from base pointer + length metadata (in stable rust).
        // However, we can construct a fat slice pointer (the inner type doesn't matter)
        // via `ptr::slice_from_raw_parts`, and then reinterpret that into `*const LayoutCollection`
        // which transfers the length metadata.
        // This is what the slice_dst crate does internally.
        let slice = ptr::slice_from_raw_parts(ptr.as_ptr(), children.len());
        let collection_ptr = slice as *const LayoutCollection;

        // SAFETY: The pointer is valid and points to a properly initialized LayoutCollection.
        unsafe { &*collection_ptr }
    }

    /// Outputs a vertical flow of cells, with each cell on a new line.
    pub fn add_vertical_collection(&mut self,
                                   children: &[LayoutNode<'a>]) -> &'a LayoutCollection<'a> {
        self.add_collection(LayoutDirection::Vertical, children)
    }

    /// Outputs a horizontal flow of cells, with each cell on the same line.
    pub fn add_horizontal_collection(&mut self,
                                     children: &[LayoutNode<'a>]) -> &'a LayoutCollection<'a> {
        self.add_collection(LayoutDirection::Horizontal, children)
    }

    /// Computes the final positions of the cells.
    fn layout(&mut self) {
        let mut position = vec2(0.0, 0.0);
        for cell in self.cells.iter_mut() {
            cell.position = position;
            let text = &self.text[cell.text.clone()];
            let text_format = TextFormat { size: self.font_size, font: self.font.clone(), ..Default::default() };
            let text_layout = TextLayout::new(&text_format, text);
            let advance = text_layout.size().x;
            cell.text_layout = Some(text_layout);
            position.x += advance + cell.style.h_gap as f32 * self.width_unit;
        }
    }

    pub fn draw(&mut self, scene: &mut PaintScene) {
        for cell in self.cells.iter() {
            let text = &self.text[cell.text.clone()];
            let mut text_format = TextFormat::default();
            text_format.color = cell.style.text_color.unwrap_or(Srgba8::WHITE);
            text_format.size = self.font_size;
            text_format.font = editor_font();
            let text_layout = TextLayout::new(&text_format, text);
            scene.draw_text_layout(cell.position, &text_layout, text_format.color);
        }
    }
}

/// Computes the final cell positions.
fn layout(root: &LayoutNode, out_cells: &mut Vec<DocCell>) {

}

struct DocumentProperties {
    /// Default font.
    font: Font,
    /// Default font size, in pixels.
    font_size: f32,
    /// Atomic unit of width, in pixels.
    ///
    /// Corresponds to the width of a space character in the current font and size.
    width_unit: f32,
    /// Indent size in basic width units.
    indent: u32,
    /// Line height.
    line_height: f32,
}

struct LayoutState<'a> {
    position: Vec2,
    out_cells: &'a mut Vec<DocCell>,
    props: &'a DocumentProperties,
    indent: u32,
    block_start: bool,
}

impl<'a> LayoutState<'a> {
    fn process_node(&mut self, node: &LayoutNode) {
        match node {
            LayoutNode::Text(n) => {
                self.process_text(n);
            }
            LayoutNode::Collection(n) => {
                self.process_collection(n);
            }
        }

    }

    fn process_text(&mut self, text_node: &LayoutTextNode) {

        // Layout text with the document font.
        let text_format = TextFormat { size: self.props.font_size, font: self.props.font.clone(), ..Default::default() };
        let text_layout = TextLayout::new(&text_format, text_node.text);
        let advance = text_layout.size().x;


        let cell = DocCell {
            text: 0..text_node.text.len(),
            style: *text_node.style,
            layout: CellLayout { newline_after: false },
            position: self.position,
            text_layout: None,
        };
        self.out_cells.push(cell);

        self.advance(advance);
    }

    /// Advances current line position.
    fn advance(&mut self, advance: f32) {
        self.position.x += advance;
        self.block_start = false;
    }

    /// Move position to the start of a new block.
    fn move_to_new_block(&mut self) {
        if !self.block_start {
            self.position.x = self.indent as f32 * self.props.width_unit;
            self.position.y += self.props.line_height;
            self.block_start = true;
        }
    }

    fn push_indent(&mut self) {
        self.indent += self.props.indent;
    }

    fn pop_indent(&mut self) {
        self.indent -= self.props.indent;
    }

    fn process_collection(&mut self, collection_node: &LayoutCollection) {
        match collection_node.direction {
            LayoutDirection::Horizontal => {
                for child in collection_node.children.iter() {
                    self.process_node(child);
                }
            }
            LayoutDirection::Vertical => {
                self.move_to_new_block();
                for child in collection_node.children.iter() {
                    self.process_node(child);
                }
            }
            LayoutDirection::Indent => {
                self.push_indent();
                self.move_to_new_block();

                self.pop_indent();
            }
        }
    }
}


#[cfg(test)]
mod tests {
    #[test]
    fn test_document_builder() {}
}
