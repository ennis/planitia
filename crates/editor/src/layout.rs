//! Definition of layout cells.
use gamelib::color::Srgba8;
use crate::model::LinkDecl;

/// Text underline style.
#[derive(Debug, Clone, Copy)]
pub struct Underline {
    pub color: Srgba8,
    pub thickness: f32,
}

/// Cell border style.
#[derive(Debug, Clone, Copy)]
pub struct Border {
    pub color: Srgba8,
    pub thickness: f32,
}

/// Cell style options.
#[derive(Debug, Clone, Copy)]
pub struct Style {
    pub font: u32 = 0,
    pub baseline: Option<f32> = None,
    pub text_color: Option<Srgba8> = None,
    pub background_color: Option<Srgba8> = None,
    pub underline: Option<Underline> = None,
    pub border: Option<Border> = None,
    pub punctuation_left: bool = false,
    pub punctuation_right: bool = false,
}

pub static DEFAULT_STYLE: Style = Style { .. };

impl Default for Style {
    fn default() -> Self {
        Style { .. }
    }
}

/// Cell layout options.
#[derive(Debug, Clone, Copy)]
pub struct LayoutProperties {
    /// Indentation for this cell, if it's the first in a line.
    pub indent: Option<u32> = None,
    /// Ends the current line after this cell.
    pub newline_after: bool = false,
    /// Ends the current line before this cell.
    pub newline_before: bool = false,
}

impl Default for LayoutProperties {
    fn default() -> Self {
        LayoutProperties { .. }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct LTextCell<'a> {
    pub text: &'a str,
    pub style: &'a Style,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum OuterLayoutMode {
    /// Box is displayed inline with the current flow.
    Inline,
    /// Box is displayed as a block element, starting on a new line.
    Block,
    /// Box isn't displayed at all.
    None,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum LayoutMode {
    /// Child nodes are laid out as blocks, horizontally.
    Horizontal,
    /// Child nodes are laid out as blocks, vertically.
    Vertical,
    /// Child nodes are laid out inline, wrapping to the next line as needed.
    Inline,
}

#[derive(Debug, Clone, Copy)]
pub struct LChild<'a> {
    pub link: &'a LinkDecl<'a>,
    pub style: &'a Style,
}

#[derive(Debug)]
#[repr(C)]
pub struct LCollection<'a> {
    pub direction: LayoutMode,
    pub children: &'a [LCell<'a>],
}

#[derive(Debug, Clone, Copy)]
pub struct LChildCollection<'a> {
    pub direction: LayoutMode,
    pub children: &'a LinkDecl<'a>,
}

/// Layout cell.
///
/// Every node expands to one layout cell.
#[derive(Debug, Clone, Copy)]
pub enum LCell<'a> {
    /// Text element.
    Text(&'a LTextCell<'a>),
    /// Child node.
    Child(&'a LChild<'a>),
    /// Collection of cells.
    Collection(&'a LCollection<'a>),
    /// Collection of child nodes.
    ChildCollection(&'a LChildCollection<'a>),
}

#[doc(hidden)]
#[macro_export]
macro_rules! layout_seq {
    ( $node:ident () [$($acc:tt)*] ) => {
        [ $($acc)* ]
    };
    ( $node:ident ($kw:literal $($rest:tt)*) [$($acc:tt)*] ) => {
        $crate::layout_seq!($node ($($rest)*) [ $($acc)* $crate::layout!($node $kw), ])
    };
    ( $node:ident (% $kw:ident $($rest:tt)*) [$($acc:tt)*] ) => {
        $crate::layout_seq!($node ($($rest)*) [ $($acc)* $crate::layout!($node % $kw), ])
    };
    ( $node:ident (@ $kw:ident $($rest:tt)*) [$($acc:tt)*] ) => {
        $crate::layout_seq!($node ($($rest)*) [ $($acc)* $crate::layout!($node @ $kw), ])
    };
    ( $node:ident ([H $($contents:tt)*] $($rest:tt)*) [$($acc:tt)*] ) => {
        $crate::layout_seq!($node ($($rest)*) [ $($acc)* $crate::layout!($node [H $($contents)*]), ])
    };
    ( $node:ident ([V $($contents:tt)*] $($rest:tt)*) [$($acc:tt)*] ) => {
        $crate::layout_seq!($node ($($rest)*) [ $($acc)* $crate::layout!($node [V $($contents)*]), ])
    };
    ( $node:ident ([I $($contents:tt)*] $($rest:tt)*) [$($acc:tt)*] ) => {
        $crate::layout_seq!($node ($($rest)*) [ $($acc)* $crate::layout!($node [I $($contents)*]), ])
    };
}

#[macro_export]
macro_rules! layout {

    // Value cell
    ($node:ident % $link:ident) => {
        $crate::layout::LCell::Child(&$crate::layout::LChild { link: &$node::$link, style: &$crate::layout::DEFAULT_STYLE })
    };

    // Text cell
    ($node:ident $kw:literal) => {
        $crate::layout::LCell::Text(&$crate::layout::LTextCell { text: $kw, style: &$crate::layout::DEFAULT_STYLE })
    };

    // Horizontal collection
    ($node:ident [H $($contents:tt)* ]) => {
        $crate::layout::LCell::Collection(&$crate::layout::LCollection { direction: $crate::layout::LayoutMode::Horizontal, children: &$crate::layout_seq!($node ($($contents)*) []) })
    };

    // Vertical collection
    ($node:ident [V $($contents:tt)* ]) => {
        $crate::layout::LCell::Collection(&$crate::layout::LCollection { direction: $crate::layout::LayoutMode::Vertical, children: &$crate::layout_seq!($node ($($contents)*) []) })
    };

    ($node:ident [I $($contents:tt)* ]) => {
        $crate::layout::LCell::Collection(&$crate::layout::LCollection { direction: $crate::layout::LayoutMode::Inline, children: &$crate::layout_seq!($node ($($contents)*) []) })
    };

    // Value collection (vertical)
    ($node:ident (V % $link:ident)) => {
        $crate::layout::LCell::ChildCollection(&$crate::layout::LChildCollection { direction: $crate::layout::LayoutMode::Vertical, children: &$node::$link })
    };

    // Value collection (horizontal)
    ($node:ident (H % $link:ident)) => {
        $crate::layout::LCell::ChildCollection(&$crate::layout::LChildCollection { direction: $crate::layout::LayoutMode::Horizontal, children: &$node::$link })
    };

    // Value collection (inline)
    ($node:ident (I % $link:ident)) => {
        $crate::layout::LCell::ChildCollection(&$crate::layout::LChildCollection { direction: $crate::layout::LayoutMode::Inline, children: &$node::$link })
    };
}

pub use layout_seq;
pub use layout;

