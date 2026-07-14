//! Definition of layout items.
//!
//! Each node declaration has an associated `LayoutItem` that describes how a node and its children are laid out in the editor.

use crate::decl::FieldDecl;
use crate::style::Style;

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

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum LayoutMode {
    /// Block-level element.
    ///
    /// If within an inline layout context, this will break the current line and layout the block on a new line.
    Block,
    /// Inline element.
    Inline,
    // TODO: inline-block elements that are positioned inline but still define new block formatting contexts.
}

/// Layout item.
///
/// Every node expands to one layout item.
#[derive(Debug, Clone, Copy)]
pub struct LayoutItem<'a> {
    pub mode: LayoutMode,
    pub kind: LayoutItemKind<'a>,
}

/// Children of a layout cell
#[derive(Debug, Clone, Copy)]
pub enum LayoutItemKind<'a> {
    /// Text element.
    Text {
        text: &'a str,
        style: &'a Style,
    },
    /// Collection of child layouts.
    Collection(&'a [LayoutItem<'a>]),
    /// Field (which may be a collection).
    Field {
        style: &'a Style,
        link: &'a FieldDecl<'a>,
    },
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

/// Macro to define a [layout item](LayoutItem) for a node declaration.
///
/// # Syntax
///
/// * Define a text cell: `layout!(NodeType "text")`
/// * Define a value cell: `layout!(NodeType %field_name)`
/// * Define a block collection: `layout!(NodeType [V ...])`
/// * Define an inline collection: `layout!(NodeType [I ...])`
#[macro_export]
macro_rules! layout {

    ($node:ident % $link:ident) => {
        $crate::layout::LayoutItem {
            mode: $crate::layout::LayoutMode::Inline,
            kind: $crate::layout::LayoutItemKind::Field {
                link: &$node::$link,
                style: &$crate::style::DEFAULT_STYLE
            }
        }
    };

    ($node:ident $kw:literal) => {
        $crate::layout::LayoutItem {
            mode: $crate::layout::LayoutMode::Inline,
            kind: $crate::layout::LayoutItemKind::Text {
                text: $kw,
                style: &$crate::style::DEFAULT_STYLE
            }
        }
    };

    ($node:ident [V $($contents:tt)* ]) => {
        $crate::layout::LayoutItem {
            mode: $crate::layout::LayoutMode::Block,
            kind: $crate::layout::LayoutItemKind::Collection(&$crate::layout_seq!($node ($($contents)*) []))
        }
    };

    ($node:ident [I $($contents:tt)* ]) => {
        $crate::layout::LayoutItem {
            mode: $crate::layout::LayoutMode::Inline,
            kind: $crate::layout::LayoutItemKind::Collection(&$crate::layout_seq!($node ($($contents)*) []))
        }
    };

    ($node:ident (V % $link:ident)) => {
        $crate::layout::LayoutItem {
            mode: $crate::layout::LayoutMode::Block,
            kind: $crate::layout::LayoutItemKind::Field {
                link: &$node::$link,
                style: &$crate::style::DEFAULT_STYLE
            }
        }
    };

    ($node:ident (I % $link:ident)) => {
        $crate::layout::LayoutItem {
            mode: $crate::layout::LayoutMode::Inline,
            kind: $crate::layout::LayoutItemKind::Field {
                link: &$node::$link,
                style: &$crate::style::DEFAULT_STYLE
            }
        }
    };
}

pub use layout_seq;
pub use layout;

