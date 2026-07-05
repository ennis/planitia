use crate::layout::LCell;
use std::alloc::{Layout, alloc};
use std::cell::RefCell;
use std::ptr;
use std::rc::Rc;


/// Document state.
pub struct DocumentState {

    /// Root node of the document tree.
    pub root: Node,
    /// Previous versions of the document tree, for undo/redo.
    pub history: Vec<Node>,
    /// Index of the current version in the history.
    pub history_index: usize,
}


/// Strong ref to a generic node in the document tree.
pub type Node = Rc<NodeData>;

/// Weak reference to a node in the document tree.
pub type WeakNode = std::rc::Weak<NodeData>;

/// Represents a value in a node property.
#[derive(Clone, PartialEq)]
pub enum Value {
    String(String),
    Number(f64),
    Boolean(bool),
}


// We can get away with using UnsafeCell instead of RefCell because we lock the node tree behind
// accessor functions that only return copies:
// - either copies of Rc<Node>
// - or copies of the value inside a RefCell<NodeChild>.
// This means that it's impossible to have a reference to a child node *without* also owning a strong
// ref to the node.

// Alternatives to Rc?
// - Allocate nodes in a big arena; no way to deallocate individual nodes, but trivial to implement.
// - "garbage collection" is basically copying the live nodes to a new arena and dropping the old one.
//      - might be viable
//
// - Issue: how to ensure that pointers can't "escape" the tree?
//     - when there's a reference, the document is borrowed (shared borrow), so it can't be "garbage collected" while the reference is alive.

#[derive(Clone)]
pub enum NodeChild {
    // All variants are options because the representation is meant for editing,
    // and when a node is just inserted the "no value" state makes sense.
    // It would not represent a valid document tree, but it is a valid editing state.
    Value(RefCell<Option<Value>>),
    Node(RefCell<Option<Node>>),
    Collection(RefCell<Vec<Node>>),
    RefNode(RefCell<Option<WeakNode>>),
    RefCollection(RefCell<Vec<WeakNode>>),
}

impl NodeChild {
    pub fn as_value(&self) -> Option<&RefCell<Option<Value>>> {
        match self {
            NodeChild::Value(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_node(&self) -> Option<&RefCell<Option<Node>>> {
        match self {
            NodeChild::Node(n) => Some(n),
            _ => None,
        }
    }

    pub fn as_collection(&self) -> Option<&RefCell<Vec<Node>>> {
        match self {
            NodeChild::Collection(c) => Some(c),
            _ => None,
        }
    }

    pub fn as_ref_node(&self) -> Option<&RefCell<Option<WeakNode>>> {
        match self {
            NodeChild::RefNode(r) => Some(r),
            _ => None,
        }
    }

    pub fn as_ref_collection(&self) -> Option<&RefCell<Vec<WeakNode>>> {
        match self {
            NodeChild::RefCollection(r) => Some(r),
            _ => None,
        }
    }
}

#[repr(C)]
struct NodeDataHeader {
    parent: Option<WeakNode>,
    decl: &'static NodeDecl<'static>,
}

#[repr(C)]
pub struct NodeData {
    h: NodeDataHeader,
    children: [NodeChild],
}

impl NodeData {
    fn alloc_layout(count: usize) -> (Layout, usize) {
        let (layout, array_offset) =
            Layout::new::<NodeDataHeader>().extend(Layout::array::<NodeChild>(count).unwrap()).unwrap();
        (layout.pad_to_align(), array_offset)
    }
}

impl NodeData {
    pub(crate) fn new(decl: &'static NodeDecl<'static>) -> Node {
        // We can't construct NodeData directly since it's a DST. Instead, we construct
        // a `NodeDataInit<[T; N]>`, which is then unsize-coerced to `NodeDataInit<[T]>`, which we
        // can then reinterpret as `NodeData` since they have the same layout.

        let n = decl.children.len();
        let (layout, child_offset) = NodeData::alloc_layout(n);
        let node = unsafe {
            // Allocate header + children array.
            let ptr = alloc(layout);
            let child_ptr = ptr.add(child_offset) as *mut RefCell<NodeChild>;

            // Write the header.
            ptr::write(ptr as *mut NodeDataHeader, NodeDataHeader { parent: None, decl });

            // Initialize children array to default values.
            for i in 0..n {
                let init = match (decl.children[i].ty, decl.children[i].multiplicity) {
                    (Type::Primitive(_), _) => NodeChild::Value(RefCell::new(None)),
                    (Type::NodeInstance(_), Multiplicity::One | Multiplicity::Optional) => {
                        NodeChild::Node(RefCell::new(None))
                    }
                    (Type::NodeInstance(_), Multiplicity::Many | Multiplicity::Many1) => {
                        NodeChild::Collection(RefCell::new(Vec::new()))
                    }
                    (Type::NodeRef(_), Multiplicity::One | Multiplicity::Optional) => {
                        NodeChild::RefNode(RefCell::new(None))
                    }
                    (Type::NodeRef(_), Multiplicity::Many | Multiplicity::Many1) => {
                        NodeChild::RefCollection(RefCell::new(Vec::new()))
                    }
                };
                ptr::write(child_ptr.add(i), RefCell::new(init));
            }

            // Cast it to a dummy slice of `[NodeDataHeader]` to set the length metadata.
            // The resulting pointer is not valid for such an array, but it will never be read from,
            // and is only there to build a fat pointer with length metadata.
            let ptr = ptr::slice_from_raw_parts_mut(ptr, n);
            // Now cast it to the final type, and wrap in Box.
            let ptr = ptr as *mut NodeData;
            // SAFETY: ptr points to memory allocated with the global allocator, and its layout
            //         is valid for an instance NodeData.
            Box::from_raw(ptr)
        };

        // Convert to Rc
        let rc_node = Rc::from(node);
        rc_node
    }

    /// Returns the layout description of this node, if any.
    pub fn layout(&self) -> Option<&LCell> {
        self.h.decl.layout
    }

    pub fn child(&self, index: usize) -> Option<&NodeChild> {
        self.children.get(index)
    }
}

/// Declares the structure of a node.
///
/// (a.k.a. "Concept" in MPS)
#[derive(Copy, Clone, Debug)]
pub struct NodeDecl<'a> {
    /// Name of the node type.
    pub name: &'a str,
    /// Children.
    pub children: &'a [&'a LinkDecl<'a>],
    /// Constructor function.
    pub ctor: fn() -> Node,
    /// Layout description.
    pub layout: Option<&'a LCell<'a>> = None,
}

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub enum Multiplicity {
    /// Exactly one child.
    One,
    /// Zero or one child.
    Optional,
    /// Zero or more children.
    Many,
    /// One or more children.
    Many1,
}

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub enum PrimitiveType {
    String,
    Number,
    Boolean,
}

/// Describes the type of a property.
#[derive(Copy, Clone, Debug)]
pub enum Type<'a> {
    Primitive(PrimitiveType),
    NodeInstance(&'a NodeDecl<'a>),
    NodeRef(&'a NodeDecl<'a>),
}

/// Describes a collection of child nodes.
#[derive(Copy, Clone, Debug)]
pub struct LinkDecl<'a> {
    pub parent: &'a NodeDecl<'a>,
    pub index: usize,
    pub name: &'a str,
    pub ty: Type<'a>,
    pub multiplicity: Multiplicity,
}

#[macro_export]
macro_rules! declare_node {
    // ------------------------------------------------------------------ entry
    (
        $(#[$attr:meta])*
        $vis:vis static $name:ident : $decl_name:literal {
            $( $field_name:ident : ( $($field_ty:tt)+ ) )*
        } $(=> $($layout:tt)*)?
    ) => {
        #[allow(non_snake_case)]
        #[allow(unused_imports)]
        pub mod $name {
            use super::*;

            #[repr(usize)]
            #[allow(dead_code, non_camel_case_types)]
            enum FieldIndex { $($field_name,)* __Count }

            $(
                #[allow(non_upper_case_globals)]
                pub static $field_name: $crate::model::LinkDecl = $crate::model::LinkDecl {
                    parent: &super::$name,
                    index: FieldIndex::$field_name as usize,
                    name: stringify!($field_name),
                    ty: $crate::declare_node!(@ty $($field_ty)+),
                    multiplicity: $crate::declare_node!(@mult $($field_ty)+),
                };
            )*

            //pub(super) use self::{ $($field_name,)* };
        }

        $(#[$attr])*
        $vis static $name: $crate::model::NodeDecl<'static> = {

            fn ctor() -> $crate::model::Node {
                $crate::model::NodeData::new(&$name)
            }

            $crate::model::NodeDecl {
                name: $decl_name,
                ctor: ctor,
                //traits: &[],
                children: &[
                    $(&$name::$field_name),*
                ],
                $(layout: Some(&$crate::layout!($name $($layout)*)),)?
                ..
            }
        };
    };

    // ------------------------------------------------------------------ @ty
    (@ty str)                  => { $crate::model::Type::Primitive($crate::model::PrimitiveType::String)  };
    (@ty str ?)                => { $crate::model::Type::Primitive($crate::model::PrimitiveType::String)  };
    (@ty num)                  => { $crate::model::Type::Primitive($crate::model::PrimitiveType::Number)  };
    (@ty num ?)                => { $crate::model::Type::Primitive($crate::model::PrimitiveType::Number)  };
    (@ty bool)                 => { $crate::model::Type::Primitive($crate::model::PrimitiveType::Boolean) };
    (@ty bool ?)               => { $crate::model::Type::Primitive($crate::model::PrimitiveType::Boolean) };
    (@ty $node:ident)          => { $crate::model::Type::NodeInstance(&$node) };
    (@ty $node:ident ?)        => { $crate::model::Type::NodeInstance(&$node) };
    (@ty [ $node:ident ])      => { $crate::model::Type::NodeInstance(&$node) };
    (@ty [ $node:ident ] +)    => { $crate::model::Type::NodeInstance(&$node) };
    (@ty @ $node:ident)        => { $crate::model::Type::NodeRef(&$node) };
    (@ty @ $node:ident ?)      => { $crate::model::Type::NodeRef(&$node) };
    (@ty [ @ $node:ident ])    => { $crate::model::Type::NodeRef(&$node) };
    (@ty [ @ $node:ident ] +)  => { $crate::model::Type::NodeRef(&$node) };

    // --------------------------------------------------------------- @mult
    (@mult str)                => { $crate::model::Multiplicity::One      };
    (@mult str ?)              => { $crate::model::Multiplicity::Optional };
    (@mult num)                => { $crate::model::Multiplicity::One      };
    (@mult num ?)              => { $crate::model::Multiplicity::Optional };
    (@mult bool)               => { $crate::model::Multiplicity::One      };
    (@mult bool ?)             => { $crate::model::Multiplicity::Optional };
    (@mult $node:ident)        => { $crate::model::Multiplicity::One      };
    (@mult $node:ident ?)      => { $crate::model::Multiplicity::Optional };
    (@mult [ $node:ident ])    => { $crate::model::Multiplicity::Many     };
    (@mult [ $node:ident ] +)  => { $crate::model::Multiplicity::Many1    };
    (@mult @ $node:ident)      => { $crate::model::Multiplicity::One      };
    (@mult @ $node:ident ?)    => { $crate::model::Multiplicity::Optional };
    (@mult [ @ $node:ident ])  => { $crate::model::Multiplicity::Many     };
    (@mult [ @ $node:ident ] +) => { $crate::model::Multiplicity::Many1   };
}

pub use declare_node;
