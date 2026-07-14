//! Node declarations and related types.
//!
//! [Node declarations](NodeDecl) describe the expected structure of a document [Node](Node).
//! They hold a list of [Field declarations](FieldDecl) which describe its expected fields
//! and their types (primitive, child node, collection of child nodes...).

use crate::layout::LayoutItem;

/// Declares the structure of a node.
///
/// (a.k.a. "Concept" in MPS)
#[derive(Copy, Clone, Debug)]
pub struct NodeDecl<'a> {
    /// Name of the node type.
    pub name: &'a str,
    /// Fields.
    pub fields: &'a [&'a FieldDecl<'a>],
    /// Layout item.
    pub layout: Option<&'a LayoutItem<'a>> = None,
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

#[derive(Copy, Clone, Debug)]
pub struct FieldDecl<'a> {
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
                pub static $field_name: $crate::decl::FieldDecl = $crate::decl::FieldDecl {
                    parent: &super::$name,
                    index: FieldIndex::$field_name as usize,
                    name: stringify!($field_name),
                    ty: $crate::declare_node!(@ty $($field_ty)+),
                    multiplicity: $crate::declare_node!(@mult $($field_ty)+),
                };
            )*
        }

        $(#[$attr])*
        $vis static $name: $crate::decl::NodeDecl<'static> = {
            $crate::decl::NodeDecl {
                name: $decl_name,
                //traits: &[],
                fields: &[
                    $(&$name::$field_name),*
                ],
                $(layout: Some(&$crate::layout!($name $($layout)*)),)?
                ..
            }
        };
    };

    // ------------------------------------------------------------------ @ty
    (@ty str)                  => { $crate::decl::Type::Primitive($crate::decl::PrimitiveType::String)  };
    (@ty str ?)                => { $crate::decl::Type::Primitive($crate::decl::PrimitiveType::String)  };
    (@ty num)                  => { $crate::decl::Type::Primitive($crate::decl::PrimitiveType::Number)  };
    (@ty num ?)                => { $crate::decl::Type::Primitive($crate::decl::PrimitiveType::Number)  };
    (@ty bool)                 => { $crate::decl::Type::Primitive($crate::decl::PrimitiveType::Boolean) };
    (@ty bool ?)               => { $crate::decl::Type::Primitive($crate::decl::PrimitiveType::Boolean) };
    (@ty $node:ident)          => { $crate::decl::Type::NodeInstance(&$node) };
    (@ty $node:ident ?)        => { $crate::decl::Type::NodeInstance(&$node) };
    (@ty [ $node:ident ])      => { $crate::decl::Type::NodeInstance(&$node) };
    (@ty [ $node:ident ] +)    => { $crate::decl::Type::NodeInstance(&$node) };
    (@ty @ $node:ident)        => { $crate::decl::Type::NodeRef(&$node) };
    (@ty @ $node:ident ?)      => { $crate::decl::Type::NodeRef(&$node) };
    (@ty [ @ $node:ident ])    => { $crate::decl::Type::NodeRef(&$node) };
    (@ty [ @ $node:ident ] +)  => { $crate::decl::Type::NodeRef(&$node) };

    // --------------------------------------------------------------- @mult
    (@mult str)                => { $crate::decl::Multiplicity::One      };
    (@mult str ?)              => { $crate::decl::Multiplicity::Optional };
    (@mult num)                => { $crate::decl::Multiplicity::One      };
    (@mult num ?)              => { $crate::decl::Multiplicity::Optional };
    (@mult bool)               => { $crate::decl::Multiplicity::One      };
    (@mult bool ?)             => { $crate::decl::Multiplicity::Optional };
    (@mult $node:ident)        => { $crate::decl::Multiplicity::One      };
    (@mult $node:ident ?)      => { $crate::decl::Multiplicity::Optional };
    (@mult [ $node:ident ])    => { $crate::decl::Multiplicity::Many     };
    (@mult [ $node:ident ] +)  => { $crate::decl::Multiplicity::Many1    };
    (@mult @ $node:ident)      => { $crate::decl::Multiplicity::One      };
    (@mult @ $node:ident ?)    => { $crate::decl::Multiplicity::Optional };
    (@mult [ @ $node:ident ])  => { $crate::decl::Multiplicity::Many     };
    (@mult [ @ $node:ident ] +) => { $crate::decl::Multiplicity::Many1   };
}

pub use declare_node;
