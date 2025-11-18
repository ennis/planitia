/*enum EntityData {
    SceneObject,
    Task,
    Variable,
    Shader,
    Event,
}*/

//! Constraints on entity data:
//! - must be copyable, relocatable plain-old-data (POD) types
//! - must not contain any references or pointers to other data (IDs are fine)
//! - must not have any destructors
//! - ideally should not have any indeterminate padding bytes, so that hashing is consistent
//!
//! This allows us to trivially copy/clone and serialize entities by memory copy.

use slotmap::{SlotMap, new_key_type};
use std::cell::{RefCell, RefMut};

const ENTITY_DATA_ALIGNMENT: usize = 16;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Entity {
    parent: Option<ID>,
    ty: EntityTypeId,
    /// Pointer to entity data. The concrete type is determined by `ty`.
    data: *const (),
}

new_key_type! {
    pub struct ID;
}

pub struct World {
    entities: RefCell<SlotMap<ID, Entity>>,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct EntityTypeId(pub u16);

macro_rules! gen_entity_data {
    (
        $($data_ty:ident),*
    ) => {
        enum EntityTy {
            $(
                $data_ty($data_ty),
            )*
        }

        $(

        impl EntityTypeId for $data_ty {
            fn entity_type_id(&self) -> EntityTypeId {
                EntityData::$data_ty(self.clone())
            }
        }

        )*
    };
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum Change {
    Added {
        id: ID,
        /// Added entity. The data pointer points to the data segment.
        entity: Entity,
    },
    Removed {
        id: ID,
        /// Pointer in data segment.
        prev_data: *const (),
    },
    Modified {
        id: ID,
        /// Pointer in data segment.
        prev_data: *const (),
        /// Pointer in data segment.
        new_data: *const (),
    },
}

pub struct Batch<'a> {
    world: RefMut<'a, World>,
    changes: Vec<Change>,
    /// Data segment.
    data: bumpalo::Bump<ENTITY_DATA_ALIGNMENT>,
}

impl<'a> Batch<'a> {
    pub fn rollback(&self) {
        // TODO
    }

    pub fn commit(self) {
        // TODO
    }
}
