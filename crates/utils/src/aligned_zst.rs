//! ZST types with alignment requirements specified by const generics.
//!
//! Useful to set the alignment requirements of a type from a const generic parameter, instead of
//! `#[repr(align(N))]`.

use std::hash::Hash;

mod private {
    pub trait Sealed {}
}

#[doc(hidden)]
pub struct AlignedLookup;

impl private::Sealed for AlignedLookup {}

pub trait AlignedTrait<const N: usize>: private::Sealed {
    #[doc(hidden)]
    type Aligned: Sized + Default + Copy + Ord + Hash + Send + Sync + 'static;
}

macro_rules! impl_aligned {
    ( $($aligned_ty:ident: $align:literal),* ) => {
        $(
            #[doc(hidden)]
            #[derive(Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
            #[repr(align($align))]
            pub struct $aligned_ty;

            impl AlignedTrait<$align> for AlignedLookup {
                type Aligned = $aligned_ty;
            }
        )*
    };
}

impl_aligned!(
    Aligned1: 1,
    Aligned2: 2,
    Aligned4: 4,
    Aligned8: 8,
    Aligned16: 16,
    Aligned32: 32,
    Aligned64: 64,
    Aligned128: 128,
    Aligned256: 256,
    Aligned512: 512,
    Aligned1024: 1024,
    Aligned2048: 2048,
    Aligned4096: 4096
);

#[repr(transparent)]
#[derive(Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AlignedZst<const N: usize>([<AlignedLookup as AlignedTrait<N>>::Aligned; 0])
where
    AlignedLookup: AlignedTrait<N>;
