
macro_rules! handle {
    ($h:ident,$s:ident) => {
        #[repr(C)]
        pub struct $s {
            _data: (),
            _marker: ::core::marker::PhantomData<(*mut u8, ::core::marker::PhantomPinned)>,
        }
        #[repr(transparent)]
        #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Ord, PartialOrd)]
        pub struct $h(pub *mut $s);
        unsafe impl Send for $h {}
        unsafe impl Sync for $h {}
        impl $h {
            pub const fn null() -> Self {
                Self(ptr::null_mut())
            }
            pub const fn is_null(self) -> bool {
                self.0.is_null()
            }
        }
        impl Default for $h {
            fn default() -> Self {
                Self::null()
            }
        }
    };
}
pub(crate) use handle;

macro_rules! non_dispatchable_handle {
    ($h:ident) => {
        #[repr(transparent)]
        #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Ord, PartialOrd)]
        pub struct $h(pub u64);
        impl $h {
            pub const fn null() -> Self {
                Self(0)
            }
            pub const fn is_null(self) -> bool {
                self.0 == 0
            }
        }
        impl Default for $h {
            fn default() -> Self {
                Self::null()
            }
        }
    };
}
pub(crate) use non_dispatchable_handle;
