
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


macro_rules! dispatch_table {
    ($name:ident; $([$inherits_m:ident:$inherits_ty:ty])? $($cmd:ident ($($argname:ident: $args:ty),*) -> $rty:ty,$pfn:ty,$procname:literal;)*) => {
        #[derive(Copy, Clone)]
        #[repr(C)]
        pub struct $name {
            $(pub $inherits_m : $inherits_ty,)?
            $(pub $cmd: $pfn,)*
        }
        impl $name {
            pub unsafe fn load_with(mut load_fn: impl FnMut(&CStr) -> PFN_vkVoidFunction) -> Self {
                $(unsafe extern "system" fn $cmd($(_ : $args),*) -> $rty {
                    $crate::proc_not_found($procname);
                })*
                Self {
                    $($inherits_m: unsafe { <$inherits_ty>::load_with(&mut load_fn) },)?
                    $($cmd: unsafe { if let Some(f) = load_fn($procname) { ::core::mem::transmute(f) } else { $cmd } },)*
                }
            }
        }
        $(impl ::core::ops::Deref for $name {
            type Target = $inherits_ty;
            fn deref(&self) -> &Self::Target {
                &self.$inherits_m
            }
        })?
    };
}
pub(crate) use dispatch_table;


/// Convenience macro to call vulkan functions that return arrays via a count/output pointer pair.
#[macro_export]
macro_rules! vkarraycall {
    ($p:ident$(.$ps:ident)* ($($args:expr,)* @count let $count:ident, @out let $array:ident)) => {
        vkarraycall!(let _ = $p$(.$ps)* ($($args,)* @count let $count, @out let $array));
    };
    (let $result:pat = $p:ident$(.$ps:ident)* ($($args:expr,)* @count let $count:ident, @out let $array:ident)) => {
        let mut $count = 0;
        let mut $array = vec![];
        let __result = $p$(.$ps)*($($args,)* &mut $count, ::core::ptr::null_mut());
        if __result < 0 {
            $crate::panic_vulkan_api_call_failed(__result);
        }
        $array.reserve($count as usize);
        let __result = $p$(.$ps)*($($args,)* &mut $count, $array.as_mut_ptr());
        if __result < 0 {
            $crate::panic_vulkan_api_call_failed(__result);
        }
        let $result = __result;
        unsafe { $array.set_len($count as usize); }
    };
}

/// Same as [`vkarraycall`] but without result checks.
#[macro_export]
macro_rules! vkarraycallnc {
    ($p:ident$(.$ps:ident)* ($($args:expr,)* @count let $count:ident, @out let $array:ident)) => {
        let mut $count = 0;
        let mut $array = vec![];
        $p$(.$ps)*($($args,)* &mut $count, ::core::ptr::null_mut());
        $array.reserve($count as usize);
        $p$(.$ps)*($($args,)* &mut $count, $array.as_mut_ptr());
        unsafe { $array.set_len($count as usize); }
    };
}

/// Convenience macro to call vulkan functions that return results via output pointer parameters.
#[macro_export]
macro_rules! vkcallnc {
    ($p:ident$(.$ps:ident)* ($($args:expr,)* $(@out let $out:ident),*)) => {
        $(let mut $out = ::core::mem::MaybeUninit::uninit();)*
        let _ = $p$(.$ps)*($($args,)* $($out.as_mut_ptr()),*);
        $(let $out = unsafe { $out.assume_init() };)*
    };
}

/// Same as [`vkcallnc`] but panics on an unsuccessful result, and puts the VkResult in a variable.
#[macro_export]
macro_rules! vkcall {
     ($p:ident$(.$ps:ident)* ($($args:expr),*)) => {
        let __result = $p$(.$ps)*($($args),*);
        if __result < 0 {
            $crate::panic_vulkan_api_call_failed(__result);
        }
    };
    ($p:ident$(.$ps:ident)* ($($args:expr,)* $(@out let $out:ident),*)) => {
        $(let mut $out = ::core::mem::MaybeUninit::uninit();)*
        let __result = $p$(.$ps)*($($args,)* $($out.as_mut_ptr()),*);
        if __result < 0 {
            $crate::panic_vulkan_api_call_failed(__result);
        }
        $(let $out = unsafe { $out.assume_init() };)*
    };
    (let $result:pat = $p:ident$(.$ps:ident)* ($($args:expr,)* $(@out let $out:ident),*)) => {
        $(let mut $out = ::core::mem::MaybeUninit::uninit();)*
        let __result = $p$(.$ps)*($($args,)* $($out.as_mut_ptr()),*);
        if __result < 0 {
            $crate::panic_vulkan_api_call_failed(__result);
        }
        let $result = __result;
        $(let $out = unsafe { $out.assume_init() };)*
    };
}
