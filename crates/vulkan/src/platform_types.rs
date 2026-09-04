use std::ffi::{c_uint, c_ulong, c_void};

macro_rules! opaque_type {
    ($name:ident) => {
        #[repr(C)]
        pub struct $name {
            _data: (),
            _marker: ::core::marker::PhantomData<(*mut u8, ::core::marker::PhantomPinned)>,
        }
    };
}

opaque_type!(Display);
opaque_type!(ANativeWindow);
opaque_type!(AHardwareBuffer);
opaque_type!(CAMetalLayer);
opaque_type!(IDirectFB);
opaque_type!(IDirectFBSurface);
opaque_type!(__IOSurface);
opaque_type!(xcb_connection_t);
opaque_type!(wl_display);
opaque_type!(wl_surface);
opaque_type!(_screen_buffer);
opaque_type!(_screen_context);
opaque_type!(_screen_window);
opaque_type!(SECURITY_ATTRIBUTES);

// This part below adapted from ash (https://github.com/ash-rs/ash)
//
// platform_types.rs
//
// Copyright (c) 2016 ASH
//
// Permission is hereby granted, free of charge, to any
// person obtaining a copy of this software and associated
// documentation files (the "Software"), to deal in the
// Software without restriction, including without
// limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software
// is furnished to do so, subject to the following
// conditions:
//
// The above copyright notice and this permission notice
// shall be included in all copies or substantial portions
// of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF
// ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
// TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
// PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT
// SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
// IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
pub type RROutput = c_ulong;
pub type VisualID = c_uint;
pub type Window = c_ulong;
pub type xcb_window_t = u32;
pub type xcb_visualid_t = u32;
pub type MirConnection = *const c_void;
pub type MirSurface = *const c_void;
/// <https://microsoft.github.io/windows-docs-rs/doc/windows/Win32/Foundation/struct.HANDLE.html>
pub type HANDLE = isize;
/// <https://microsoft.github.io/windows-docs-rs/doc/windows/Win32/Foundation/struct.HINSTANCE.html>
pub type HINSTANCE = HANDLE;
/// <https://microsoft.github.io/windows-docs-rs/doc/windows/Win32/Foundation/struct.HWND.html>
pub type HWND = HANDLE;
/// <https://microsoft.github.io/windows-docs-rs/doc/windows/Win32/Graphics/Gdi/struct.HMONITOR.html>
pub type HMONITOR = HANDLE;
pub type DWORD = c_ulong;
pub type LPCWSTR = *const u16;
pub type zx_handle_t = u32;
// This definition is behind an NDA with a best effort guess from
// https://github.com/google/gapid/commit/22aafebec4638c6aaa77667096bca30f6e842d95#diff-ab3ab4a7d89b4fc8a344ff4e9332865f268ea1669ee379c1b516a954ecc2e7a6R20-R21
pub type GgpStreamDescriptor = u32;
pub type GgpFrameToken = u64;
pub type IOSurfaceRef = *mut __IOSurface;
pub type MTLBuffer_id = *mut c_void;
pub type MTLCommandQueue_id = *mut c_void;
pub type MTLDevice_id = *mut c_void;
pub type MTLSharedEvent_id = *mut c_void;
pub type MTLTexture_id = *mut c_void;
