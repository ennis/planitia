// Contains portions of code adapted from ash (https://github.com/ash-rs/ash)
//
// entry.rs
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

use crate::{PFN_vkVoidFunction, Vulkan_1_0_EntryDispatch, Vulkan_1_1_EntryDispatch};
use libloading::Library;
use std::error::Error;
use std::fmt;

impl Vulkan_1_0_EntryDispatch {
    pub fn load() -> Result<Vulkan_1_0_EntryDispatch, LoadError> {
        let lib = load_vulkan_lib()?;
        unsafe { Ok(Self::load_with(|proc| get_proc_addr(&lib, proc))) }
    }
}

impl Vulkan_1_1_EntryDispatch {
    pub fn load() -> Result<Vulkan_1_1_EntryDispatch, LoadError> {
        let lib = load_vulkan_lib()?;
        unsafe { Ok(Self::load_with(|proc| get_proc_addr(&lib, proc))) }
    }
}

unsafe fn get_proc_addr(lib: &Library, name: &std::ffi::CStr) -> PFN_vkVoidFunction {
    unsafe {
        match lib.get::<PFN_vkVoidFunction>(name.to_bytes_with_nul()) {
            Ok(symbol) => *symbol,
            Err(_) => panic!("vulkan entry point not found: {}", name.to_string_lossy()),
        }
    }
}

fn load_vulkan_lib() -> Result<Library, LoadError> {
    #[cfg(windows)]
    static LIB_PATH: &str = "vulkan-1.dll";

    #[cfg(all(unix, not(any(target_os = "macos", target_os = "ios", target_os = "android", target_os = "fuchsia"))))]
    const LIB_PATH: &str = "libvulkan.so.1";

    #[cfg(any(target_os = "android", target_os = "fuchsia"))]
    const LIB_PATH: &str = "libvulkan.so";

    unsafe {
        match Library::new(LIB_PATH) {
            Ok(lib) => Ok(lib),
            Err(err) => Err(LoadError(format!("failed to load Vulkan library: {}", err))),
        }
    }
}

/// Entry load error.
#[derive(Debug)]
pub struct LoadError(String);

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Error for LoadError {}
