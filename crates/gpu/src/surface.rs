#[cfg(windows)]
mod platform {
    use crate::{vkcall, Instance};
    use crate::instance::get_vulkan_entry;
    use raw_window_handle::RawWindowHandle;
    use std::ptr;
    use std::sync::LazyLock;
    use vulkan::*;

    pub fn create_vulkan_surface(handle: RawWindowHandle) -> VkSurfaceKHR {
        static KHR_WIN32_SURFACE: LazyLock<khr_win32_surface::InstanceDispatch> = LazyLock::new(|| unsafe {
            khr_win32_surface::InstanceDispatch::load_with(|name| {
                let instance = Instance::get().instance;
                get_vulkan_entry().GetInstanceProcAddr(instance, name.as_ptr())
            })
        });

        let win32_handle = match handle {
            RawWindowHandle::Win32(h) => h,
            _ => panic!("incompatible window handle"),
        };
        let create_info = VkWin32SurfaceCreateInfoKHR {
            flags: Default::default(),
            hinstance: win32_handle.hinstance.unwrap().get() as HINSTANCE,
            hwnd: win32_handle.hwnd.get() as HWND,
            ..
        };
        unsafe {
            let vk_instance = Instance::get().instance;
            vkcall!(KHR_WIN32_SURFACE.CreateWin32SurfaceKHR(vk_instance, &create_info, ptr::null(), @out let surface));
            surface
        }
    }
}

pub use self::platform::create_vulkan_surface;
