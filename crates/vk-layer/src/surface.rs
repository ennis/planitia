use crate::INSTANCE_MAP;
use ash::vk;
use dashmap::DashMap;
use std::ffi::c_void;
use std::sync::LazyLock;
use windows::Win32::Foundation::HWND;

struct SurfaceInfo {
    hwnd: HWND,
}

unsafe impl Send for SurfaceInfo {}
unsafe impl Sync for SurfaceInfo {}

static SURFACES: LazyLock<DashMap<vk::SurfaceKHR, SurfaceInfo>> = LazyLock::new(DashMap::new);

#[unsafe(no_mangle)]
unsafe extern "system" fn layer_vkCreateWin32SurfaceKHR(
    instance: vk::Instance,
    p_create_info: *const vk::Win32SurfaceCreateInfoKHR<'_>,
    p_allocator: *const vk::AllocationCallbacks<'_>,
    p_surface: *mut vk::SurfaceKHR,
) -> vk::Result {
    let dispatch = INSTANCE_MAP.get(&instance).unwrap();

    let hwnd = HWND((*p_create_info).hwnd as *mut c_void);
    let surface_info = SurfaceInfo { hwnd };

    let result = (dispatch.khr_win32_surface.create_win32_surface_khr)(instance, p_create_info, p_allocator, p_surface);
    if result == vk::Result::SUCCESS {
        SURFACES.insert(*p_surface, surface_info);
    }
    result
}

const _: vk::PFN_vkCreateWin32SurfaceKHR = layer_vkCreateWin32SurfaceKHR;

/// Returns the HWND corresponding to the specified VkSurface handle.
pub fn get_hwnd_for_surface(surface: vk::SurfaceKHR) -> Option<HWND> {
    SURFACES.get(&surface).map(|info| info.hwnd)
}
