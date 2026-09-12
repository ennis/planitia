use crate::INSTANCE_MAP;
use vulkan::*;
use dashmap::DashMap;
use std::ffi::c_void;
use std::sync::LazyLock;
use windows::Win32::Foundation::HWND;

struct SurfaceInfo {
    hwnd: HWND,
}

unsafe impl Send for SurfaceInfo {}
unsafe impl Sync for SurfaceInfo {}

static SURFACES: LazyLock<DashMap<VkSurfaceKHR, SurfaceInfo>> = LazyLock::new(DashMap::new);

#[unsafe(no_mangle)]
pub unsafe extern "system" fn layer_vkCreateWin32SurfaceKHR(
    instance: VkInstance,
    p_create_info: *const VkWin32SurfaceCreateInfoKHR,
    p_allocator: *const VkAllocationCallbacks,
    p_surface: *mut VkSurfaceKHR,
) -> VkResult {
    let dispatch = INSTANCE_MAP.get(&instance).unwrap();

    let hwnd = HWND((*p_create_info).hwnd as *mut c_void);
    let surface_info = SurfaceInfo { hwnd };

    let result = (dispatch.CreateWin32SurfaceKHR)(instance, p_create_info, p_allocator, p_surface);
    if result == VK_SUCCESS {
        eprintln!("Registering HWND({}) for VkSurfaceKHR({:?})", hwnd.0 as usize, *p_surface);
        SURFACES.insert(*p_surface, surface_info);
    }
    result
}

const _: PFN_vkCreateWin32SurfaceKHR = layer_vkCreateWin32SurfaceKHR;

/// Returns the HWND corresponding to the specified VkSurface handle.
pub fn get_hwnd_for_surface(surface: VkSurfaceKHR) -> Option<HWND> {
    SURFACES.get(&surface).map(|info| info.hwnd)
}
