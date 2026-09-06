use ash::vk;
use core::ptr;
use std::ffi::CStr;
use std::sync::LazyLock;
use vulkan::*;

/// We require vulkan 1.4.
const VK_API_VERSION: u32 = vk::make_api_version(0, 1, 4, 0);

#[cfg(windows)]
static INSTANCE_EXTENSIONS: [&CStr; 4] =
    [c"VK_KHR_get_surface_capabilities2", c"VK_EXT_debug_utils", c"VK_KHR_surface", c"VK_KHR_win32_surface"];

//--------------------------------------------------------------------------------------------------

pub struct Instance {
    pub instance: VkInstance,
    pub fns: Vulkan_1_3_InstanceDispatch,
    pub khr_surface: khr_surface::InstanceDispatch,
}

impl Instance {
    /// Returns the global vulkan instance object.
    pub fn get() -> &'static Instance {
        static VULKAN_INSTANCE: LazyLock<Instance> = LazyLock::new(create_vulkan_instance);
        &VULKAN_INSTANCE
    }
}

pub fn get_vulkan_entry() -> &'static Vulkan_1_1_EntryDispatch {
    static VULKAN_ENTRY: LazyLock<Vulkan_1_1_EntryDispatch> =
        LazyLock::new(|| Vulkan_1_1_EntryDispatch::load().unwrap());
    &VULKAN_ENTRY
}

fn create_vulkan_instance() -> Instance {
    unsafe {
        let entry = get_vulkan_entry();
        let extension_names = INSTANCE_EXTENSIONS.map(|s| s.as_ptr());
        let application_info = VkApplicationInfo {
            // TODO let the user provide their own name here
            pApplicationName: c"GRAAL".as_ptr(),
            applicationVersion: 0,
            pEngineName: c"GRAAL".as_ptr(),
            engineVersion: 0,
            apiVersion: VK_API_VERSION,
            ..
        };
        let instance_create_info = VkInstanceCreateInfo {
            flags: Default::default(),
            pApplicationInfo: &application_info,
            enabledLayerCount: 0,
            ppEnabledLayerNames: ptr::null(),
            enabledExtensionCount: extension_names.len() as u32,
            ppEnabledExtensionNames: extension_names.as_ptr(),
            ..
        };
        let instance = entry.CreateInstance(&instance_create_info, ptr::null()).unwrap();
        let fns = Vulkan_1_3_InstanceDispatch::load_with(|name| entry.GetInstanceProcAddr(instance, name.as_ptr()));
        let khr_surface =
            khr_surface::InstanceDispatch::load_with(|name| entry.GetInstanceProcAddr(instance, name.as_ptr()));
        Instance { instance, fns, khr_surface }
    }
}
