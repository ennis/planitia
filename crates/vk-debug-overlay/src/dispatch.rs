use std::ffi::{CStr, c_void};
use std::mem;
use std::ops::Deref;
use vulkan::*;

pub struct InstanceDispatch {
    pub instance: VkInstance,
    pub d: InstanceDispatchCombined,
    pub next_get_instance_proc_addr: PFN_vkGetInstanceProcAddr,
    pub next_get_physical_device_proc_addr: layer::PFN_GetPhysicalDeviceProcAddr,
    //pub ext_descriptor_heap: ext_descriptor_heap::InstanceDispatch,
    //pub khr_win32_surface: khr_win32_surface::InstanceDispatch,
}

impl Deref for InstanceDispatch {
    type Target = InstanceDispatchCombined;

    fn deref(&self) -> &Self::Target {
        &self.d
    }
}

impl InstanceDispatch {
    pub unsafe fn new(
        next_get_instance_proc_addr: PFN_vkGetInstanceProcAddr,
        next_get_physical_device_proc_addr: layer::PFN_GetPhysicalDeviceProcAddr,
        instance: VkInstance,
    ) -> Self {
        let load_fn = |name: &CStr| next_get_instance_proc_addr(instance, name.as_ptr());
        //let entry = Vulkan_1_1_EntryDispatch::load_with(load_fn);
        let instance_dispatch = InstanceDispatchCombined::load_with(load_fn);
        //let ext_descriptor_heap = ext_descriptor_heap::InstanceDispatch::load_with(load_fn);
        //let khr_win32_surface = khr_win32_surface::InstanceDispatch::load_with(load_fn);

        InstanceDispatch {
            instance,
            d: instance_dispatch,
            next_get_instance_proc_addr,
            next_get_physical_device_proc_addr,
            //ext_descriptor_heap,
            //khr_win32_surface,
        }
    }
}

/// Device functions dispatch tables.
pub struct DeviceDispatch {
    pub device: VkDevice,
    pub d: DeviceDispatchCombined,
    pub next_get_device_proc_addr: PFN_vkGetDeviceProcAddr,
    pub set_device_loader_data: NonNullPFN_vkSetDeviceLoaderData,
    //pub khr_swapchain: khr_swapchain::DeviceDispatch,
    //pub ext_debug_utils: ext_debug_utils::DeviceDispatch,
    //pub khr_dynamic_rendering: khr_dynamic_rendering::DeviceDispatch,
    //pub khr_push_descriptors: khr_push_descriptor::DeviceDispatch,
    //pub ext_descriptor_heap: ext_descriptor_heap::DeviceDispatch,
}

impl DeviceDispatch {
    pub unsafe fn set_device_loader_data(&self, handle: impl VulkanHandle) {
        let _ = (self.set_device_loader_data)(self.device, handle.as_raw() as *mut _);
    }
}

impl Deref for DeviceDispatch {
    type Target = DeviceDispatchCombined;
    fn deref(&self) -> &Self::Target {
        &self.d
    }
}

impl DeviceDispatch {
    pub unsafe fn new(
        device: VkDevice,
        next_get_device_proc_addr: PFN_vkGetDeviceProcAddr,
        set_device_loader_data: layer::PFN_vkSetDeviceLoaderData,
    ) -> Result<DeviceDispatch, VkResult> {
        // Load device function pointers.
        let load_fn = |func: &CStr| mem::transmute(next_get_device_proc_addr(device, func.as_ptr()));
        let d = DeviceDispatchCombined::load_with(load_fn);
        //let khr_swapchain = khr_swapchain::DeviceDispatch::load_with(load_fn);
        //let khr_dynamic_rendering = khr_dynamic_rendering::DeviceDispatch::load_with(load_fn);
        //let khr_push_descriptors = khr_push_descriptor::DeviceDispatch::load_with(load_fn);
        //let ext_debug_utils = ext_debug_utils::DeviceDispatch::load_with(load_fn);
        //let ext_descriptor_heap = ext_descriptor_heap::DeviceDispatch::load_with(load_fn);

        Ok(DeviceDispatch {
            device,
            d,
            next_get_device_proc_addr,
            set_device_loader_data: set_device_loader_data.unwrap(),
            //khr_swapchain,
            //ext_debug_utils,
            //khr_dynamic_rendering,
            //khr_push_descriptors,
            //ext_descriptor_heap,
        })
    }
}

type NonNullPFN_vkSetDeviceLoaderData =
    unsafe extern "C" fn(device: VkDevice, object: *mut ::std::os::raw::c_void) -> VkResult;

//--------------------------------------------------------------------------------------------------

#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct DispatchKey(*const c_void);
unsafe impl Send for DispatchKey {}
unsafe impl Sync for DispatchKey {}
impl DispatchKey {
    unsafe fn from_dispatchable_handle<T>(handle: *mut T) -> DispatchKey {
        let ptr = handle as *const *const c_void;
        DispatchKey(*ptr)
    }
}

pub unsafe trait DeviceDispatchableHandle: Sized {
    unsafe fn key(self) -> DispatchKey;
}

unsafe impl DeviceDispatchableHandle for VkDevice {
    unsafe fn key(self) -> DispatchKey {
        DispatchKey::from_dispatchable_handle(self.as_raw_ptr())
    }
}
unsafe impl DeviceDispatchableHandle for VkCommandBuffer {
    unsafe fn key(self) -> DispatchKey {
        DispatchKey::from_dispatchable_handle(self.as_raw_ptr())
    }
}
unsafe impl DeviceDispatchableHandle for VkQueue {
    unsafe fn key(self) -> DispatchKey {
        DispatchKey::from_dispatchable_handle(self.as_raw_ptr())
    }
}
