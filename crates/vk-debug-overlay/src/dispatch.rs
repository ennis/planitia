use std::ffi::{CStr, c_void};
use std::mem;
use std::ops::Deref;
use vulkan::*;

pub struct InstanceDispatch {
    pub instance: VkInstance,
    pub fns: InstanceDispatchCombined,
    pub next_get_instance_proc_addr: PFN_vkGetInstanceProcAddr,
    pub next_get_physical_device_proc_addr: vk_layer::PFN_GetPhysicalDeviceProcAddr,
}

impl Deref for InstanceDispatch {
    type Target = InstanceDispatchCombined;
    fn deref(&self) -> &Self::Target {
        &self.fns
    }
}

impl InstanceDispatch {
    pub unsafe fn new(
        next_get_instance_proc_addr: PFN_vkGetInstanceProcAddr,
        next_get_physical_device_proc_addr: vk_layer::PFN_GetPhysicalDeviceProcAddr,
        instance: VkInstance,
    ) -> Self {
        let load_fn = |name: &CStr| next_get_instance_proc_addr(instance, name.as_ptr());
        let instance_dispatch = InstanceDispatchCombined::load_with(load_fn);
        InstanceDispatch {
            instance,
            fns: instance_dispatch,
            next_get_instance_proc_addr,
            next_get_physical_device_proc_addr,
        }
    }
}

/// Device function dispatch tables.
pub struct DeviceDispatch {
    pub device: VkDevice,
    pub d: DeviceDispatchCombined,
    pub next_get_device_proc_addr: PFN_vkGetDeviceProcAddr,
    pub set_device_loader_data: NonNullPFN_vkSetDeviceLoaderData,
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
        set_device_loader_data: vk_layer::PFN_vkSetDeviceLoaderData,
    ) -> Result<DeviceDispatch, VkResult> {
        // Load device function pointers.
        let load_fn = |func: &CStr| mem::transmute(next_get_device_proc_addr(device, func.as_ptr()));
        let d = DeviceDispatchCombined::load_with(load_fn);
        Ok(DeviceDispatch {
            device,
            d,
            next_get_device_proc_addr,
            set_device_loader_data,
        })
    }
}

type NonNullPFN_vkSetDeviceLoaderData =
    unsafe extern "system" fn(device: VkDevice, object: *mut ::std::os::raw::c_void) -> VkResult;

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
