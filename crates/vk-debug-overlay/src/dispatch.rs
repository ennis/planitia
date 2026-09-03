//mod table;

use ash::vk::{Handle, PFN_vkGetDeviceProcAddr, PFN_vkGetInstanceProcAddr};
use ash::{ext, khr, vk};
use ash_layer::{PFN_vkSetDeviceLoaderData, PFN_vk_layerGetPhysicalDeviceProcAddr,
};
use std::ffi::{CStr, c_void};
use std::mem;
use std::ops::Deref;
use vulkan_headers::vulkan::vulkan as vkh;
use vulkan_headers::vulkan::vulkan::{VkCommandBuffer, VkDevice, VkStructureType};

/*
#[repr(C)]
#[derive(Copy, Clone)]
pub struct VkDeviceMemoryCopyKHR {
    /*
    VkStructureType             sType;
    const void*                 pNext;
    VkDeviceAddressRangeKHR     srcRange;
    VkAddressCommandFlagsKHR    srcFlags;
    VkDeviceAddressRangeKHR     dstRange;
    VkAddressCommandFlagsKHR    dstFlags;
     */
    sType: VkStructureType,
    pNext: *const c_void,
    srcRange: VkDeviceAddressRangeKHR,
    srcFlags: u32, // VkAddressCommandFlagsKHR
    dstRange: VkDeviceAddressRangeKHR,
    dstFlags: u32, // VkAddressCommandFlagsKHR

}

#[repr(C)]
#[derive(Copy,Clone)]
pub struct VkCopyDeviceMemoryInfoKHR {
    /*
    VkStructureType                 sType;
    const void*                     pNext;
    uint32_t                        regionCount;
    const VkDeviceMemoryCopyKHR*    pRegions;
     */
}

pub type NonNullPFN_vkCmdCopyMemoryKHR = unsafe extern "system" fn(commandBuffer: VkCommandBuffer, pCopyMemoryInfo: *const VkCopyDeviceMemoryInfoKHR);
*/

pub struct KhrDeviceAddressCommandsDevice {
    pub cmd_copy_memory: vkh::NonNullPFN_vkCmdCopyMemoryToMicromapEXT,
}

pub(crate) struct ExtDescriptorHeapInstance {
    pub(crate) get_physical_descriptor_size: vkh::NonNullPFN_vkGetPhysicalDeviceDescriptorSizeEXT,
}

impl ExtDescriptorHeapInstance {
    pub(crate) unsafe fn load(instance: vk::Instance, get_instance_proc_addr: PFN_vkGetInstanceProcAddr) -> Self {
        let get_proc_addr = |name: &CStr| {
            let addr = get_instance_proc_addr(instance, name.as_ptr());
            if addr.is_none() {
                panic!("failed to load function pointer for {:?}", name);
            }
            addr
        };

        unsafe {
            Self {
                get_physical_descriptor_size: mem::transmute(get_proc_addr(c"vkGetPhysicalDeviceDescriptorSizeEXT")),
            }
        }
    }
}


pub struct ExtDescriptorHeapDevice {
    pub cmd_bind_resource_heap: vkh::NonNullPFN_vkCmdBindResourceHeapEXT,
    pub cmd_bind_sampler_heap: vkh::NonNullPFN_vkCmdBindSamplerHeapEXT,
    pub cmd_push_data: vkh::NonNullPFN_vkCmdPushDataEXT,
    pub write_resource_descriptors: vkh::NonNullPFN_vkWriteResourceDescriptorsEXT,
    pub write_sampler_descriptors: vkh::NonNullPFN_vkWriteSamplerDescriptorsEXT,
}

impl ExtDescriptorHeapDevice {
    pub(crate) unsafe fn load(device: vk::Device, get_device_proc_addr: PFN_vkGetDeviceProcAddr) -> Self {
        let get_proc_addr = |name: &CStr| {
            let addr = get_device_proc_addr(device, name.as_ptr());
            if addr.is_none() {
                panic!("failed to load function pointer for {:?}", name);
            }
            addr
        };

        unsafe {
            Self {
                cmd_bind_resource_heap: mem::transmute(get_proc_addr(c"vkCmdBindResourceHeapEXT")),
                cmd_bind_sampler_heap: mem::transmute(get_proc_addr(c"vkCmdBindSamplerHeapEXT")),
                cmd_push_data: mem::transmute(get_proc_addr(c"vkCmdPushDataEXT")),
                write_resource_descriptors: mem::transmute(get_proc_addr(c"vkWriteResourceDescriptorsEXT")),
                write_sampler_descriptors: mem::transmute(get_proc_addr(c"vkWriteSamplerDescriptorsEXT")),
            }
        }
    }
}

pub struct InstanceDispatch {
    pub d: ash::Instance,
    pub ext_descriptor_heap: ExtDescriptorHeapInstance,
    pub khr_win32_surface: khr::win32_surface::InstanceFn,
    pub next_get_instance_proc_addr: vk::PFN_vkGetInstanceProcAddr,
    pub next_get_physical_device_proc_addr: PFN_vk_layerGetPhysicalDeviceProcAddr,
}

impl Deref for InstanceDispatch {
    type Target = ash::Instance;

    fn deref(&self) -> &Self::Target {
        &self.d
    }
}

impl InstanceDispatch {
    pub unsafe fn new(
        next_get_instance_proc_addr: vk::PFN_vkGetInstanceProcAddr,
        next_get_physical_device_proc_addr: PFN_vk_layerGetPhysicalDeviceProcAddr,
        instance: vk::Instance,
    ) -> Self {
        let load_fn = |name: &CStr| mem::transmute(next_get_instance_proc_addr(instance, name.as_ptr()));
        let entry = ash::Entry::from_static_fn(ash::StaticFn { get_instance_proc_addr: next_get_instance_proc_addr });
        let ash_instance = ash::Instance::load(entry.static_fn(), instance);
        let ext_descriptor_heap = unsafe { ExtDescriptorHeapInstance::load(instance, next_get_instance_proc_addr) };
        let khr_win32_surface = khr::win32_surface::InstanceFn::load(load_fn);

        InstanceDispatch {
            d: ash_instance,
            ext_descriptor_heap,
            next_get_instance_proc_addr,
            next_get_physical_device_proc_addr,
            khr_win32_surface,
        }
    }
}

/// Device functions dispatch tables.
pub struct DeviceDispatch {
    pub device: ash::Device,
    pub next_get_device_proc_addr: vk::PFN_vkGetDeviceProcAddr,
    pub set_device_loader_data: PFN_vkSetDeviceLoaderData,
    pub khr_swapchain: khr::swapchain::DeviceFn,
    pub ext_debug_utils: ext::debug_utils::DeviceFn,
    pub khr_dynamic_rendering: khr::dynamic_rendering::DeviceFn,
    pub khr_push_descriptors: khr::push_descriptor::DeviceFn,
    pub ext_descriptor_heap: ExtDescriptorHeapDevice,
}

impl DeviceDispatch {
    pub unsafe fn set_device_loader_data(&self, handle: impl vk::Handle) {
        let _ = (self.set_device_loader_data)(self.device.handle(), handle.as_raw() as *mut _);
    }
}

impl Deref for DeviceDispatch {
    type Target = ash::Device;
    fn deref(&self) -> &Self::Target {
        &self.device
    }
}

impl DeviceDispatch {
    pub unsafe fn new(
        device: vk::Device,
        next_get_device_proc_addr: PFN_vkGetDeviceProcAddr,
        set_device_loader_data: PFN_vkSetDeviceLoaderData,
    ) -> Result<DeviceDispatch, vk::Result> {
        // Load device function pointers.
        let load_fn = |func: &CStr| mem::transmute(next_get_device_proc_addr(device, func.as_ptr()));
        let ash_device = ash::Device::load_with(load_fn, device);
        let khr_swapchain = khr::swapchain::DeviceFn::load(load_fn);
        let khr_dynamic_rendering = khr::dynamic_rendering::DeviceFn::load(load_fn);
        let khr_push_descriptors = khr::push_descriptor::DeviceFn::load(load_fn);
        let ext_debug_utils = ext::debug_utils::DeviceFn::load(load_fn);
        let ext_descriptor_heap = ExtDescriptorHeapDevice::load(device, next_get_device_proc_addr);

        Ok(DeviceDispatch {
            device: ash_device,
            next_get_device_proc_addr,
            set_device_loader_data,
            khr_swapchain,
            ext_debug_utils,
            khr_dynamic_rendering,
            khr_push_descriptors,
            ext_descriptor_heap,
        })
    }
}

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
        DispatchKey::from_dispatchable_handle(self)
    }
}
unsafe impl DeviceDispatchableHandle for VkCommandBuffer {
    unsafe fn key(self) -> DispatchKey {
        DispatchKey::from_dispatchable_handle(self)
    }
}
unsafe impl DeviceDispatchableHandle for vk::Device {
    unsafe fn key(self) -> DispatchKey {
        DispatchKey::from_dispatchable_handle(self.as_raw() as *mut c_void)
    }
}
unsafe impl DeviceDispatchableHandle for vk::CommandBuffer {
    unsafe fn key(self) -> DispatchKey {
        DispatchKey::from_dispatchable_handle(self.as_raw() as *mut c_void)
    }
}
unsafe impl DeviceDispatchableHandle for vk::Queue {
    unsafe fn key(self) -> DispatchKey {
        DispatchKey::from_dispatchable_handle(self.as_raw() as *mut c_void)
    }
}
