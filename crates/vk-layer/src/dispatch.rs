use std::ffi::CStr;
use std::mem;
use std::ops::Deref;
use ash::{khr, vk};
use ash::vk::PFN_vkGetDeviceProcAddr;
use ash_layer::{get_device_chain_info, LayerFunction, PFN_vkSetDeviceLoaderData, PFN_vk_layerGetPhysicalDeviceProcAddr};

pub struct InstanceDispatch {
    pub d: ash::Instance,
    pub next_get_instance_proc_addr: vk::PFN_vkGetInstanceProcAddr,
    pub next_get_physical_device_proc_addr: PFN_vk_layerGetPhysicalDeviceProcAddr,
}

impl Deref for InstanceDispatch {
    type Target = ash::Instance;

    fn deref(&self) -> &Self::Target {
        &self.d
    }
}

/// Device functions dispatch tables.
pub struct DeviceDispatch {
    pub device: ash::Device,
    pub next_get_device_proc_addr: vk::PFN_vkGetDeviceProcAddr,
    pub set_device_loader_data: PFN_vkSetDeviceLoaderData,
    pub khr_swapchain: khr::swapchain::DeviceFn,
    pub khr_dynamic_rendering: khr::dynamic_rendering::DeviceFn,
    pub khr_push_descriptors: khr::push_descriptor::DeviceFn,
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
    pub unsafe fn new(device: vk::Device,
                      next_get_device_proc_addr: PFN_vkGetDeviceProcAddr,
                      set_device_loader_data: PFN_vkSetDeviceLoaderData
    ) -> Result<DeviceDispatch, vk::Result> {


        // Load device function pointers.
        let ash_device = ash::Device::load_with(
            |func| {
                let fnaddr = next_get_device_proc_addr(device, func.as_ptr());
                mem::transmute(fnaddr)
            },
            device,
        );

        let load_fn = |name: &CStr| mem::transmute(next_get_device_proc_addr(device, name.as_ptr()));
        let khr_swapchain = khr::swapchain::DeviceFn::load(load_fn);
        let khr_dynamic_rendering = khr::dynamic_rendering::DeviceFn::load(load_fn);
        let khr_push_descriptors = khr::push_descriptor::DeviceFn::load(load_fn);

        Ok(DeviceDispatch {
            device: ash_device,
            next_get_device_proc_addr,
            set_device_loader_data,
            khr_swapchain,
            khr_dynamic_rendering,
            khr_push_descriptors,
        })
    }
}