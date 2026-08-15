#![allow(non_snake_case)]

mod dispatch;
mod font;
mod helper;
mod init;
mod overlay;
mod state_tracker;
mod util;

macro_rules! layer_fn {
    (#[proc($pfn:path)] fn $name:ident ($($args:tt)*) $(-> $rt:ty)? { $($body:tt)* }) => {
        #[no_mangle]
        pub(crate) unsafe extern "system" fn $name(
            $($args)*
        ) $(-> $rt)? {
            $($body)*
        }
        const _: $pfn = $name;
    };
    () => {};
}

use crate::dispatch::{DeviceDispatch, InstanceDispatch};
use crate::helper::{DeviceHelper, Image, Pipeline, PrivateData};
use crate::init::{layer_vkCreateDevice, layer_vkCreateInstance, layer_vkDestroyDevice, layer_vkDestroyInstance};
use crate::overlay::{FrameResources, OverlayResources, StaticResources};
use crate::state_tracker::swapchain::{
    layer_vkCreateSwapchainKHR, layer_vkDestroySwapchainKHR, layer_vkQueuePresentKHR,
};
use ash::vk;
use ash::vk::{PFN_vkGetDeviceProcAddr, PFN_vkGetInstanceProcAddr};
use ash_layer::*;
use core::ffi::{c_char, CStr};
use core::mem;
use dashmap::DashMap;
pub(crate) use layer_fn;
use state_tracker::pipeline::layer_vkCreateGraphicsPipelines;
use state_tracker::queue::layer_vkGetDeviceQueue;
use std::ops::Deref;
use std::slice;
use std::sync::{LazyLock, Mutex};

// ---------------------------------------------------------------------------
// Per-instance and per-device data
// ---------------------------------------------------------------------------

const FRAMES_IN_FLIGHT: usize = 3;

struct TrackedResources {
    pipelines: Vec<PipelineData>,
    swapchains: Vec<LayerSwapchain>,
    present_image_copy: Option<Image>,
}

struct PipelineData {
    pipeline: vk::Pipeline,
}

struct LayerSwapchain {
    device: vk::Device,
    format: vk::Format,
    extent: vk::Extent2D,
    swapchain: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    image_views: Vec<vk::ImageView>,
    render_to_present: Vec<vk::Semaphore>,
}

/// Per-device layer state
///
/// This holds the state tracker and resources for rendering the overlay.
pub struct DeviceState {
    pub helper: DeviceHelper,
    pub frame_resources: Mutex<FrameResources>,
    pub static_resources: StaticResources,
    pub overlay_resources: Mutex<OverlayResources>,
    pub tracked_resources: Mutex<TrackedResources>,
}

impl Deref for DeviceState {
    type Target = DeviceHelper;

    fn deref(&self) -> &Self::Target {
        &self.helper
    }
}

impl DeviceState {
    unsafe fn new(
        instance_dispatch: &InstanceDispatch,
        device: vk::Device,
        create_info: &vk::DeviceCreateInfo,
        physical_device: vk::PhysicalDevice,
        next_get_device_proc_addr: PFN_vkGetDeviceProcAddr,
        set_device_loader_data: PFN_vkSetDeviceLoaderData,
    ) -> DeviceState {
        let dispatch = DeviceDispatch::new(device, next_get_device_proc_addr, set_device_loader_data).unwrap();
        let first_queue_family = {
            let qcis =
                slice::from_raw_parts(create_info.p_queue_create_infos, create_info.queue_create_info_count as usize);
            qcis.first().map_or(0, |q| q.queue_family_index)
        };

        let mem_props = instance_dispatch.d.get_physical_device_memory_properties(physical_device);
        let helper = DeviceHelper::new(dispatch, mem_props, first_queue_family);

        let static_resources = overlay::initialize_static_resources(&helper);
        let frame_resources = overlay::initialize_frame_resources(&helper);
        let tracked_resources =
            TrackedResources { pipelines: Vec::new(), swapchains: Vec::new(), present_image_copy: None };
        let overlay_resources = OverlayResources::default();

        DeviceState {
            helper,
            static_resources,
            frame_resources: Mutex::new(frame_resources),
            tracked_resources: Mutex::new(tracked_resources),
            overlay_resources: Mutex::new(overlay_resources),
        }
    }
}

static DEVICE_STATE: LazyLock<DashMap<vk::Device, DeviceState>> = LazyLock::new(DashMap::new);

/// Returns the [`DeviceState`] object for the given `VkDevice`.
fn state(device: vk::Device) -> dashmap::mapref::one::Ref<'static, vk::Device, DeviceState> {
    DEVICE_STATE.get(&device).expect("unknown device")
}

// ---------------------------------------------------------------------------
// Global state
// ---------------------------------------------------------------------------

static INSTANCE_MAP: LazyLock<DashMap<vk::Instance, InstanceDispatch>> = LazyLock::new(DashMap::new);
static PHY_TO_INSTANCE: LazyLock<DashMap<vk::PhysicalDevice, vk::Instance>> = LazyLock::new(DashMap::new);

/*
unsafe fn next_get_instance_proc_addr(instance: vk::Instance, p_name: *const c_char) -> PFN_vkVoidFunction {
    let inst = INSTANCE_MAP.get(&instance).expect("unknown instance");
    (inst.next_get_instance_proc_addr)(instance, p_name)
}

unsafe fn next_layer_get_physical_device_proc_addr(
    instance: vk::Instance,
    p_name: *const c_char,
) -> PFN_vkVoidFunction {
    let inst = INSTANCE_MAP.get(&instance).expect("unknown instance");
    (inst.next_get_physical_device_proc_addr)(instance, p_name)
}*/

// ---------------------------------------------------------------------------
// Layer entry points
// ---------------------------------------------------------------------------

layer_fn! {
    #[proc(PFN_vkGetInstanceProcAddr)]
    fn layer_vkGetInstanceProcAddr(
        instance: vk::Instance,
        p_name: *const c_char,
    ) -> vk::PFN_vkVoidFunction {
        let name = CStr::from_ptr(p_name);
        let pfn: *const () = match name.to_bytes() {
            b"vkGetInstanceProcAddr" => layer_vkGetInstanceProcAddr as _,
            b"vkCreateInstance" => layer_vkCreateInstance as _,
            b"vkDestroyInstance" => layer_vkDestroyInstance as _,
            b"vkGetDeviceProcAddr" => layer_vkGetDeviceProcAddr as _,
            b"vkCreateDevice" => layer_vkCreateDevice as _,
            b"vkDestroyDevice" => layer_vkDestroyDevice as _,
            b"vk_layerGetPhysicalDeviceProcAddr" => layer_vk_layerGetPhysicalDeviceProcAddr as _,
            _ => {
                let inst = INSTANCE_MAP.get(&instance).expect("unknown instance");
                return (inst.next_get_instance_proc_addr)(instance, p_name);
            }
        };
        mem::transmute(pfn)
    }
}

layer_fn! {
    #[proc(PFN_vkGetDeviceProcAddr)]
    fn layer_vkGetDeviceProcAddr(
        device: vk::Device,
        p_name: *const c_char,
    ) -> vk::PFN_vkVoidFunction {
        let name = CStr::from_ptr(p_name);
        let pfn: *const () = match name.to_bytes() {
            b"vkGetDeviceProcAddr" => layer_vkGetDeviceProcAddr as _,
            b"vkCreateDevice" => layer_vkCreateDevice as _,
            b"vkDestroyDevice" => layer_vkDestroyDevice as _,
            b"vkGetDeviceQueue" => layer_vkGetDeviceQueue as _,
            b"vkCreateSwapchainKHR" => layer_vkCreateSwapchainKHR as _,
            b"vkDestroySwapchainKHR" => layer_vkDestroySwapchainKHR as _,
            b"vkQueuePresentKHR" => layer_vkQueuePresentKHR as _,
            b"vkCreateGraphicsPipelines" => layer_vkCreateGraphicsPipelines as _,
            _ => {
                let d = DEVICE_STATE.get(&device).expect("unknown device");
                return (d.next_get_device_proc_addr)(device, p_name)
            }
        };
        mem::transmute(pfn)
    }
}

layer_fn! {
    #[proc(PFN_vk_layerGetPhysicalDeviceProcAddr)]
    fn layer_vk_layerGetPhysicalDeviceProcAddr(
        instance: vk::Instance,
        p_name: *const c_char,
    ) -> vk::PFN_vkVoidFunction {
        let name = CStr::from_ptr(p_name);
        let pfn: *const () = match name.to_bytes() {
            b"vkCreateDevice" => layer_vkCreateDevice as _,
            _ => {
                let inst = INSTANCE_MAP.get(&instance).expect("unknown instance");
                return (inst.next_get_physical_device_proc_addr)(instance, p_name);
            }
        };
        mem::transmute(pfn)
    }
}

layer_fn! {
    #[proc(PFN_vkNegotiateLoaderLayerInterfaceVersion)]
    fn vkNegotiateLoaderLayerInterfaceVersion(
        p_version_struct: *mut NegotiateLayerInterface,
    ) -> vk::Result {
        let v = &mut *p_version_struct;
        v.loader_layer_interface_version = 2;
        v.pfn_get_instance_proc_addr = Some(layer_vkGetInstanceProcAddr);
        v.pfn_get_device_proc_addr = Some(layer_vkGetDeviceProcAddr);
        v.pfn_get_physical_device_proc_addr = Some(layer_vk_layerGetPhysicalDeviceProcAddr);
        vk::Result::SUCCESS
    }
}
