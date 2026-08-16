#![allow(non_snake_case)]

mod dispatch;
mod font;
mod helper;
mod init;
mod overlay;
mod state_tracker;
mod util;
mod reflection;

use crate::dispatch::{DeviceDispatch, InstanceDispatch};
use crate::helper::{DeviceHelper, Image, Pipeline, PrivateData};
use crate::init::{layer_vkCreateDevice, layer_vkCreateInstance, layer_vkDestroyDevice, layer_vkDestroyInstance};
use crate::overlay::{FrameResources, OverlayResources, StaticResources};
use ash::vk;
use ash::vk::{
    Handle, PFN_vkAllocateCommandBuffers, PFN_vkBeginCommandBuffer, PFN_vkCmdDispatch, PFN_vkCmdDraw,
    PFN_vkCmdDrawIndexed, PFN_vkCmdDrawIndirect, PFN_vkCreateComputePipelines, PFN_vkCreateGraphicsPipelines,
    PFN_vkCreateSwapchainKHR, PFN_vkDestroyPipeline, PFN_vkDestroySwapchainKHR, PFN_vkEndCommandBuffer,
    PFN_vkFreeCommandBuffers, PFN_vkGetDeviceProcAddr, PFN_vkGetDeviceQueue, PFN_vkGetInstanceProcAddr,
    PFN_vkQueuePresentKHR,
};
use ash_layer::*;
use core::ffi::{c_char, CStr};
use core::mem;
use dashmap::DashMap;
use std::ffi::c_void;

use std::ops::Deref;
use std::ptr::NonNull;
use std::slice;
use std::sync::{LazyLock, Mutex};
use vulkan_headers::vulkan::vulkan::{
    NonNullPFN_vkCmdBindResourceHeapEXT, NonNullPFN_vkCmdBindSamplerHeapEXT, NonNullPFN_vkCmdPushDataEXT,
    NonNullPFN_vkWriteResourceDescriptorsEXT, NonNullPFN_vkWriteSamplerDescriptorsEXT, VkBindHeapInfoEXT,
    VkCommandBuffer, VkDevice, VkHostAddressRangeEXT, VkPushDataInfoEXT, VkResourceDescriptorInfoEXT, VkResult,
    VkSamplerCreateInfo,
};

// ---------------------------------------------------------------------------
// Per-instance and per-device data
// ---------------------------------------------------------------------------

const FRAMES_IN_FLIGHT: usize = 3;

struct TrackedResources {
    pipelines: Vec<vk::Pipeline>,
    swapchains: Vec<LayerSwapchain>,
    present_image_copy: Option<Image>,
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

static DEVICE_STATE: LazyLock<DashMap<DispatchKey, DeviceState>> = LazyLock::new(DashMap::new);

/// Returns the [`DeviceState`] object for the given `VkDevice`.
fn device_state<T: DeviceDispatchableHandle>(
    handle: T,
) -> dashmap::mapref::one::Ref<'static, DispatchKey, DeviceState> {
    DEVICE_STATE.get(&unsafe { handle.key() }).expect("unknown device")
}

unsafe fn private_data<'a, T>(handle: T::Handle) -> Option<NonNull<T>>
where
    T: PrivateData,
    T::Handle: DeviceDispatchableHandle,
{
    let device = device_state(handle);
    device.get_private_data(handle)
}

// ---------------------------------------------------------------------------
// Global state
// ---------------------------------------------------------------------------

static INSTANCE_MAP: LazyLock<DashMap<vk::Instance, InstanceDispatch>> = LazyLock::new(DashMap::new);
static PHY_TO_INSTANCE: LazyLock<DashMap<vk::PhysicalDevice, vk::Instance>> = LazyLock::new(DashMap::new);

// ---------------------------------------------------------------------------
// Layer entry points
// ---------------------------------------------------------------------------

#[no_mangle]
pub(crate) unsafe extern "system" fn layer_vkGetInstanceProcAddr(
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
const _: PFN_vkGetInstanceProcAddr = layer_vkGetInstanceProcAddr;

#[no_mangle]
pub(crate) unsafe extern "system" fn layer_vk_layerGetPhysicalDeviceProcAddr(
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
const _: PFN_vk_layerGetPhysicalDeviceProcAddr = layer_vk_layerGetPhysicalDeviceProcAddr;

#[no_mangle]
unsafe extern "system" fn vkNegotiateLoaderLayerInterfaceVersion(
    p_version_struct: *mut NegotiateLayerInterface,
) -> vk::Result {
    let v = &mut *p_version_struct;
    v.loader_layer_interface_version = 2;
    v.pfn_get_instance_proc_addr = Some(layer_vkGetInstanceProcAddr);
    v.pfn_get_device_proc_addr = Some(layer_vkGetDeviceProcAddr);
    v.pfn_get_physical_device_proc_addr = Some(layer_vk_layerGetPhysicalDeviceProcAddr);
    vk::Result::SUCCESS
}
const _: PFN_vkNegotiateLoaderLayerInterfaceVersion = vkNegotiateLoaderLayerInterfaceVersion;

// ---------------------------------------------------------------------------
// Hook table
// ---------------------------------------------------------------------------

macro_rules! device_hooks {
    ($(
        [$ep_name:literal, $pfn:path] fn $name:ident ($dispatch_arg:ident : $dispatch_ty:ty $(, $arg:ident : $argty:ty)*) $(-> $rt:ty)? = $method:ident;
    )*) => {
        $(
        #[no_mangle]
        pub(crate) unsafe extern "system" fn $name(
            $dispatch_arg: $dispatch_ty
            $(, $arg: $argty)*
        ) $(-> $rt)? {
            device_state($dispatch_arg).$method($dispatch_arg $(, $arg)*)
        }
        const _: $pfn = $name;
        )*

        #[no_mangle]
        unsafe extern "system" fn layer_vkGetDeviceProcAddr(
            device: vk::Device,
            p_name: *const c_char,
        ) -> vk::PFN_vkVoidFunction {
            let name = unsafe { CStr::from_ptr(p_name) };
            let pfn: *const () = match name.to_bytes() {
                // Those functions are implemented manually
                b"vkGetDeviceProcAddr" => mem::transmute(layer_vkGetDeviceProcAddr as *const ()),
                b"vkCreateDevice" => mem::transmute(layer_vkCreateDevice as *const ()),
                b"vkDestroyDevice" => mem::transmute(layer_vkDestroyDevice as *const ()),
                // Device hooks
                $(
                    $ep_name => $name as *const (),
                )*
                // Unhooked functions
                _ => {
                    let d = DEVICE_STATE.get(&device.key()).expect("unknown device");
                    return (d.next_get_device_proc_addr)(device, p_name);
                }
            };
            mem::transmute(pfn)
        }
        const _: PFN_vkGetDeviceProcAddr = layer_vkGetDeviceProcAddr;
    };
}

device_hooks! {
    [b"vkGetDeviceQueue", PFN_vkGetDeviceQueue] fn layer_vkGetDeviceQueue(device: vk::Device, queue_family_index: u32, queue_index: u32, p_queue: *mut vk::Queue) = hook_get_device_queue;

    [b"vkCmdPushDataEXT", NonNullPFN_vkCmdPushDataEXT] fn layer_vkCmdPushDataEXT(commandBuffer: VkCommandBuffer, pPushDataInfo: *const VkPushDataInfoEXT) = hook_cmd_push_data_ext;
    [b"vkCmdBindResourceHeapEXT", NonNullPFN_vkCmdBindResourceHeapEXT] fn layer_vkCmdBindResourceHeapEXT(commandBuffer: VkCommandBuffer, pBindInfo: *const VkBindHeapInfoEXT) = hook_cmd_bind_resource_heap_ext;
    [b"vkCmdBindSamplerHeapEXT", NonNullPFN_vkCmdBindSamplerHeapEXT] fn layer_vkCmdBindSamplerHeapEXT(commandBuffer: VkCommandBuffer, pBindInfo: *const VkBindHeapInfoEXT) = hook_cmd_bind_sampler_heap_ext;
    [b"vkWriteResourceDescriptorsEXT", NonNullPFN_vkWriteResourceDescriptorsEXT] fn layer_vkWriteResourceDescriptorsEXT(device: VkDevice, resourceCount: u32, pResources: *const VkResourceDescriptorInfoEXT, pDescriptors: *const VkHostAddressRangeEXT) -> VkResult = hook_write_resource_descriptors_ext;
    [b"vkWriteSamplerDescriptorsEXT", NonNullPFN_vkWriteSamplerDescriptorsEXT] fn layer_vkWriteSamplerDescriptorsEXT(device: VkDevice, samplerCount: u32, pSamplers: *const VkSamplerCreateInfo, pDescriptors: *const VkHostAddressRangeEXT) -> VkResult = hook_write_sampler_descriptors_ext;

    [b"vkAllocateCommandBuffers", PFN_vkAllocateCommandBuffers] fn layer_vkAllocateCommandBuffers(device: vk::Device, pAllocateInfo: *const vk::CommandBufferAllocateInfo, pCommandBuffers: *mut vk::CommandBuffer) -> vk::Result = hook_allocate_command_buffers;
    [b"vkFreeCommandBuffers", PFN_vkFreeCommandBuffers] fn layer_vkFreeCommandBuffers(device: vk::Device, commandPool: vk::CommandPool, commandBufferCount: u32, pCommandBuffers: *const vk::CommandBuffer) = hook_free_command_buffers;
    [b"vkBeginCommandBuffer", PFN_vkBeginCommandBuffer] fn layer_vkBeginCommandBuffer(commandBuffer: vk::CommandBuffer, pBeginInfo: *const vk::CommandBufferBeginInfo) -> vk::Result = hook_begin_command_buffer;
    [b"vkEndCommandBuffer", PFN_vkEndCommandBuffer] fn layer_vkEndCommandBuffer(commandBuffer: vk::CommandBuffer) -> vk::Result = hook_end_command_buffer;

    [b"vkCreateGraphicsPipelines", PFN_vkCreateGraphicsPipelines] fn layer_vkCreateGraphicsPipelines(device: vk::Device, pipelineCache: vk::PipelineCache, createInfoCount: u32, pCreateInfos: *const vk::GraphicsPipelineCreateInfo, pAllocator: *const vk::AllocationCallbacks, pPipelines: *mut vk::Pipeline) -> vk::Result = hook_create_graphics_pipelines;
    [b"vkCreateComputePipelines", PFN_vkCreateComputePipelines] fn layer_vkCreateComputePipelines(device: vk::Device, pipelineCache: vk::PipelineCache, createInfoCount: u32, pCreateInfos: *const vk::ComputePipelineCreateInfo, pAllocator: *const vk::AllocationCallbacks, pPipelines: *mut vk::Pipeline) -> vk::Result = hook_create_compute_pipelines;
    [b"vkDestroyPipeline", PFN_vkDestroyPipeline] fn layer_vkDestroyPipeline(device: vk::Device, pipeline: vk::Pipeline, pAllocator: *const vk::AllocationCallbacks) = hook_destroy_pipeline;

    [b"vkCmdDraw", PFN_vkCmdDraw] fn layer_vkCmdDraw(commandBuffer: vk::CommandBuffer, vertexCount: u32, instanceCount: u32, firstVertex: u32, firstInstance: u32) = hook_cmd_draw;
    [b"vkCmdDrawIndexed", PFN_vkCmdDrawIndexed] fn layer_vkCmdDrawIndexed(commandBuffer: vk::CommandBuffer, indexCount: u32, instanceCount: u32, firstIndex: u32, vertexOffset: i32, firstInstance: u32) = hook_cmd_draw_indexed;
    [b"vkCmdDrawIndirect", PFN_vkCmdDrawIndirect] fn layer_vkCmdDrawIndirect(commandBuffer: vk::CommandBuffer, buffer: vk::Buffer, offset: vk::DeviceSize, drawCount: u32, stride: u32) = hook_cmd_draw_indirect;
    [b"vkCmdDispatch", PFN_vkCmdDispatch] fn layer_vkCmdDispatch(commandBuffer: vk::CommandBuffer, groupCountX: u32, groupCountY: u32, groupCountZ: u32) = hook_cmd_dispatch;

    [b"vkCreateSwapchainKHR", PFN_vkCreateSwapchainKHR] fn layer_vkCreateSwapchainKHR(device: vk::Device, pCreateInfo: *const vk::SwapchainCreateInfoKHR, pAllocator: *const vk::AllocationCallbacks, pSwapchain: *mut vk::SwapchainKHR) -> vk::Result = hook_create_swapchain_khr;
    [b"vkDestroySwapchainKHR", PFN_vkDestroySwapchainKHR] fn layer_vkDestroySwapchainKHR(device: vk::Device, swapchain: vk::SwapchainKHR, pAllocator: *const vk::AllocationCallbacks) = hook_destroy_swapchain_khr;
    [b"vkQueuePresentKHR", PFN_vkQueuePresentKHR] fn layer_vkQueuePresentKHR(queue: vk::Queue, p_present_info: *const vk::PresentInfoKHR) -> vk::Result = hook_queue_present_khr;
}
