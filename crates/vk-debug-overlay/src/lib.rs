#![allow(non_snake_case)]
#![allow(unsafe_op_in_unsafe_fn, reason = "too verbose")]
extern crate core;

mod bump;
mod debugger;
mod dispatch;
mod event;
mod helper;
mod init;
mod overlay;
mod spirv;
mod state_tracker;
mod surface;
mod util;
mod format;

use crate::bump::BumpAllocator;
use crate::debugger::{Debugger, DebuggerResources};
use crate::dispatch::{DeviceDispatch, DeviceDispatchableHandle, DispatchKey, InstanceDispatch};
use crate::event::EventTimeline;
use crate::helper::{DeviceHelper, Pipeline};
use crate::init::{layer_vkCreateDevice, layer_vkCreateInstance, layer_vkDestroyDevice, layer_vkDestroyInstance};
use crate::overlay::gui::GuiState;
use crate::overlay::input::InputState;
use crate::overlay::renderer::OverlayResources;
use crate::spirv::Module;
use crate::state_tracker::command::Command;
use crate::state_tracker::memory::AddressMap;
use crate::surface::layer_vkCreateWin32SurfaceKHR;
use ash::vk;
use ash::vk::{
    PFN_vkAllocateCommandBuffers, PFN_vkBeginCommandBuffer, PFN_vkBindBufferMemory, PFN_vkBindBufferMemory2,
    PFN_vkBindImageMemory, PFN_vkBindImageMemory2, PFN_vkCmdBeginDebugUtilsLabelEXT, PFN_vkCmdBeginRenderPass,
    PFN_vkCmdBeginRenderPass2, PFN_vkCmdBeginRendering, PFN_vkCmdBindPipeline, PFN_vkCmdDispatch, PFN_vkCmdDraw,
    PFN_vkCmdDrawIndexed, PFN_vkCmdDrawIndirect, PFN_vkCmdEndDebugUtilsLabelEXT, PFN_vkCmdEndRenderPass,
    PFN_vkCmdEndRenderPass2, PFN_vkCmdEndRendering, PFN_vkCreateBuffer, PFN_vkCreateComputePipelines,
    PFN_vkCreateGraphicsPipelines, PFN_vkCreateImage, PFN_vkCreateImageView, PFN_vkCreateSwapchainKHR,
    PFN_vkDestroyBuffer, PFN_vkDestroyImage, PFN_vkDestroyImageView, PFN_vkDestroyPipeline, PFN_vkDestroySwapchainKHR,
    PFN_vkEndCommandBuffer, PFN_vkFreeCommandBuffers, PFN_vkGetDeviceProcAddr, PFN_vkGetDeviceQueue,
    PFN_vkGetInstanceProcAddr, PFN_vkQueuePresentKHR, PFN_vkQueueSubmit, PFN_vkSetDebugUtilsObjectNameEXT,
};
use ash_layer::*;
use bumpalo::Bump;
use core::ffi::{CStr, c_char};
use core::mem;
use dashmap::DashMap;
use parking_lot::Mutex;
use slotmap::{SlotMap, new_key_type};
use std::ops::Deref;
use std::slice;
use std::sync::LazyLock;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering::Relaxed;
use vulkan_headers::vulkan::vulkan::{
    NonNullPFN_vkCmdBindResourceHeapEXT, NonNullPFN_vkCmdBindSamplerHeapEXT, NonNullPFN_vkCmdPushDataEXT,
    NonNullPFN_vkWriteResourceDescriptorsEXT, NonNullPFN_vkWriteSamplerDescriptorsEXT, VkBindHeapInfoEXT,
    VkCommandBuffer, VkDevice, VkHostAddressRangeEXT, VkPushDataInfoEXT, VkResourceDescriptorInfoEXT, VkResult,
    VkSamplerCreateInfo,
};

/// Maximum number of frames in flight.
///
/// Used as an upper bound for the number of in-flight resources allocated for the overlay.
const FRAMES_IN_FLIGHT: usize = 3;

/// Device state.
///
/// This holds the state tracker and resources for rendering the overlay.
pub struct Device {
    pub helper: DeviceHelper,                   // device dispatch tables + vulkan helpers
    pub tracked_objects: Mutex<TrackedObjects>, // various tracked objects
    pub addrmap: Mutex<AddressMap>,             // Map from device addresses to buffers
    pub submissions: Mutex<SubmissionState>,
    pub event_timeline: Mutex<EventTimeline>, // Used to generate event IDs (EIDs) that are coherent between frames.
    pub gui: Mutex<GuiState>,                 // GUI state
    pub overlay: OverlayResources,            // Overlay-related state and resources
    pub debugger_resources: DebuggerResources, // Immutable debugger resources
    pub debugger: Mutex<Debugger>,            // Debugger state
    pub modules: Mutex<SlotMap<ModuleId, Module>>, // SPIR-V modules
    pub input: Mutex<InputState>,             // Tracks inputs to the main window
    pub bump: Mutex<BumpAllocator>,           // FIXME Not sure what this is for
    pub frame_index: AtomicU64,               // Frame counter
}

impl Deref for Device {
    type Target = DeviceHelper;

    fn deref(&self) -> &Self::Target {
        &self.helper
    }
}

impl Device {
    unsafe fn new(
        instance_dispatch: &InstanceDispatch,
        device: vk::Device,
        create_info: &vk::DeviceCreateInfo,
        physical_device: vk::PhysicalDevice,
        next_get_device_proc_addr: PFN_vkGetDeviceProcAddr,
        set_device_loader_data: PFN_vkSetDeviceLoaderData,
    ) -> Device {
        let dispatch = DeviceDispatch::new(device, next_get_device_proc_addr, set_device_loader_data).unwrap();
        let first_queue_family = {
            let qcis =
                slice::from_raw_parts(create_info.p_queue_create_infos, create_info.queue_create_info_count as usize);
            qcis.first().map_or(0, |q| q.queue_family_index)
        };

        let mem_props = instance_dispatch.d.get_physical_device_memory_properties(physical_device);
        let helper = DeviceHelper::new(dispatch, mem_props, first_queue_family);
        let tracked_resources = TrackedObjects { pipelines: Vec::new(), swapchains: Vec::new() };
        let overlay_resources = OverlayResources::new(&helper);
        let debugger_resources = DebuggerResources::new(&helper);
        let debugger = Debugger::new();
        let event_timeline = EventTimeline::new();

        Device {
            helper,
            tracked_objects: Mutex::new(tracked_resources),
            addrmap: Mutex::new(AddressMap::new()),
            submissions: Mutex::new(SubmissionState::new()),
            debugger_resources,
            debugger: Mutex::new(debugger),
            modules: Mutex::new(SlotMap::with_key()),
            input: Mutex::new(InputState::new()),
            bump: Mutex::new(BumpAllocator::new()),
            gui: Mutex::new(GuiState::new()),
            event_timeline: Mutex::new(event_timeline),
            overlay: overlay_resources,
            frame_index: Default::default(),
        }
    }

    unsafe fn end_frame(&self) {
        let mut sbs = self.submissions.lock();
        sbs.subs.clear();
        sbs.submission_count = 0;
        let mut bump = self.bump.lock();
        bump.reset();
        let mut dbg = self.debugger.lock();
        dbg.end_frame(self);
        self.frame_index.fetch_add(1, Relaxed);
    }

    fn get_frame_index(&self) -> u64 {
        self.frame_index.load(Relaxed)
    }
}

new_key_type! {
    pub struct ModuleId;
}

pub type ModuleMap = SlotMap<ModuleId, Module>;

/// Information about a swapchain.
struct SwapchainInfo {
    surface: vk::SurfaceKHR,
    device: vk::Device,
    format: vk::Format,
    extent: vk::Extent2D,
    swapchain: vk::SwapchainKHR,
    images: Vec<VkImage>,
    image_views: Vec<vk::ImageView>,
    render_to_present: Vec<vk::Semaphore>,
}

/// Represents a submitted command buffer.
///
/// There's one per VkCommandBuffer, not vkQueueSubmit.
pub struct Submission {
    cmd_buf: vk::CommandBuffer,
    commands: Vec<Command>,
}

pub struct SubmissionState {
    // Submitted command buffers, in order of submission
    subs: Vec<Submission>,
    submission_count: usize,
}

impl SubmissionState {
    fn new() -> SubmissionState {
        SubmissionState { subs: vec![], submission_count: 0 }
    }
}

pub struct TrackedObjects {
    pipelines: Vec<vk::Pipeline>,
    swapchains: Vec<SwapchainInfo>,
}

static DEVICE_STATE: LazyLock<DashMap<DispatchKey, Device>> = LazyLock::new(DashMap::new);

/// Returns the [`Device`] for the given `VkDevice`.
fn device_state<T: DeviceDispatchableHandle>(handle: T) -> dashmap::mapref::one::Ref<'static, DispatchKey, Device> {
    DEVICE_STATE.get(&unsafe { handle.key() }).expect("unknown device")
}

// ---------------------------------------------------------------------------
// Global state
// ---------------------------------------------------------------------------

static INSTANCE_MAP: LazyLock<DashMap<vk::Instance, InstanceDispatch>> = LazyLock::new(DashMap::new);
static PHY_TO_INSTANCE: LazyLock<DashMap<vk::PhysicalDevice, vk::Instance>> = LazyLock::new(DashMap::new);

thread_local! {
    // Static bump allocator for long-lived things, like reflection data for pipelines (we never free those).
    // The arena themselves are thread-local, but the 'static references they produce are shareable
    // across threads.
    static THREAD_LOCAL_BUMP_ALLOC: &'static Bump = Box::leak(Box::new(Bump::new()));
}

pub fn thread_local_bump_alloc() -> &'static Bump {
    THREAD_LOCAL_BUMP_ALLOC.with(|a| *a)
}

// ---------------------------------------------------------------------------
// Layer entry points
// ---------------------------------------------------------------------------

#[unsafe(no_mangle)]
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
        b"vkCreateWin32SurfaceKHR" => layer_vkCreateWin32SurfaceKHR as _,
        b"vk_layerGetPhysicalDeviceProcAddr" => layer_vk_layerGetPhysicalDeviceProcAddr as _,
        _ => {
            // It's possible to call getInstanceProcAddr with a device function,
            // so query our hook table for that.
            return match get_device_proc_addr_hook(p_name) {
                Some(ptr) => ptr,
                None => {
                    // Otherwise, fallback to next getInstanceProcAddr in chain
                    let inst = INSTANCE_MAP.get(&instance).expect("unknown instance");
                    (inst.next_get_instance_proc_addr)(instance, p_name)
                }
            };
        }
    };
    mem::transmute(pfn)
}
const _: PFN_vkGetInstanceProcAddr = layer_vkGetInstanceProcAddr;

#[unsafe(no_mangle)]
unsafe extern "system" fn layer_vkGetDeviceProcAddr(
    device: vk::Device,
    p_name: *const c_char,
) -> vk::PFN_vkVoidFunction {
    match get_device_proc_addr_hook(p_name) {
        Some(ptr) => ptr,
        None => {
            // Unhooked functions
            let d = DEVICE_STATE.get(&device.key()).expect("unknown device");
            (d.next_get_device_proc_addr)(device, p_name)
        }
    }
}
const _: PFN_vkGetDeviceProcAddr = layer_vkGetDeviceProcAddr;

#[unsafe(no_mangle)]
pub(crate) unsafe extern "system" fn layer_vk_layerGetPhysicalDeviceProcAddr(
    instance: vk::Instance,
    p_name: *const c_char,
) -> vk::PFN_vkVoidFunction {
    let name = CStr::from_ptr(p_name);
    let pfn = match name.to_bytes() {
        b"vkCreateDevice" => layer_vkCreateDevice as *const (),
        _ => {
            let inst = INSTANCE_MAP.get(&instance).expect("unknown instance");
            return (inst.next_get_physical_device_proc_addr)(instance, p_name);
        }
    };
    mem::transmute(pfn)
}
const _: PFN_vk_layerGetPhysicalDeviceProcAddr = layer_vk_layerGetPhysicalDeviceProcAddr;

#[unsafe(no_mangle)]
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
// Device hook table
// ---------------------------------------------------------------------------

macro_rules! device_hooks {
    ($(
        [$($ep_name:literal),*; $pfn:path] fn $name:ident ($dispatch_arg:ident : $dispatch_ty:ty $(, $arg:ident : $argty:ty)*) $(-> $rt:ty)? = $method:ident;
    )*) => {
        $(
        #[unsafe(no_mangle)]
        pub(crate) unsafe extern "system" fn $name(
            $dispatch_arg: $dispatch_ty
            $(, $arg: $argty)*
        ) $(-> $rt)? {
            device_state($dispatch_arg).$method($dispatch_arg $(, $arg)*)
        }
        const _: $pfn = $name;
        )*

        unsafe fn get_device_proc_addr_hook(
            p_name: *const c_char,
        ) -> Option<vk::PFN_vkVoidFunction> {
            let name = unsafe { CStr::from_ptr(p_name) };
            let pfn = match name.to_bytes() {
                // Those functions are implemented manually
                b"vkGetDeviceProcAddr" => layer_vkGetDeviceProcAddr as *const (),
                b"vkCreateDevice" => layer_vkCreateDevice as *const (),
                b"vkDestroyDevice" => layer_vkDestroyDevice as *const (),
                // Device hooks
                $(
                    $(| $ep_name)* => $name as *const (),
                )*
                _ => return None,
            };
            Some(mem::transmute(pfn))
        }
    };
}

device_hooks! {
    [b"vkGetDeviceQueue"; PFN_vkGetDeviceQueue] fn layer_vkGetDeviceQueue(device: vk::Device, queue_family_index: u32, queue_index: u32, p_queue: *mut vk::Queue) = hook_get_device_queue;

    [b"vkCmdPushDataEXT"; NonNullPFN_vkCmdPushDataEXT] fn layer_vkCmdPushDataEXT(commandBuffer: VkCommandBuffer, pPushDataInfo: *const VkPushDataInfoEXT) = hook_cmd_push_data_ext;
    [b"vkCmdBindResourceHeapEXT"; NonNullPFN_vkCmdBindResourceHeapEXT] fn layer_vkCmdBindResourceHeapEXT(commandBuffer: VkCommandBuffer, pBindInfo: *const VkBindHeapInfoEXT) = hook_cmd_bind_resource_heap_ext;
    [b"vkCmdBindSamplerHeapEXT"; NonNullPFN_vkCmdBindSamplerHeapEXT] fn layer_vkCmdBindSamplerHeapEXT(commandBuffer: VkCommandBuffer, pBindInfo: *const VkBindHeapInfoEXT) = hook_cmd_bind_sampler_heap_ext;
    [b"vkWriteResourceDescriptorsEXT"; NonNullPFN_vkWriteResourceDescriptorsEXT] fn layer_vkWriteResourceDescriptorsEXT(device: VkDevice, resourceCount: u32, pResources: *const VkResourceDescriptorInfoEXT, pDescriptors: *const VkHostAddressRangeEXT) -> VkResult = hook_write_resource_descriptors_ext;
    [b"vkWriteSamplerDescriptorsEXT"; NonNullPFN_vkWriteSamplerDescriptorsEXT] fn layer_vkWriteSamplerDescriptorsEXT(device: VkDevice, samplerCount: u32, pSamplers: *const VkSamplerCreateInfo, pDescriptors: *const VkHostAddressRangeEXT) -> VkResult = hook_write_sampler_descriptors_ext;

    [b"vkAllocateCommandBuffers"; PFN_vkAllocateCommandBuffers] fn layer_vkAllocateCommandBuffers(device: vk::Device, pAllocateInfo: *const vk::CommandBufferAllocateInfo, pCommandBuffers: *mut vk::CommandBuffer) -> vk::Result = hook_allocate_command_buffers;
    [b"vkFreeCommandBuffers"; PFN_vkFreeCommandBuffers] fn layer_vkFreeCommandBuffers(device: vk::Device, commandPool: vk::CommandPool, commandBufferCount: u32, pCommandBuffers: *const vk::CommandBuffer) = hook_free_command_buffers;
    [b"vkBeginCommandBuffer"; PFN_vkBeginCommandBuffer] fn layer_vkBeginCommandBuffer(commandBuffer: vk::CommandBuffer, pBeginInfo: *const vk::CommandBufferBeginInfo) -> vk::Result = hook_begin_command_buffer;
    [b"vkEndCommandBuffer"; PFN_vkEndCommandBuffer] fn layer_vkEndCommandBuffer(commandBuffer: vk::CommandBuffer) -> vk::Result = hook_end_command_buffer;

    [b"vkCreateGraphicsPipelines"; PFN_vkCreateGraphicsPipelines] fn layer_vkCreateGraphicsPipelines(device: vk::Device, pipelineCache: vk::PipelineCache, createInfoCount: u32, pCreateInfos: *const vk::GraphicsPipelineCreateInfo, pAllocator: *const vk::AllocationCallbacks, pPipelines: *mut vk::Pipeline) -> vk::Result = hook_create_graphics_pipelines;
    [b"vkCreateComputePipelines"; PFN_vkCreateComputePipelines] fn layer_vkCreateComputePipelines(device: vk::Device, pipelineCache: vk::PipelineCache, createInfoCount: u32, pCreateInfos: *const vk::ComputePipelineCreateInfo, pAllocator: *const vk::AllocationCallbacks, pPipelines: *mut vk::Pipeline) -> vk::Result = hook_create_compute_pipelines;
    [b"vkDestroyPipeline"; PFN_vkDestroyPipeline] fn layer_vkDestroyPipeline(device: vk::Device, pipeline: vk::Pipeline, pAllocator: *const vk::AllocationCallbacks) = hook_destroy_pipeline;

    [b"vkCmdBindPipeline"; PFN_vkCmdBindPipeline] fn layer_vkCmdBindPipeline(command_buffer: vk::CommandBuffer, pipeline_bind_point: vk::PipelineBindPoint, pipeline: vk::Pipeline) = hook_cmd_bind_pipeline;
    [b"vkCmdDraw"; PFN_vkCmdDraw] fn layer_vkCmdDraw(commandBuffer: vk::CommandBuffer, vertexCount: u32, instanceCount: u32, firstVertex: u32, firstInstance: u32) = hook_cmd_draw;
    [b"vkCmdDrawIndexed"; PFN_vkCmdDrawIndexed] fn layer_vkCmdDrawIndexed(commandBuffer: vk::CommandBuffer, indexCount: u32, instanceCount: u32, firstIndex: u32, vertexOffset: i32, firstInstance: u32) = hook_cmd_draw_indexed;
    [b"vkCmdDrawIndirect"; PFN_vkCmdDrawIndirect] fn layer_vkCmdDrawIndirect(commandBuffer: vk::CommandBuffer, buffer: vk::Buffer, offset: vk::DeviceSize, drawCount: u32, stride: u32) = hook_cmd_draw_indirect;
    [b"vkCmdDispatch"; PFN_vkCmdDispatch] fn layer_vkCmdDispatch(commandBuffer: vk::CommandBuffer, groupCountX: u32, groupCountY: u32, groupCountZ: u32) = hook_cmd_dispatch;
    [b"vkCmdBeginDebugUtilsLabelEXT"; PFN_vkCmdBeginDebugUtilsLabelEXT] fn layer_vkCmdBeginDebugUtilsLabelEXT(commandBuffer: vk::CommandBuffer, pLabelInfo: *const vk::DebugUtilsLabelEXT<'_>) = hook_cmd_begin_debug_utils_label;
    [b"vkCmdBeginRenderPass"; PFN_vkCmdBeginRenderPass] fn layer_vkCmdBeginRenderPass(commandBuffer: vk::CommandBuffer, pRenderPassBegin: *const vk::RenderPassBeginInfo<'_>, contents: vk::SubpassContents) = hook_cmd_begin_render_pass;
    [b"vkCmdEndRenderPass"; PFN_vkCmdEndRenderPass] fn layer_vkCmdEndRenderPass(commandBuffer: vk::CommandBuffer) = hook_cmd_end_render_pass;
    [b"vkCmdBeginRenderPass2"; PFN_vkCmdBeginRenderPass2] fn layer_vkCmdBeginRenderPass2(command_buffer: vk::CommandBuffer, p_render_pass_begin: *const vk::RenderPassBeginInfo<'_>, p_subpass_begin_info: *const vk::SubpassBeginInfo<'_>) = hook_cmd_begin_render_pass2;
    [b"vkCmdEndRenderPass2"; PFN_vkCmdEndRenderPass2] fn layer_vkCmdEndRenderPass2(command_buffer: vk::CommandBuffer, p_subpass_end_info: *const vk::SubpassEndInfo<'_>) = hook_cmd_end_render_pass2;
    [b"vkCmdBeginRendering", b"vkCmdBeginRenderingKHR"; PFN_vkCmdBeginRendering] fn layer_vkCmdBeginRendering(commandBuffer: vk::CommandBuffer, pRenderingInfo: *const vk::RenderingInfo<'_>) = hook_cmd_begin_rendering;
    [b"vkCmdEndRendering", b"vkCmdEndRenderingKHR"; PFN_vkCmdEndRendering] fn layer_vkCmdEndRendering(commandBuffer: vk::CommandBuffer) = hook_cmd_end_rendering;

    [b"vkCreateSwapchainKHR"; PFN_vkCreateSwapchainKHR] fn layer_vkCreateSwapchainKHR(device: vk::Device, pCreateInfo: *const vk::SwapchainCreateInfoKHR, pAllocator: *const vk::AllocationCallbacks, pSwapchain: *mut vk::SwapchainKHR) -> vk::Result = hook_create_swapchain_khr;
    [b"vkDestroySwapchainKHR"; PFN_vkDestroySwapchainKHR] fn layer_vkDestroySwapchainKHR(device: vk::Device, swapchain: vk::SwapchainKHR, pAllocator: *const vk::AllocationCallbacks) = hook_destroy_swapchain_khr;

    [b"vkQueueSubmit"; PFN_vkQueueSubmit] fn layer_vkQueueSubmit(queue: vk::Queue, submit_count: u32, p_submits: *const vk::SubmitInfo<'_>, fence: vk::Fence) -> vk::Result = hook_queue_submit;
    [b"vkQueuePresentKHR"; PFN_vkQueuePresentKHR] fn layer_vkQueuePresentKHR(queue: vk::Queue, p_present_info: *const vk::PresentInfoKHR) -> vk::Result = hook_queue_present_khr;

    [b"vkCreateBuffer"; PFN_vkCreateBuffer] fn layer_vkCreateBuffer(device: vk::Device, pCreateInfo: *const vk::BufferCreateInfo, pAllocator: *const vk::AllocationCallbacks, pBuffer: *mut vk::Buffer) -> vk::Result = hook_create_buffer;
    [b"vkDestroyBuffer"; PFN_vkDestroyBuffer] fn layer_vkDestroyBuffer(device: vk::Device, buffer: vk::Buffer, pAllocator: *const vk::AllocationCallbacks) = hook_destroy_buffer;
    [b"vkBindBufferMemory"; PFN_vkBindBufferMemory] fn layer_vkBindBufferMemory(device: vk::Device, buffer: vk::Buffer, memory: vk::DeviceMemory, memoryOffset: vk::DeviceSize) -> vk::Result = hook_bind_buffer_memory;
    [b"vkBindBufferMemory2"; PFN_vkBindBufferMemory2] fn layer_vkBindBufferMemory2(device: vk::Device, bind_info_count: u32, p_bind_infos: *const vk::BindBufferMemoryInfo<'_>) -> vk::Result = hook_bind_buffer_memory_2;

    [b"vkCmdEndDebugUtilsLabelEXT"; PFN_vkCmdEndDebugUtilsLabelEXT] fn layer_vkCmdEndDebugUtilsLabelEXT(commandBuffer: vk::CommandBuffer) = hook_cmd_end_debug_utils_label;
    [b"vkSetDebugUtilsObjectNameEXT"; PFN_vkSetDebugUtilsObjectNameEXT] fn layer_vkSetDebugUtilsObjectNameEXT(device: vk::Device, p_name_info: *const vk::DebugUtilsObjectNameInfoEXT<'_>) -> vk::Result = hook_set_debug_utils_object_name;

    [b"vkCreateImageView"; PFN_vkCreateImageView] fn layer_vkCreateImageView(device: vk::Device, pCreateInfo: *const vk::ImageViewCreateInfo<'_>, pAllocator: *const vk::AllocationCallbacks<'_>, pView: *mut vk::ImageView) -> vk::Result = hook_create_image_view;
    [b"vkDestroyImageView"; PFN_vkDestroyImageView] fn layer_vkDestroyImageView(device: vk::Device, imageView: vk::ImageView, pAllocator: *const vk::AllocationCallbacks<'_>) = hook_destroy_image_view;

    [b"vkCreateImage"; PFN_vkCreateImage] fn layer_vkCreateImage(device: vk::Device, pCreateInfo: *const vk::ImageCreateInfo<'_>, pAllocator: *const vk::AllocationCallbacks<'_>, pImage: *mut vk::Image) -> vk::Result = hook_create_image;
    [b"vkDestroyImage"; PFN_vkDestroyImage] fn layer_vkDestroyImage(device: vk::Device, image: vk::Image, pAllocator: *const vk::AllocationCallbacks<'_>) = hook_destroy_image;
    [b"vkBindImageMemory"; PFN_vkBindImageMemory] fn layer_vkBindImageMemory(device: vk::Device, image: vk::Image, memory: vk::DeviceMemory, memoryOffset: vk::DeviceSize) -> vk::Result = hook_bind_image_memory;
    [b"vkBindImageMemory2"; PFN_vkBindImageMemory2] fn layer_vkBindImageMemory2(device: vk::Device, bind_info_count: u32, p_bind_infos: *const vk::BindImageMemoryInfo<'_>) -> vk::Result = hook_bind_image_memory_2;

}
