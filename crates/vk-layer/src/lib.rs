// Vulkan requires non-snake-case names for exported symbols and internal
// dispatch helpers that match the Vulkan naming convention.
#![allow(non_snake_case)]

mod font;
mod helper;
mod overlay;
mod swapchain;

use ash::vk::PFN_vkVoidFunction;
use ash::{khr, vk};
use ash_layer::*;
use core::ffi::{c_char, CStr};
use core::mem;
use dashmap::DashMap;
use std::ffi::c_void;
use std::ops::Deref;
use std::slice;
use std::sync::{LazyLock, Mutex};

// ---------------------------------------------------------------------------
// Per-instance and per-device data
// ---------------------------------------------------------------------------

const FRAMES_IN_FLIGHT: usize = 3;

#[derive(Copy, Clone, Default)]
struct FrameData {
    cmdbuf: vk::CommandBuffer,
    fence: vk::Fence,
    vtxbuf: Buffer,
}

#[derive(Copy, Clone, Default)]
struct Image {
    image: vk::Image,
    image_view: vk::ImageView,
    memory: vk::DeviceMemory,
}

#[derive(Copy, Clone, Default)]
struct Buffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    ptr: *mut c_void,
}

#[derive(Copy, Clone, Default)]
struct Pipeline {
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    descriptor_set_layout: vk::DescriptorSetLayout,
}

unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

//#[derive(Default)]
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

struct LayerInstance {
    d: ash::Instance,
    next_get_instance_proc_addr: vk::PFN_vkGetInstanceProcAddr,
    next_get_physical_device_proc_addr: PFN_vk_layerGetPhysicalDeviceProcAddr,
}

struct LayerQueue {
    device: vk::Device,
    queue_family_index: u32,
}

/// Device functions dispatch tables.
struct DeviceDispatch {
    device: ash::Device,
    next_get_device_proc_addr: vk::PFN_vkGetDeviceProcAddr,
    set_device_loader_data: PFN_vkSetDeviceLoaderData,
    khr_swapchain: khr::swapchain::DeviceFn,
    khr_dynamic_rendering: khr::dynamic_rendering::DeviceFn,
    khr_push_descriptors: khr::push_descriptor::DeviceFn,
}

impl Deref for DeviceDispatch {
    type Target = ash::Device;

    fn deref(&self) -> &Self::Target {
        &self.device
    }
}

/// Device & command pool wrapper with useful utilities.
struct DeviceHelper {
    dispatch: DeviceDispatch,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    mem_props: vk::PhysicalDeviceMemoryProperties,
}

impl Deref for DeviceHelper {
    type Target = DeviceDispatch;

    fn deref(&self) -> &Self::Target {
        &self.dispatch
    }
}

struct StaticResources {
    font_tex: Image,
    font_sampler: vk::Sampler,
    pipeline: Pipeline,
}

#[derive(Default)]
struct OverlayResources {
    tmp_image: Option<Image>,
    last_width: u32,
    last_height: u32,
}

struct FrameResources {
    frame_index: usize,
    frame_data: [FrameData; FRAMES_IN_FLIGHT],
}

/// Per-device layer data.
struct DeviceData {
    helper: DeviceHelper,
    static_resources: StaticResources,
    frame_resources: Mutex<FrameResources>,
    overlay_resources: Mutex<OverlayResources>,
    tracked_resources: Mutex<TrackedResources>,
    first_queue_family: u32,
}

impl Deref for DeviceData {
    type Target = DeviceHelper;

    fn deref(&self) -> &Self::Target {
        &self.helper
    }
}

unsafe fn find_next<N>(prev: &impl vk::TaggedStructure) -> Option<*const N>
where
    N: vk::TaggedStructure,
{
    let base_in_struct = prev as *const _ as *const vk::BaseInStructure;
    let mut p_next = (*base_in_struct).p_next;
    while let Some(base) = p_next.as_ref() {
        if base.s_type == N::STRUCTURE_TYPE {
            return Some(p_next.cast::<N>());
        }
        p_next = base.p_next;
    }
    None
}

// ---------------------------------------------------------------------------
// Global state
// ---------------------------------------------------------------------------

static INSTANCE_MAP: LazyLock<DashMap<vk::Instance, LayerInstance>> = LazyLock::new(DashMap::new);
static PHY_TO_INSTANCE: LazyLock<DashMap<vk::PhysicalDevice, vk::Instance>> = LazyLock::new(DashMap::new);
static DEVICE_DATA: LazyLock<DashMap<vk::Device, DeviceData>> = LazyLock::new(DashMap::new);

//static SWAPCHAIN_MAP: LazyLock<DashMap<vk::SwapchainKHR, LayerSwapchain>> = LazyLock::new(DashMap::new);
/*
/// Per-device next-layer GDPA, stored so device functions can be forwarded.
static GDPA_MAP: LazyLock<DashMap<vk::Device, vk::PFN_vkGetDeviceProcAddr>> =
    LazyLock::new(DashMap::new);*/

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
}

unsafe fn next_get_device_proc_addr(device: vk::Device, p_name: *const c_char) -> PFN_vkVoidFunction {
    let d = DEVICE_DATA.get(&device).expect("unknown device");
    (d.next_get_device_proc_addr)(device, p_name)
}

fn device_data(device: vk::Device) -> dashmap::mapref::one::Ref<'static, vk::Device, DeviceData> {
    DEVICE_DATA.get(&device).expect("unknown device")
}

// ---------------------------------------------------------------------------
// Negotiate loader interface (layer interface version 2)
// ---------------------------------------------------------------------------

#[no_mangle]
pub unsafe extern "system" fn vkNegotiateLoaderLayerInterfaceVersion(
    p_version_struct: *mut NegotiateLayerInterface,
) -> vk::Result {
    let v = &mut *p_version_struct;
    v.loader_layer_interface_version = 2;
    v.pfn_get_instance_proc_addr = Some(layer_vkGetInstanceProcAddr);
    v.pfn_get_device_proc_addr = Some(layer_vkGetDeviceProcAddr);
    v.pfn_get_physical_device_proc_addr = Some(layer_vk_layerGetPhysicalDeviceProcAddr);
    vk::Result::SUCCESS
}

// ---------------------------------------------------------------------------
// vkGetInstanceProcAddr
// ---------------------------------------------------------------------------

#[no_mangle]
unsafe extern "system" fn layer_vkGetInstanceProcAddr(
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
            return next_get_instance_proc_addr(instance, p_name);
        }
    };
    mem::transmute(pfn)
}

// ---------------------------------------------------------------------------
// vkGetDeviceProcAddr
// ---------------------------------------------------------------------------

#[no_mangle]
unsafe extern "system" fn layer_vkGetDeviceProcAddr(
    device: vk::Device,
    p_name: *const c_char,
) -> vk::PFN_vkVoidFunction {
    let name = CStr::from_ptr(p_name);
    let pfn: *const () = match name.to_bytes() {
        b"vkGetDeviceProcAddr" => layer_vkGetDeviceProcAddr as _,
        b"vkCreateDevice" => layer_vkCreateDevice as _,
        b"vkDestroyDevice" => layer_vkDestroyDevice as _,
        b"vkGetDeviceQueue" => layer_vkGetDeviceQueue as _,
        b"vkCreateSwapchainKHR" => swapchain::layer_vkCreateSwapchainKHR as _,
        b"vkDestroySwapchainKHR" => swapchain::layer_vkDestroySwapchainKHR as _,
        b"vkQueuePresentKHR" => swapchain::layer_vkQueuePresentKHR as _,
        b"vkCreateGraphicsPipelines" => layer_vkCreateGraphicsPipelines as _,
        _ => {
            return next_get_device_proc_addr(device, p_name);
        }
    };
    mem::transmute(pfn)
}

// ---------------------------------------------------------------------------
// vk_layerGetPhysicalDeviceProcAddr
// ---------------------------------------------------------------------------

#[no_mangle]
unsafe extern "system" fn layer_vk_layerGetPhysicalDeviceProcAddr(
    instance: vk::Instance,
    p_name: *const c_char,
) -> vk::PFN_vkVoidFunction {
    let name = CStr::from_ptr(p_name);
    let pfn: *const () = match name.to_bytes() {
        b"vkCreateDevice" => layer_vkCreateDevice as _,
        _ => {
            return next_layer_get_physical_device_proc_addr(instance, p_name);
        }
    };
    mem::transmute(pfn)
}

// ---------------------------------------------------------------------------
// vkCreateInstance / vkDestroyInstance
// ---------------------------------------------------------------------------

#[no_mangle]
unsafe extern "system" fn layer_vkCreateInstance(
    p_create_info: *const vk::InstanceCreateInfo,
    p_allocator: *const vk::AllocationCallbacks,
    p_instance: *mut vk::Instance,
) -> vk::Result {
    let create_info = *p_create_info;

    let chain_info = match get_instance_chain_info(&create_info, LayerFunction::LAYER_LINK_INFO) {
        Some(mut p) => p.as_mut(),
        None => return vk::Result::ERROR_INITIALIZATION_FAILED,
    };

    // Consume the head of the layer-info linked list.
    let layer_info = *chain_info.u.p_layer_info;
    chain_info.u.p_layer_info = layer_info.p_next;

    let gipa = layer_info.pfn_next_get_instance_proc_addr.expect("pfnNextGetInstanceProcAddr is null");
    let gpdpa = layer_info.pfn_next_get_physical_device_proc_addr.expect("pfnNextGetPhysicalDeviceProcAddr is null");

    // Call down the chain.
    let create_instance: vk::PFN_vkCreateInstance =
        mem::transmute(gipa(vk::Instance::null(), c"vkCreateInstance".as_ptr()));
    let res = create_instance(p_create_info, p_allocator, p_instance);
    if res != vk::Result::SUCCESS {
        return res;
    }

    let instance = *p_instance;

    // Load ash instance function pointers (next layer's pointers).
    let entry = ash::Entry::from_static_fn(ash::StaticFn { get_instance_proc_addr: gipa });
    let ash_instance = ash::Instance::load(entry.static_fn(), instance);

    // Map every physical device to its parent instance for vkCreateDevice lookup.
    if let Ok(phy_devices) = ash_instance.enumerate_physical_devices() {
        for pd in phy_devices {
            PHY_TO_INSTANCE.insert(pd, instance);
        }
    }

    eprintln!("[planitia-layer] vkCreateInstance {:?}", instance);
    INSTANCE_MAP.insert(
        instance,
        LayerInstance { d: ash_instance, next_get_instance_proc_addr: gipa, next_get_physical_device_proc_addr: gpdpa },
    );

    vk::Result::SUCCESS
}

#[no_mangle]
unsafe extern "system" fn layer_vkDestroyInstance(instance: vk::Instance, p_allocator: *const vk::AllocationCallbacks) {
    if let Some((_, layer_instance)) = INSTANCE_MAP.remove(&instance) {
        if let Ok(phy_devices) = layer_instance.d.enumerate_physical_devices() {
            for pd in phy_devices {
                PHY_TO_INSTANCE.remove(&pd);
            }
        }
        eprintln!("[planitia-layer] vkDestroyInstance {:?}", instance);
        (layer_instance.d.fp_v1_0().destroy_instance)(instance, p_allocator);
    }
}

// ---------------------------------------------------------------------------
// vkCreateDevice / vkDestroyDevice
// ---------------------------------------------------------------------------

#[no_mangle]
unsafe extern "system" fn layer_vkCreateDevice(
    physical_device: vk::PhysicalDevice,
    p_create_info: *const vk::DeviceCreateInfo,
    p_allocator: *const vk::AllocationCallbacks,
    p_device: *mut vk::Device,
) -> vk::Result {
    let instance = *PHY_TO_INSTANCE.get(&physical_device).expect("unknown physical device");
    let layer_instance = INSTANCE_MAP.get(&instance).expect("unknown instance");

    let create_info = *p_create_info;
    let chain_info = match get_device_chain_info(&create_info, LayerFunction::LAYER_LINK_INFO) {
        Some(mut p) => p.as_mut(),
        None => return vk::Result::ERROR_INITIALIZATION_FAILED,
    };

    let layer_info = *chain_info.u.p_layer_info;
    chain_info.u.p_layer_info = layer_info.p_next;

    //let next_get_instance_proc_addr = layer_info.pfn_next_get_instance_proc_addr.expect("pfnNextGetInstanceProcAddr is null");
    let next_get_device_proc_addr = layer_info.pfn_next_get_device_proc_addr.expect("pfnNextGetDeviceProcAddr is null");

    let set_device_loader_data = match get_device_chain_info(&create_info, LayerFunction::LOADER_DATA_CALLBACK) {
        Some(mut p) => p.as_mut().u.pfn_set_device_loader_data.expect("pfnSetDeviceLoaderData is null"),
        None => return vk::Result::ERROR_INITIALIZATION_FAILED,
    };

    // Call down the chain.
    let res = (layer_instance.d.fp_v1_0().create_device)(physical_device, p_create_info, p_allocator, p_device);
    if res != vk::Result::SUCCESS {
        return res;
    }

    // Load device function pointers.
    let device = *p_device;
    let device_d = ash::Device::load_with(
        |func| {
            let fnaddr = next_get_device_proc_addr(device, func.as_ptr());
            mem::transmute(fnaddr)
        },
        device,
    );

    let first_queue_family = {
        let qcis =
            slice::from_raw_parts(create_info.p_queue_create_infos, create_info.queue_create_info_count as usize);
        qcis.first().map_or(0, |q| q.queue_family_index)
    };
    let load_fn = |name: &CStr| mem::transmute(next_get_device_proc_addr(device, name.as_ptr()));
    let khr_swapchain = khr::swapchain::DeviceFn::load(load_fn);
    let khr_dynamic_rendering = khr::dynamic_rendering::DeviceFn::load(load_fn);
    let khr_push_descriptors = khr::push_descriptor::DeviceFn::load(load_fn);

    let command_pool = device_d
        .create_command_pool(
            &vk::CommandPoolCreateInfo {
                flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
                queue_family_index: first_queue_family,
                ..Default::default()
            },
            None,
        )
        .expect("create_command_pool failed");

    let mem_props = layer_instance.d.get_physical_device_memory_properties(physical_device);
    let queue = device_d.get_device_queue(first_queue_family, 0);

    let device_dispatch = DeviceDispatch {
        device: device_d,
        next_get_device_proc_addr,
        set_device_loader_data,
        khr_swapchain,
        khr_dynamic_rendering,
        khr_push_descriptors,
    };

    let device_helper = DeviceHelper { dispatch: device_dispatch, command_pool, queue, mem_props };
    device_helper.set_device_loader_data(queue);

    let static_resources = overlay::initialize_static_resources(&device_helper);
    let frame_resources = overlay::initialize_frame_resources(&device_helper);
    let tracked_resources =
        TrackedResources { pipelines: Vec::new(), swapchains: Vec::new(), present_image_copy: None };
    let overlay_resources = OverlayResources::default();

    let device_data = DeviceData {
        helper: device_helper,
        static_resources,
        frame_resources: Mutex::new(frame_resources),
        first_queue_family,
        tracked_resources: Mutex::new(tracked_resources),
        overlay_resources: Mutex::new(overlay_resources),
    };

    DEVICE_DATA.insert(device, device_data);
    eprintln!("[planitia-layer] vkCreateDevice {:?}", device);
    vk::Result::SUCCESS
}

#[no_mangle]
unsafe extern "system" fn layer_vkDestroyDevice(device: vk::Device, p_allocator: *const vk::AllocationCallbacks) {
    if let Some((_, device_data)) = DEVICE_DATA.remove(&device) {
        eprintln!("[planitia-layer] vkDestroyDevice {:?}", device);
        (device_data.fp_v1_0().destroy_device)(device, p_allocator);
    }
}

// ---------------------------------------------------------------------------
// Queue tracking (queue handle → queue family index)
// ---------------------------------------------------------------------------

static QUEUE_MAP: LazyLock<DashMap<vk::Queue, LayerQueue>> = LazyLock::new(DashMap::new);

#[no_mangle]
unsafe extern "system" fn layer_vkGetDeviceQueue(
    device: vk::Device,
    queue_family_index: u32,
    queue_index: u32,
    p_queue: *mut vk::Queue,
) {
    (device_data(device).fp_v1_0().get_device_queue)(device, queue_family_index, queue_index, p_queue);

    if let Some(&queue) = p_queue.as_ref() {
        if queue != vk::Queue::null() {
            QUEUE_MAP.insert(queue, LayerQueue { device, queue_family_index });
        }
    }
}

// ---------------------------------------------------------------------------
// vkCreateGraphicsPipelines
// ---------------------------------------------------------------------------
#[no_mangle]
unsafe extern "system" fn layer_vkCreateGraphicsPipelines(
    device: vk::Device,
    pipeline_cache: vk::PipelineCache,
    create_info_count: u32,
    p_create_infos: *const vk::GraphicsPipelineCreateInfo<'_>,
    p_allocator: *const vk::AllocationCallbacks<'_>,
    p_pipelines: *mut vk::Pipeline,
) -> vk::Result {
    eprintln!("[planitia-layer] vkCreateGraphicsPipelines {:?} count={}", device, create_info_count);

    // TODO:
    // - read shader module data, extract SPIR-V
    // - extract types
    // - create the debug overlay pipeline
    // - hook vkQueuePresent
    //    - before present: wait on the semaphores specified in vkQueuePresent
    //          - for now we can rely on those being binary semaphores only
    //    - vkDeviceWaitIdle, just to be sure
    //    - extract swapchain image
    //    - render overlay
    //    - signal semaphore
    //    - vkQueuePresent with the other semaphore

    let create_infos = slice::from_raw_parts(p_create_infos, create_info_count as usize);

    for create_info in create_infos {
        let stages = slice::from_raw_parts(create_info.p_stages, create_info.stage_count as usize);
        for stage in stages {
            if let Some(smci) = find_next::<vk::ShaderModuleCreateInfo>(stage) {
                // Process the shader module create info
                eprintln!(
                    "[planitia-layer] Found ShaderModuleCreateInfo for stage {:?}, code size: {} bytes",
                    stage.stage,
                    (*smci).code_size
                );
            }
        }
    }

    let result = (device_data(device).fp_v1_0().create_graphics_pipelines)(
        device,
        pipeline_cache,
        create_info_count,
        p_create_infos,
        p_allocator,
        p_pipelines,
    );

    if result == vk::Result::SUCCESS {
        let pipelines = slice::from_raw_parts(p_pipelines, create_info_count as usize);
        for i in 0..create_info_count {
            let dd = device_data(device);
            dd.tracked_resources.lock().unwrap().pipelines.push(PipelineData { pipeline: pipelines[i as usize] });
        }
    }

    result
}

// Type checks
const _: PFN_vkNegotiateLoaderLayerInterfaceVersion = vkNegotiateLoaderLayerInterfaceVersion;
const _: vk::PFN_vkGetInstanceProcAddr = layer_vkGetInstanceProcAddr;
const _: vk::PFN_vkGetDeviceProcAddr = layer_vkGetDeviceProcAddr;
const _: PFN_vk_layerGetPhysicalDeviceProcAddr = layer_vk_layerGetPhysicalDeviceProcAddr;
const _: vk::PFN_vkCreateInstance = layer_vkCreateInstance;
const _: vk::PFN_vkDestroyInstance = layer_vkDestroyInstance;
const _: vk::PFN_vkCreateDevice = layer_vkCreateDevice;
const _: vk::PFN_vkDestroyDevice = layer_vkDestroyDevice;
const _: vk::PFN_vkGetDeviceQueue = layer_vkGetDeviceQueue;
const _: vk::PFN_vkCreateGraphicsPipelines = layer_vkCreateGraphicsPipelines;
