//! Vulkan device function wrappers (hooks).
//! Layer entry points & device function wrappers (hooks).

use crate::dispatch::{DeviceDispatchableHandle, InstanceDispatch};
use crate::{DEVICE_STATE, Device, INSTANCE_MAP, PHY_TO_INSTANCE, device_state};
use dashmap::DashMap;
use std::ffi::{CStr, c_char, c_void};
use std::mem;
use std::ptr::NonNull;
use std::sync::LazyLock;
use vulkan::vk_layer::{
    VK_LAYER_LINK_INFO, VK_LOADER_DATA_CALLBACK, VkLayerDeviceCreateInfo, VkLayerFunction, VkLayerInstanceCreateInfo,
};
use vulkan::*;
use windows::Win32::Foundation::HWND;

pub mod buffer;
pub mod command;
pub mod image;
pub mod memory;
pub mod pipeline;
pub mod queue;
pub mod swapchain;

//--------------------------------------------------------------------------------------------------
// Layer DLL interfaces
//--------------------------------------------------------------------------------------------------

const _: PFN_vkGetInstanceProcAddr = layer_vkGetInstanceProcAddr;
const _: PFN_vkGetDeviceProcAddr = layer_vkGetDeviceProcAddr;
const _: vk_layer::PFN_GetPhysicalDeviceProcAddr = layer_vk_layerGetPhysicalDeviceProcAddr;
const _: vk_layer::PFN_vkNegotiateLoaderLayerInterfaceVersion = vkNegotiateLoaderLayerInterfaceVersion;

#[unsafe(no_mangle)]
pub(crate) unsafe extern "system" fn layer_vkGetInstanceProcAddr(
    instance: VkInstance,
    p_name: *const c_char,
) -> PFN_vkVoidFunction {
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
#[unsafe(no_mangle)]
unsafe extern "system" fn layer_vkGetDeviceProcAddr(device: VkDevice, p_name: *const c_char) -> PFN_vkVoidFunction {
    match get_device_proc_addr_hook(p_name) {
        Some(ptr) => ptr,
        None => {
            // Unhooked functions
            let d = DEVICE_STATE.get(&device.key()).expect("unknown device");
            (d.next_get_device_proc_addr)(device, p_name)
        }
    }
}

#[unsafe(no_mangle)]
pub(crate) unsafe extern "system" fn layer_vk_layerGetPhysicalDeviceProcAddr(
    instance: VkInstance,
    p_name: *const c_char,
) -> PFN_vkVoidFunction {
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

#[unsafe(no_mangle)]
unsafe extern "system" fn vkNegotiateLoaderLayerInterfaceVersion(
    p_version_struct: *mut vk_layer::VkNegotiateLayerInterface,
) -> VkResult {
    (*p_version_struct).loaderLayerInterfaceVersion = 2;
    (*p_version_struct).pfnGetInstanceProcAddr = layer_vkGetInstanceProcAddr;
    (*p_version_struct).pfnGetDeviceProcAddr = layer_vkGetDeviceProcAddr;
    (*p_version_struct).pfnGetPhysicalDeviceProcAddr = layer_vk_layerGetPhysicalDeviceProcAddr;
    VK_SUCCESS
}

//--------------------------------------------------------------------------------------------------
// Instance/device creation hooks
//--------------------------------------------------------------------------------------------------

const _: PFN_vkCreateInstance = layer_vkCreateInstance;
const _: PFN_vkDestroyInstance = layer_vkDestroyInstance;
const _: PFN_vkCreateDevice = layer_vkCreateDevice;
const _: PFN_vkDestroyDevice = layer_vkDestroyDevice;

// Adapted from ash_layer
pub unsafe fn get_instance_chain_info(
    create_info: &VkInstanceCreateInfo,
    function: VkLayerFunction,
) -> Option<NonNull<VkLayerInstanceCreateInfo>> {
    let mut chain_info_ptr = create_info.pNext.cast::<VkLayerInstanceCreateInfo>();
    while !chain_info_ptr.is_null() {
        let chain_info = chain_info_ptr.read();
        if chain_info.sType == VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO && chain_info.function == function {
            return Some(NonNull::new_unchecked(chain_info_ptr as _));
        }
        chain_info_ptr = chain_info.pNext.cast()
    }
    None
}

pub unsafe fn get_device_chain_info(
    create_info: &VkDeviceCreateInfo,
    function: VkLayerFunction,
) -> Option<NonNull<VkLayerDeviceCreateInfo>> {
    let mut chain_info_ptr = create_info.pNext.cast::<VkLayerDeviceCreateInfo>();
    while !chain_info_ptr.is_null() {
        let chain_info = chain_info_ptr.read();
        if chain_info.sType == VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO && chain_info.function == function {
            return Some(NonNull::new_unchecked(chain_info_ptr as _));
        }
        chain_info_ptr = chain_info.pNext.cast()
    }
    None
}

#[unsafe(no_mangle)]
pub(crate) unsafe extern "system" fn layer_vkCreateInstance(
    p_create_info: *const VkInstanceCreateInfo,
    p_allocator: *const VkAllocationCallbacks,
    p_instance: *mut VkInstance,
) -> VkResult {
    let mut create_info = *p_create_info;
    let chain_info = match get_instance_chain_info(&create_info, VK_LAYER_LINK_INFO) {
        Some(mut p) => p.as_mut(),
        None => return VK_ERROR_INITIALIZATION_FAILED,
    };
    // Consume the head of the layer-info linked list.
    let layer_info = *chain_info.u.pLayerInfo;
    chain_info.u.pLayerInfo = layer_info.pNext;
    let gipa = layer_info.pfnNextGetInstanceProcAddr.unwrap();
    let gpdpa = layer_info.pfnNextGetPhysicalDeviceProcAddr.unwrap();

    // Bump API version to 1.4
    let mut app_info = VkApplicationInfo { .. };
    if create_info.pApplicationInfo.is_null() {
        app_info.apiVersion = vk_make_api_version(0, 1, 4, 0);
    } else {
        app_info = *(create_info.pApplicationInfo);
        let api_version = app_info.apiVersion;
        eprintln!("[planitia-layer] Application Name: {}", CStr::from_ptr(app_info.pApplicationName).to_string_lossy());
        eprintln!(
            "[planitia-layer] Vulkan API version: {}.{}.{}",
            vk_api_version_major(api_version),
            vk_api_version_minor(api_version),
            vk_api_version_patch(api_version)
        );
        app_info.apiVersion = vk_make_api_version(vk_api_version_variant(api_version), 1, 4, 0);
    }
    create_info.pApplicationInfo = &app_info;

    // Call down the chain.
    let create_instance: PFN_vkCreateInstance = mem::transmute(gipa(VkInstance::null(), c"vkCreateInstance".as_ptr()));
    let res = create_instance(&create_info, p_allocator, p_instance);
    if res < 0 {
        return res;
    }
    let instance = *p_instance;
    // Load ash instance function pointers (next layer's pointers).
    eprintln!("[planitia-layer] vkCreateInstance {:?}", instance);
    let dispatch = InstanceDispatch::new(gipa, gpdpa, instance);
    // Map every physical device to its parent instance for vkCreateDevice lookup.
    vkarraycall!(dispatch.EnumeratePhysicalDevices(instance, @count let count, @out let phy_devices));
    for pd in phy_devices {
        PHY_TO_INSTANCE.insert(pd, instance);
    }
    INSTANCE_MAP.insert(instance, dispatch);
    res
}

#[unsafe(no_mangle)]
pub(crate) unsafe extern "system" fn layer_vkDestroyInstance(
    instance: VkInstance,
    p_allocator: *const VkAllocationCallbacks,
) {
    if let Some((_, layer_instance)) = INSTANCE_MAP.remove(&instance) {
        vkarraycall!(layer_instance.fns.EnumeratePhysicalDevices(instance, @count let count, @out let phy_devices));
        for pd in phy_devices {
            PHY_TO_INSTANCE.remove(&pd);
        }
        eprintln!("[planitia-layer] vkDestroyInstance {:?}", instance);
        layer_instance.fns.DestroyInstance(instance, p_allocator);
    }
}

#[unsafe(no_mangle)]
pub(crate) unsafe extern "system" fn layer_vkCreateDevice(
    physical_device: VkPhysicalDevice,
    p_create_info: *const VkDeviceCreateInfo,
    p_allocator: *const VkAllocationCallbacks,
    p_device: *mut VkDevice,
) -> VkResult {
    let instance = *PHY_TO_INSTANCE.get(&physical_device).expect("unknown physical device");
    let instance_dispatch = INSTANCE_MAP.get(&instance).expect("unknown instance");
    let chain_info = match get_device_chain_info(&*p_create_info, VK_LAYER_LINK_INFO) {
        Some(mut p) => p.as_mut(),
        None => return VK_ERROR_INITIALIZATION_FAILED,
    };
    let layer_info = *chain_info.u.pLayerInfo;
    chain_info.u.pLayerInfo = layer_info.pNext;
    //let next_get_instance_proc_addr = layer_info.pfn_next_get_instance_proc_addr.expect("pfnNextGetInstanceProcAddr is null");
    let next_get_device_proc_addr = layer_info.pfnNextGetDeviceProcAddr.unwrap();
    let set_device_loader_data = match get_device_chain_info(&*p_create_info, VK_LOADER_DATA_CALLBACK) {
        Some(mut p) => p.as_mut().u.pfnSetDeviceLoaderData.unwrap(),
        None => return VK_ERROR_INITIALIZATION_FAILED,
    };
    let res = instance_dispatch.CreateDevice(physical_device, p_create_info, p_allocator, p_device);
    if res < 0 {
        return res;
    }
    let device = *p_device;
    let create_info = &*p_create_info;
    let device_state = Device::new(
        &instance_dispatch,
        device,
        create_info,
        physical_device,
        next_get_device_proc_addr,
        set_device_loader_data,
    );
    DEVICE_STATE.insert(device.key(), device_state);
    //eprintln!("[planitia-layer] vkCreateDevice {:?}", device);
    res
}

#[unsafe(no_mangle)]
pub(crate) unsafe extern "system" fn layer_vkDestroyDevice(
    device: VkDevice,
    p_allocator: *const VkAllocationCallbacks,
) {
    if let Some((_, device_state)) = DEVICE_STATE.remove(&device.key()) {
        //eprintln!("[planitia-layer] vkDestroyDevice {:?}", device);
        device_state.DestroyDevice(device, p_allocator);
    }
}

//--------------------------------------------------------------------------------------------------
// Surface hooks
//--------------------------------------------------------------------------------------------------

struct SurfaceInfo {
    // TODO other platforms
    hwnd: HWND,
}
unsafe impl Send for SurfaceInfo {}
unsafe impl Sync for SurfaceInfo {}

static SURFACES: LazyLock<DashMap<VkSurfaceKHR, SurfaceInfo>> = LazyLock::new(DashMap::new);

const _: PFN_vkCreateWin32SurfaceKHR = layer_vkCreateWin32SurfaceKHR;

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

/// Returns the HWND corresponding to the specified VkSurface handle.
#[cfg(windows)]
pub fn get_hwnd_for_surface(surface: VkSurfaceKHR) -> Option<HWND> {
    SURFACES.get(&surface).map(|info| info.hwnd)
}

//--------------------------------------------------------------------------------------------------

impl Device {
    pub unsafe fn hook_set_debug_utils_object_name(
        &self,
        device: VkDevice,
        p_name_info: *const VkDebugUtilsObjectNameInfoEXT,
    ) -> VkResult {
        let handle = (*p_name_info).objectHandle;
        let name = CStr::from_ptr((*p_name_info).pObjectName).to_string_lossy().into_owned();
        match (*p_name_info).objectType {
            VK_OBJECT_TYPE_COMMAND_BUFFER => {
                let cmd_buf = VkCommandBuffer(handle as *mut _);
                if let Some(data) = self.get_private_data_mut(cmd_buf) {
                    data.name = name;
                }
            }
            VK_OBJECT_TYPE_PIPELINE => {
                let pipeline = VkPipeline(handle);
                if let Some(data) = self.get_private_data_mut(pipeline) {
                    data.name = name;
                }
            }
            VK_OBJECT_TYPE_BUFFER => {
                let buffer = VkBuffer(handle);
                if let Some(data) = self.get_private_data_mut(buffer) {
                    data.name = name;
                }
            }
            VK_OBJECT_TYPE_IMAGE => {
                let image = VkImage(handle);
                if let Some(data) = self.get_private_data_mut(image) {
                    data.name = name;
                }
            }
            VK_OBJECT_TYPE_QUEUE => {
                // TODO
            }
            _ => {}
        }
        self.SetDebugUtilsObjectNameEXT(device, p_name_info)
    }
}

//--------------------------------------------------------------------------------------------------
// Device hook table
//--------------------------------------------------------------------------------------------------

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

        pub(crate) unsafe fn get_device_proc_addr_hook(
            p_name: *const c_char,
        ) -> Option<PFN_vkVoidFunction> {
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
    [b"vkGetDeviceQueue"; PFN_vkGetDeviceQueue] fn layer_vkGetDeviceQueue(device: VkDevice, queue_family_index: u32, queue_index: u32, p_queue: *mut VkQueue) = hook_get_device_queue;

    [b"vkCmdPushDataEXT"; PFN_vkCmdPushDataEXT] fn layer_vkCmdPushDataEXT(commandBuffer: VkCommandBuffer, pPushDataInfo: *const VkPushDataInfoEXT) = hook_cmd_push_data_ext;
    [b"vkCmdBindResourceHeapEXT"; PFN_vkCmdBindResourceHeapEXT] fn layer_vkCmdBindResourceHeapEXT(commandBuffer: VkCommandBuffer, pBindInfo: *const VkBindHeapInfoEXT) = hook_cmd_bind_resource_heap_ext;
    [b"vkCmdBindSamplerHeapEXT"; PFN_vkCmdBindSamplerHeapEXT] fn layer_vkCmdBindSamplerHeapEXT(commandBuffer: VkCommandBuffer, pBindInfo: *const VkBindHeapInfoEXT) = hook_cmd_bind_sampler_heap_ext;
    [b"vkWriteResourceDescriptorsEXT"; PFN_vkWriteResourceDescriptorsEXT] fn layer_vkWriteResourceDescriptorsEXT(device: VkDevice, resourceCount: u32, pResources: *const VkResourceDescriptorInfoEXT, pDescriptors: *const VkHostAddressRangeEXT) -> VkResult = hook_write_resource_descriptors_ext;
    [b"vkWriteSamplerDescriptorsEXT"; PFN_vkWriteSamplerDescriptorsEXT] fn layer_vkWriteSamplerDescriptorsEXT(device: VkDevice, samplerCount: u32, pSamplers: *const VkSamplerCreateInfo, pDescriptors: *const VkHostAddressRangeEXT) -> VkResult = hook_write_sampler_descriptors_ext;

    [b"vkAllocateCommandBuffers"; PFN_vkAllocateCommandBuffers] fn layer_vkAllocateCommandBuffers(device: VkDevice, pAllocateInfo: *const VkCommandBufferAllocateInfo, pCommandBuffers: *mut VkCommandBuffer) -> VkResult = hook_allocate_command_buffers;
    [b"vkFreeCommandBuffers"; PFN_vkFreeCommandBuffers] fn layer_vkFreeCommandBuffers(device: VkDevice, commandPool: VkCommandPool, commandBufferCount: u32, pCommandBuffers: *const VkCommandBuffer) = hook_free_command_buffers;
    [b"vkBeginCommandBuffer"; PFN_vkBeginCommandBuffer] fn layer_vkBeginCommandBuffer(commandBuffer: VkCommandBuffer, pBeginInfo: *const VkCommandBufferBeginInfo) -> VkResult = hook_begin_command_buffer;
    [b"vkEndCommandBuffer"; PFN_vkEndCommandBuffer] fn layer_vkEndCommandBuffer(commandBuffer: VkCommandBuffer) -> VkResult = hook_end_command_buffer;

    [b"vkCreateGraphicsPipelines"; PFN_vkCreateGraphicsPipelines] fn layer_vkCreateGraphicsPipelines(device: VkDevice, pipelineCache: VkPipelineCache, createInfoCount: u32, pCreateInfos: *const VkGraphicsPipelineCreateInfo, pAllocator: *const VkAllocationCallbacks, pPipelines: *mut VkPipeline) -> VkResult = hook_create_graphics_pipelines;
    [b"vkCreateComputePipelines"; PFN_vkCreateComputePipelines] fn layer_vkCreateComputePipelines(device: VkDevice, pipelineCache: VkPipelineCache, createInfoCount: u32, pCreateInfos: *const VkComputePipelineCreateInfo, pAllocator: *const VkAllocationCallbacks, pPipelines: *mut VkPipeline) -> VkResult = hook_create_compute_pipelines;
    [b"vkDestroyPipeline"; PFN_vkDestroyPipeline] fn layer_vkDestroyPipeline(device: VkDevice, pipeline: VkPipeline, pAllocator: *const VkAllocationCallbacks) = hook_destroy_pipeline;

    [b"vkCmdBindPipeline"; PFN_vkCmdBindPipeline] fn layer_vkCmdBindPipeline(command_buffer: VkCommandBuffer, pipeline_bind_point: VkPipelineBindPoint, pipeline: VkPipeline) = hook_cmd_bind_pipeline;
    [b"vkCmdDraw"; PFN_vkCmdDraw] fn layer_vkCmdDraw(commandBuffer: VkCommandBuffer, vertexCount: u32, instanceCount: u32, firstVertex: u32, firstInstance: u32) = hook_cmd_draw;
    [b"vkCmdDrawIndexed"; PFN_vkCmdDrawIndexed] fn layer_vkCmdDrawIndexed(commandBuffer: VkCommandBuffer, indexCount: u32, instanceCount: u32, firstIndex: u32, vertexOffset: i32, firstInstance: u32) = hook_cmd_draw_indexed;
    [b"vkCmdDrawIndirect"; PFN_vkCmdDrawIndirect] fn layer_vkCmdDrawIndirect(commandBuffer: VkCommandBuffer, buffer: VkBuffer, offset: VkDeviceSize, drawCount: u32, stride: u32) = hook_cmd_draw_indirect;
    [b"vkCmdDispatch"; PFN_vkCmdDispatch] fn layer_vkCmdDispatch(commandBuffer: VkCommandBuffer, groupCountX: u32, groupCountY: u32, groupCountZ: u32) = hook_cmd_dispatch;
    [b"vkCmdBeginDebugUtilsLabelEXT"; PFN_vkCmdBeginDebugUtilsLabelEXT] fn layer_vkCmdBeginDebugUtilsLabelEXT(commandBuffer: VkCommandBuffer, pLabelInfo: *const VkDebugUtilsLabelEXT) = hook_cmd_begin_debug_utils_label;
    [b"vkCmdBeginRenderPass"; PFN_vkCmdBeginRenderPass] fn layer_vkCmdBeginRenderPass(commandBuffer: VkCommandBuffer, pRenderPassBegin: *const VkRenderPassBeginInfo, contents: VkSubpassContents) = hook_cmd_begin_render_pass;
    [b"vkCmdEndRenderPass"; PFN_vkCmdEndRenderPass] fn layer_vkCmdEndRenderPass(commandBuffer: VkCommandBuffer) = hook_cmd_end_render_pass;
    [b"vkCmdBeginRenderPass2"; PFN_vkCmdBeginRenderPass2] fn layer_vkCmdBeginRenderPass2(command_buffer: VkCommandBuffer, p_render_pass_begin: *const VkRenderPassBeginInfo, p_subpass_begin_info: *const VkSubpassBeginInfo) = hook_cmd_begin_render_pass2;
    [b"vkCmdEndRenderPass2"; PFN_vkCmdEndRenderPass2] fn layer_vkCmdEndRenderPass2(command_buffer: VkCommandBuffer, p_subpass_end_info: *const VkSubpassEndInfo) = hook_cmd_end_render_pass2;
    [b"vkCmdBeginRendering", b"vkCmdBeginRenderingKHR"; PFN_vkCmdBeginRendering] fn layer_vkCmdBeginRendering(commandBuffer: VkCommandBuffer, pRenderingInfo: *const VkRenderingInfo) = hook_cmd_begin_rendering;
    [b"vkCmdEndRendering", b"vkCmdEndRenderingKHR"; PFN_vkCmdEndRendering] fn layer_vkCmdEndRendering(commandBuffer: VkCommandBuffer) = hook_cmd_end_rendering;

    [b"vkCreateSwapchainKHR"; PFN_vkCreateSwapchainKHR] fn layer_vkCreateSwapchainKHR(device: VkDevice, pCreateInfo: *const VkSwapchainCreateInfoKHR, pAllocator: *const VkAllocationCallbacks, pSwapchain: *mut VkSwapchainKHR) -> VkResult = hook_create_swapchain_khr;
    [b"vkDestroySwapchainKHR"; PFN_vkDestroySwapchainKHR] fn layer_vkDestroySwapchainKHR(device: VkDevice, swapchain: VkSwapchainKHR, pAllocator: *const VkAllocationCallbacks) = hook_destroy_swapchain_khr;

    [b"vkQueueSubmit"; PFN_vkQueueSubmit] fn layer_vkQueueSubmit(queue: VkQueue, submit_count: u32, p_submits: *const VkSubmitInfo, fence: VkFence) -> VkResult = hook_queue_submit;
    [b"vkQueueSubmit2"; PFN_vkQueueSubmit2] fn layer_vkQueueSubmit2(queue: VkQueue, submitCount: u32, pSubmits: *const VkSubmitInfo2, fence: VkFence) -> VkResult = hook_queue_submit_2;
    [b"vkQueuePresentKHR"; PFN_vkQueuePresentKHR] fn layer_vkQueuePresentKHR(queue: VkQueue, p_present_info: *const VkPresentInfoKHR) -> VkResult = hook_queue_present_khr;

    [b"vkCreateBuffer"; PFN_vkCreateBuffer] fn layer_vkCreateBuffer(device: VkDevice, pCreateInfo: *const VkBufferCreateInfo, pAllocator: *const VkAllocationCallbacks, pBuffer: *mut VkBuffer) -> VkResult = hook_create_buffer;
    [b"vkDestroyBuffer"; PFN_vkDestroyBuffer] fn layer_vkDestroyBuffer(device: VkDevice, buffer: VkBuffer, pAllocator: *const VkAllocationCallbacks) = hook_destroy_buffer;
    [b"vkBindBufferMemory"; PFN_vkBindBufferMemory] fn layer_vkBindBufferMemory(device: VkDevice, buffer: VkBuffer, memory: VkDeviceMemory, memoryOffset: VkDeviceSize) -> VkResult = hook_bind_buffer_memory;
    [b"vkBindBufferMemory2"; PFN_vkBindBufferMemory2] fn layer_vkBindBufferMemory2(device: VkDevice, bind_info_count: u32, p_bind_infos: *const VkBindBufferMemoryInfo) -> VkResult = hook_bind_buffer_memory_2;

    [b"vkCmdEndDebugUtilsLabelEXT"; PFN_vkCmdEndDebugUtilsLabelEXT] fn layer_vkCmdEndDebugUtilsLabelEXT(commandBuffer: VkCommandBuffer) = hook_cmd_end_debug_utils_label;
    [b"vkSetDebugUtilsObjectNameEXT"; PFN_vkSetDebugUtilsObjectNameEXT] fn layer_vkSetDebugUtilsObjectNameEXT(device: VkDevice, p_name_info: *const VkDebugUtilsObjectNameInfoEXT) -> VkResult = hook_set_debug_utils_object_name;

    [b"vkCreateImageView"; PFN_vkCreateImageView] fn layer_vkCreateImageView(device: VkDevice, pCreateInfo: *const VkImageViewCreateInfo, pAllocator: *const VkAllocationCallbacks, pView: *mut VkImageView) -> VkResult = hook_create_image_view;
    [b"vkDestroyImageView"; PFN_vkDestroyImageView] fn layer_vkDestroyImageView(device: VkDevice, imageView: VkImageView, pAllocator: *const VkAllocationCallbacks) = hook_destroy_image_view;

    [b"vkCreateImage"; PFN_vkCreateImage] fn layer_vkCreateImage(device: VkDevice, pCreateInfo: *const VkImageCreateInfo, pAllocator: *const VkAllocationCallbacks, pImage: *mut VkImage) -> VkResult = hook_create_image;
    [b"vkDestroyImage"; PFN_vkDestroyImage] fn layer_vkDestroyImage(device: VkDevice, image: VkImage, pAllocator: *const VkAllocationCallbacks) = hook_destroy_image;
    [b"vkBindImageMemory"; PFN_vkBindImageMemory] fn layer_vkBindImageMemory(device: VkDevice, image: VkImage, memory: VkDeviceMemory, memoryOffset: VkDeviceSize) -> VkResult = hook_bind_image_memory;
    [b"vkBindImageMemory2"; PFN_vkBindImageMemory2] fn layer_vkBindImageMemory2(device: VkDevice, bind_info_count: u32, p_bind_infos: *const VkBindImageMemoryInfo) -> VkResult = hook_bind_image_memory_2;
}
