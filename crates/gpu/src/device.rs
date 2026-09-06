//! Abstractions over a vulkan device & queues.
//mod bindless;
mod descriptor_heap;

use crate::device::descriptor_heap::DescriptorHeaps;
use crate::instance::vk_khr_surface;
use crate::platform::PlatformExtensions;
use crate::{
    BufferAddressRange, BufferUsage, ComputePipeline, ComputePipelineCreateInfo, DescriptorSetLayout, Error,
    FrameIndex, GraphicsPipeline, GraphicsPipelineCreateInfo, Instance, PreRasterizationShaders, Ptr, SUBGROUP_SIZE,
    SamplerParams, SamplerParamsHashable, ShaderReflection, VulkanObject, get_vulkan_entry, get_vulkan_instance,
    is_depth_and_stencil_format, signal, vkcheck,
};
use ash::vk;
use ash::vk::Handle;
use gpu::device::descriptor_heap::SamplerDescriptorHandle;
use gpu::flush;
use gpu_allocator::vulkan::AllocationCreateDesc;
use gpu_types::{SamplerHandle, ShaderStage};
use log::{debug, error, info, trace, warn};
use slotmap::{SlotMap, new_key_type};
use std::collections::{HashMap, VecDeque};
use std::ffi::{CStr, CString, c_void};
use std::mem::MaybeUninit;
use std::ops::Range;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::{Arc, LazyLock, Mutex};
use std::time::Duration;
use std::{fmt, mem, ptr};
use vulkan::*;

////////////////////////////////////////////////////////////////////////////////////////////////////

// Sizes of the global descriptor heaps (in number of descriptors).
const RESOURCE_DESCRIPTOR_HEAP_SIZE: usize = 1024 * 1024;
const SAMPLER_DESCRIPTOR_HEAP_SIZE: usize = 64 * 1024;

////////////////////////////////////////////////////////////////////////////////////////////////////

new_key_type! {
    /// Identifies an image resource (sampled or storage image) in a bindless descriptor heap.
    pub(crate) struct ResourceDescriptorIndex;
    /// Identifies a sampler in a bindless sampler descriptor heap.
    pub struct SamplerDescriptorIndex;
}

/// Device extensions.
pub(crate) struct DeviceExtensions {
    pub(crate) swapchain: khr_swapchain::DeviceDispatch,
    //pub(crate) ext_shader_object: ash::ext::,
    pub(crate) push_descriptor: khr_push_descriptor::DeviceDispatch,
    pub(crate) calibrated_timestamps: khr_calibrated_timestamps::DeviceDispatch,
    pub(crate) mesh_shader: ext_mesh_shader::DeviceDispatch,
    pub(crate) _ext_extended_dynamic_state3: ext_extended_dynamic_state3::DeviceDispatch,
    pub(crate) debug_utils: ext_debug_utils::DeviceDispatch,
    pub(crate) descriptor_heap_instance: ext_descriptor_heap::InstanceDispatch,
    pub(crate) descriptor_heap: ext_descriptor_heap::DeviceDispatch,
}

/// Device state that is unconditionally safe to access from multiple threads, even though
/// the fields themselves may not be Send or Sync.
pub(crate) struct DeviceThreadSafeState {
    pub(crate) physical_device_memory_properties: VkPhysicalDeviceMemoryProperties,
    pub(crate) physical_device_id_properties: VkPhysicalDeviceIDProperties,
    pub(crate) descriptor_heap_properties: VkPhysicalDeviceDescriptorHeapPropertiesEXT,
    physical_device_properties: VkPhysicalDeviceProperties,
    /// Timeline used to track completion of frames.
    /// It is incremented and signalled on each frame completion (see `poll`).
    // SAFETY: we're never using this as an externally-synchronized command parameter.
    pub(crate) frame_timeline: VkSemaphore,
    // SAFETY: we're never using this as an externally-synchronized command parameter.
    pub(crate) physical_device: VkPhysicalDevice,
}

unsafe impl Send for DeviceThreadSafeState {}
unsafe impl Sync for DeviceThreadSafeState {}

/// Submission-related device state locked during command buffer submission.
pub(crate) struct DeviceSubmissionState {
    pub(crate) queue: VkQueue,
    /// Sorted by create_ticket, not by order of submission.
    pub(crate) active_submissions: VecDeque<ActiveSubmission>,
}

pub struct Device {
    /// Underlying vulkan device
    //pub(crate) raw: ash::Device,
    pub(crate) vkd: VkDevice,
    pub(crate) vk: Vulkan_1_4_DeviceDispatch,
    /// Common device extensions.
    pub(crate) ext: DeviceExtensions,
    /// Platform-specific extension functions
    pub(crate) platform_extensions: PlatformExtensions,
    pub(crate) allocator: Mutex<gpu_allocator::vulkan::Allocator>,
    /// Queue family index of the main queue.
    pub(crate) queue_family: u32,
    pub(crate) thread_safe: DeviceThreadSafeState,
    pub(crate) submission_state: Mutex<DeviceSubmissionState>,
    // WIP
    pub(crate) descriptor_heaps: DescriptorHeaps,
    // --- descriptor heap ---
    /// semaphores ready for reuse.
    pub(crate) semaphores: Mutex<Vec<VkSemaphore>>,
    // Index of the next submission not yet created.
    //pub(crate) next_create_ticket: AtomicU64,
    /// The index of the frame being recorded, or, equivalently, the next frame index to be signalled.
    pub(crate) frame_index: AtomicU64,
    /// Destructors (or other function calls) that are delayed until associated command buffers
    /// have completed execution.
    ///
    /// Note that the deletion queue is sorted by create_ticket, which is not necessarily the same as
    /// submission order, in case the user submits command buffers out-of-order.
    /// This means that even if a submission has completed execution, deletion of the associated
    /// resources are delayed until all submissions **with a lower create_ticket** have also completed.
    /// This is necessary to avoid unsound scenarios where resources are deleted while still in use
    /// by the GPU, due to command buffers being submitted out-of-order.
    deletion_queue: Mutex<Vec<DeleteQueueEntry>>,
    pub(crate) sampler_cache: Mutex<HashMap<SamplerParamsHashable, SamplerDescriptorHandle>>,
}

impl fmt::Debug for Device {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("DeviceInner").finish_non_exhaustive()
    }
}

/// Data and resources associated to a submission that was submitted to the GPU.
pub(crate) struct ActiveSubmission {
    //pub(crate) create_ticket: u64,
    pub(crate) frame_index: u64,
    //pub(crate) command_pools: Vec<CommandPool>,
    //pub(crate) timestamp_query_pool: vk::QueryPool,
    //pub(crate) timestamp_query_count: u32,
    //pub(crate) timestamp_callbacks: Vec<Box<dyn FnOnce(u64) + Send>>,
}

struct DeleteQueueEntry {
    frame_index: u64,
    deleter: Option<Box<dyn FnOnce(&Device) + Send + Sync>>,
}

/// Errors during device creation.
#[derive(thiserror::Error, Debug)]
pub enum DeviceCreateError {
    #[error(transparent)]
    Vulkan(#[from] vk::Result),
}

#[derive(Copy, Clone, Debug)]
pub struct QueueFamilyConfig {
    pub family_index: u32,
    pub count: u32,
}

pub(crate) fn get_vk_sample_count(count: u32) -> VkSampleCountFlags {
    match count {
        0 => vk::SampleCountFlags::TYPE_1,
        1 => vk::SampleCountFlags::TYPE_1,
        2 => vk::SampleCountFlags::TYPE_2,
        4 => vk::SampleCountFlags::TYPE_4,
        8 => vk::SampleCountFlags::TYPE_8,
        16 => vk::SampleCountFlags::TYPE_16,
        32 => vk::SampleCountFlags::TYPE_32,
        64 => vk::SampleCountFlags::TYPE_64,
        _ => panic!("unsupported number of samples"),
    }
}

impl ResourceDescriptorIndex {
    /// Returns the index of this resource in the global resource descriptor heap.
    pub(crate) fn index(&self) -> u32 {
        (self.0.as_ffi() & 0xFFFF_FFFF) as u32
    }
}

impl SamplerDescriptorIndex {
    /// Returns the index of this sampler in the global sampler descriptor heap.
    pub(crate) fn index(&self) -> u32 {
        (self.0.as_ffi() & 0xFFFF_FFFF) as u32
    }
}

/// Describes how a resource got its memory.
#[derive(Default, Debug)]
pub(crate) enum ResourceAllocation {
    /// We don't own the memory for this resource.
    #[default]
    External,
    /// We allocated a block of memory exclusively for this resource.
    Allocation { allocation: gpu_allocator::vulkan::Allocation },
    /// The memory for this resource was imported or exported from/to an external handle.
    DeviceMemory { device_memory: VkDeviceMemory },
    /// No memory is allocated for this resource.
    ///
    /// Currently, this is only used in zero-byte [`Buffer`s](crate::Buffer)
    /// so that we have something to put in the `allocation` field.
    None,
}

/// Chooses a swap chain surface format among a list of supported formats.
///
/// TODO there's only one supported format right now...
fn get_preferred_swapchain_surface_format(surface_formats: &[VkSurfaceFormatKHR]) -> VkSurfaceFormatKHR {
    surface_formats
        .iter()
        .find_map(|&fmt| {
            if fmt.format == VK_FORMAT_B8G8R8A8_SRGB && fmt.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR {
                Some(fmt)
            } else {
                None
            }
        })
        .expect("no suitable surface format available")
}

/// Creates a `Device` compatible with the specified presentation surface.
///
/// # Safety
///
/// `present_surface` must be a valid surface handle, or `None`
unsafe fn create_device_with_surface(present_surface: Option<VkSurfaceKHR>) -> Result<Device, DeviceCreateError> {
    let device = Device::with_surface(present_surface)?;
    Ok(device)
}

/// Creates a `Device`. A physical device is chosen automatically.
fn create_device() -> Result<Device, DeviceCreateError> {
    unsafe { create_device_with_surface(None) }
}

struct PhysicalDeviceAndProperties {
    physical_device: VkPhysicalDevice,
    properties: VkPhysicalDeviceProperties,
    //features: vk::PhysicalDeviceFeatures,
}

/// Chooses a present mode among a list of supported modes.
pub(super) fn get_preferred_present_mode(available_present_modes: &[VkPresentModeKHR]) -> VkPresentModeKHR {
    if available_present_modes.contains(&VK_PRESENT_MODE_MAILBOX_KHR) {
        VK_PRESENT_MODE_MAILBOX_KHR
    } else if available_present_modes.contains(&VK_PRESENT_MODE_IMMEDIATE_KHR) {
        VK_PRESENT_MODE_IMMEDIATE_KHR
    } else {
        VK_PRESENT_MODE_FIFO_KHR
    }
}

/// Computes the preferred swap extent.
pub(super) fn get_preferred_swap_extent(
    framebuffer_size: (u32, u32),
    capabilities: &VkSurfaceCapabilitiesKHR,
) -> VkExtent2D {
    if capabilities.currentExtent.width != u32::MAX {
        capabilities.currentExtent
    } else {
        VkExtent2D {
            width: framebuffer_size.0.clamp(capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
            height: framebuffer_size
                .1
                .clamp(capabilities.minImageExtent.height, capabilities.maxImageExtent.height),
        }
    }
}

unsafe fn select_physical_device(instance: &Instance) -> PhysicalDeviceAndProperties {
    let physical_devices = {
        let mut count = 0;
        instance.fns.EnumeratePhysicalDevices(instance.instance, &mut count, ptr::null_mut()).check();
        let mut devices = Vec::with_capacity(count as usize);
        instance.fns.EnumeratePhysicalDevices(instance.instance, &mut count, devices.as_mut_ptr()).check();
        devices.set_len(count as usize);
        devices
    };
    if physical_devices.is_empty() {
        panic!("no device with vulkan support");
    }
    let mut selected_phy = None;
    let mut selected_phy_properties = Default::default();
    //let mut selected_phy_features = Default::default();
    for phy in physical_devices {
        let props = {
            let mut props = MaybeUninit::uninit();
            instance.fns.GetPhysicalDeviceProperties(phy, props.as_mut_ptr());
            props.assume_init()
        };
        let _features = {
            let mut features = MaybeUninit::uninit();
            instance.fns.GetPhysicalDeviceFeatures(phy, features.as_mut_ptr());
            features.assume_init()
        };
        if props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU {
            selected_phy = Some(phy);
            selected_phy_properties = props;
            //selected_phy_features = features;
        }
    }
    // TODO implement fallbacks
    PhysicalDeviceAndProperties {
        physical_device: selected_phy.expect("no suitable physical device"),
        properties: selected_phy_properties,
        //features: selected_phy_features,
    }
}

// TODO nuke this
unsafe fn find_queue_family(
    instance: &Instance,
    phy: VkPhysicalDevice,
    queue_families: &[VkQueueFamilyProperties],
    flags: VkQueueFlags,
    present_surface: Option<VkSurfaceKHR>,
) -> u32 {
    let mut best_queue_family: Option<u32> = None;
    let mut best_flags = 0u32;
    let mut index = 0u32;
    for queue_family in queue_families {
        if queue_family.queueFlags & flags != 0 {
            // matches the intended usage
            // if present_surface != nullptr, check that it also supports presentation
            // to the given surface
            if let Some(surface) = present_surface {
                if !instance.khr_surface.GetPhysicalDeviceSurfaceSupportKHR(phy, index, surface).unwrap() {
                    // does not support presentation, skip it
                    continue;
                }
            }
            if let Some(ref mut i) = best_queue_family {
                // there was already a queue for the specified usage,
                // change it only if it is more specialized.
                // to determine if it is more specialized, count number of bits (XXX sketchy?)
                if queue_family.queueFlags.count_ones() < best_flags.count_ones() {
                    *i = index;
                    best_flags = queue_family.queueFlags;
                }
            } else {
                best_queue_family = Some(index);
                best_flags = queue_family.queueFlags;
            }
        }
        index += 1;
    }
    best_queue_family.expect("could not find a compatible queue")
}

static DEVICE_EXTENSIONS: [&CStr; 12] = [
    c"VK_KHR_swapchain",
    c"VK_KHR_maintenance5",
    c"VK_KHR_push_descriptor",
    c"VK_EXT_extended_dynamic_state3",
    c"VK_EXT_mesh_shader",
    c"VK_EXT_conservative_rasterization",
    c"VK_EXT_fragment_shader_interlock",
    c"VK_EXT_shader_image_atomic_int64",
    c"VK_KHR_calibrated_timestamps",
    c"VK_EXT_descriptor_heap",
    c"VK_KHR_shader_untyped_pointers",
    c"VK_EXT_mutable_descriptor_type",
];

////////////////////////////////////////////////////////////////////////////////////////////////
// INITIALIZATION
////////////////////////////////////////////////////////////////////////////////////////////////

impl Device {
    /// Returns the global device instance.
    pub fn instance() -> &'static Device {
        static DEVICE: LazyLock<&'static Device> = LazyLock::new(|| {
            unsafe {
                // SAFETY: this is safe when passing `None`.
                // Technically we must choose a device that supports presentation for
                // the user's surfaces. But we don't have a surface here and I don't want to complicate
                // the API by requiring the user to somehow pass a surface handle before
                // the device singleton is initialized.
                // It should be easy to filter out devices that don't support presentation anyway.
                // At worst, we can probably use OS APIs for that.
                // Note that no such bullshit is necessary with D3D12.
                let device = Device::with_surface(None).expect("failed to create the GPU device");
                Box::leak(Box::new(device))
            }
        });
        &*DEVICE
    }

    fn find_compatible_memory_type_internal(
        &self,
        memory_type_bits: u32,
        memory_properties: VkMemoryPropertyFlags,
    ) -> Option<u32> {
        for i in 0..self.thread_safe.physical_device_memory_properties.memoryTypeCount {
            if memory_type_bits & (1 << i) != 0
                && (self.thread_safe.physical_device_memory_properties.memoryTypes[i as usize].propertyFlags
                    & memory_properties)
                    != 0
            {
                return Some(i);
            }
        }
        None
    }

    /// Returns the index of the first memory type compatible with the specified memory type bitmask and additional memory property flags.
    pub(crate) fn find_compatible_memory_type(
        &self,
        memory_type_bits: u32,
        required_memory_properties: VkMemoryPropertyFlags,
        preferred_memory_properties: VkMemoryPropertyFlags,
    ) -> Option<u32> {
        // first, try required+preferred, otherwise fallback on just required
        self.find_compatible_memory_type_internal(
            memory_type_bits,
            required_memory_properties | preferred_memory_properties,
        )
        .or_else(|| self.find_compatible_memory_type_internal(memory_type_bits, required_memory_properties))
    }

    /// Creates a new `Device` from an existing vulkan device.
    ///
    /// The device should have been created with at least one graphics queue.
    ///
    /// # Arguments
    ///
    /// * `physical_device` - the physical device that the device was created on
    /// * `device` - the vulkan device handle
    /// * `graphics_queue_family_index` - queue family index of the main graphics queue
    unsafe fn from_existing(
        physical_device: VkPhysicalDevice,
        device: VkDevice,
        graphics_queue_family_index: u32,
    ) -> Result<Device, DeviceCreateError> {
        let entry = get_vulkan_entry();
        let instance = Instance::get();
        let dd = Vulkan_1_4_DeviceDispatch::load_with(|proc| instance.fns.GetDeviceProcAddr(device, proc.as_ptr()));
        let queue = {
            let mut queue = VkQueue::null();
            dd.GetDeviceQueue(device, graphics_queue_family_index, 0, &mut queue);
            queue
        };
        let timeline = {
            let timeline_create_info =
                VkSemaphoreTypeCreateInfo { semaphoreType: VK_SEMAPHORE_TYPE_TIMELINE, initialValue: 0, .. };
            let semaphore_create_info =
                VkSemaphoreCreateInfo { pNext: &timeline_create_info as *const _ as *const c_void, .. };
            dd.CreateSemaphore(device, &semaphore_create_info, ptr::null()).unwrap()
        };
        let mut allocator = {
            let ash_instance = ash::Instance::load_with(
                |name| mem::transmute(entry.GetInstanceProcAddr(instance.instance, name.as_ptr())),
                vk::Instance::from_raw(instance.instance.0 as u64),
            );
            let ash_device = ash::Device::load(ash_instance.fp_v1_0(), ash::vk::Device::from_raw(device.0 as u64));
            let allocator_create_desc = gpu_allocator::vulkan::AllocatorCreateDesc {
                physical_device: ash::vk::PhysicalDevice::from_raw(physical_device.0 as u64),
                debug_settings: Default::default(),
                device: ash_device,
                instance: ash_instance,
                buffer_device_address: true,
                allocation_sizes: Default::default(),
            };
            gpu_allocator::vulkan::Allocator::new(&allocator_create_desc).expect("failed to create GPU allocator")
        };
        let mut descriptor_heap_properties: VkPhysicalDeviceDescriptorHeapPropertiesEXT = unsafe { mem::zeroed() };
        descriptor_heap_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_HEAP_PROPERTIES_EXT;
        let mut physical_device_id_properties: VkPhysicalDeviceIDProperties = unsafe { mem::zeroed() };
        physical_device_id_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
        let mut physical_device_properties: VkPhysicalDeviceProperties2 = unsafe { mem::zeroed() };
        physical_device_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        physical_device_id_properties.pNext = &mut descriptor_heap_properties as *mut _ as *mut c_void;
        physical_device_properties.pNext = &mut physical_device_id_properties as *mut _ as *mut c_void;
        instance.get_physical_device_properties2(physical_device, &mut physical_device_properties);
        // Extensions
        let load_fn = |proc| instance.fns.GetDeviceProcAddr(device, proc.as_ptr());
        let khr_swapchain = khr_swapchain::DeviceDispatch::load_with(load_fn);
        let khr_push_descriptor = khr_push_descriptor::DeviceDispatch::load_with(load_fn);
        let khr_calibrated_timestamps = khr_calibrated_timestamps::DeviceDispatch::load_with(load_fn);
        let ext_extended_dynamic_state3 = ext_extended_dynamic_state3::DeviceDispatch::load_with(load_fn);
        let ext_mesh_shader = ext_mesh_shader::DeviceDispatch::load_with(load_fn);
        let ext_debug_utils = ext_debug_utils::DeviceDispatch::load_with(load_fn);
        let platform_extensions = PlatformExtensions::load(entry, instance, &device);
        let descriptor_heap_device = ext_descriptor_heap::DeviceDispatch::load_with(load_fn);
        let descriptor_heap_instance = ext_descriptor_heap::InstanceDispatch::load_with(|proc| {
            entry.GetInstanceProcAddr(instance.instance, proc.as_ptr())
        });
        let descriptor_heaps = DescriptorHeaps::new(&mut allocator, &device, &descriptor_heap_properties);
        let memory_properties = {
            let mut memory_properties = MaybeUninit::uninit();
            instance.fns.GetPhysicalDeviceMemoryProperties(physical_device, memory_properties.as_mut_ptr());
            memory_properties.assume_init()
        };
        // ------ info dump ------
        let device_name = CStr::from_ptr(physical_device_properties.properties.deviceName.as_ptr()).to_string_lossy();
        info!("gpu: using device {device_name}",);
        info!(
            "    deviceType: {:?}  deviceID: {:04x}  vendorID: {:04x}",
            physical_device_properties.properties.deviceType,
            physical_device_properties.properties.deviceID,
            physical_device_properties.properties.vendorID
        );
        info!("    pipelineCacheUUID: {:02x?}", physical_device_properties.properties.pipelineCacheUUID);
        info!(
            "    apiVersion: {}.{}.{}   driverVersion: {}",
            vk::api_version_major(physical_device_properties.properties.apiVersion),
            vk::api_version_minor(physical_device_properties.properties.apiVersion),
            vk::api_version_patch(physical_device_properties.properties.apiVersion),
            physical_device_properties.properties.driverVersion
        );
        if physical_device_id_properties.deviceLUIDValid == vk::TRUE {
            info!("    deviceLUID: {:02x?}", physical_device_id_properties.deviceLUID);
        }
        info!("    Timestamp information:");
        info!("        timestampPeriod: {}", physical_device_properties.properties.limits.timestampPeriod);

        Ok(Device {
            //raw: device,
            vk: dd,
            vkd: device,
            ext: DeviceExtensions {
                swapchain: khr_swapchain,
                push_descriptor: khr_push_descriptor,
                calibrated_timestamps: khr_calibrated_timestamps,
                mesh_shader: ext_mesh_shader,
                _ext_extended_dynamic_state3: ext_extended_dynamic_state3,
                debug_utils: ext_debug_utils,
                descriptor_heap: descriptor_heap_device,
                descriptor_heap_instance,
            },
            platform_extensions,
            thread_safe: DeviceThreadSafeState {
                physical_device_memory_properties,
                physical_device_id_properties,
                descriptor_heap_properties,
                physical_device_properties: physical_device_properties.properties,
                frame_timeline: timeline,
                physical_device,
            },
            submission_state: Mutex::new(DeviceSubmissionState { queue, active_submissions: VecDeque::new() }),
            queue_family: graphics_queue_family_index,
            allocator: Mutex::new(allocator),
            //descriptor_indices: Mutex::new(DeviceDescriptorIndexTable {
            //    resource: Default::default(),
            //    sampler: Default::default(),
            //}),
            //descriptor_table,
            sampler_cache: Mutex::new(Default::default()),
            frame_index: AtomicU64::new(1),
            semaphores: Default::default(),
            deletion_queue: Mutex::new(Vec::new()),
            descriptor_heaps,
        })
    }

    /// Creates a new `Device`, automatically choosing a suitable physical device.
    pub fn new() -> Result<Device, DeviceCreateError> {
        unsafe { Self::with_surface(None) }
    }

    /// Returns the list of supported swapchain formats for the given surface.
    pub unsafe fn get_surface_formats(&self, surface: vk::SurfaceKHR) -> Vec<VkSurfaceFormatKHR> {
        vk_khr_surface().get_physical_device_surface_formats(self.thread_safe.physical_device, surface).unwrap()
    }

    /// Returns one supported surface format. Use if you don't care about the format of your swapchain.
    pub unsafe fn get_preferred_surface_format(&self, surface: vk::SurfaceKHR) -> VkSurfaceFormatKHR {
        let surface_formats = self.get_surface_formats(surface);
        get_preferred_swapchain_surface_format(&surface_formats)
    }

    /// Creates a new `Device` that can render to the specified `present_surface` if one is specified.
    ///
    /// Also creates queues as requested.
    pub unsafe fn with_surface(present_surface: Option<VkSurfaceKHR>) -> Result<Device, DeviceCreateError> {
        let instance = Instance::get();
        let phy = select_physical_device(instance);
        let queue_family_properties = {
            let mut count = 0;
            instance.fns.GetPhysicalDeviceQueueFamilyProperties(phy.physical_device, &mut count, ptr::null_mut());
            let mut qfps = Vec::with_capacity(count as usize);
            instance.fns.GetPhysicalDeviceQueueFamilyProperties(phy.physical_device, &mut count, qfps.as_mut_ptr());
            qfps.set_len(count as usize);
            qfps
        };
        let graphics_queue_family = find_queue_family(
            instance,
            phy.physical_device,
            &queue_family_properties,
            VK_QUEUE_GRAPHICS_BIT,
            present_surface,
        );
        let queue_priorities = [1.0f32];
        let device_queue_create_infos = &[VkDeviceQueueCreateInfo {
            flags: 0,
            queueFamilyIndex: graphics_queue_family,
            queueCount: 1,
            pQueuePriorities: queue_priorities.as_ptr(),
            ..
        }];

        // ------ BEGIN SHOPPING LIST ------
        // TODO: this code should probably be generated by a JSON profile.
        //       on occasion, try the vulkan profiles library
        let mut shader_untyped_pointers = VkPhysicalDeviceShaderUntypedPointersFeaturesKHR {
            pNext: ptr::null_mut(),
            shaderUntypedPointers: VK_TRUE,
            ..
        };
        let mut descriptor_heap_features = VkPhysicalDeviceDescriptorHeapFeaturesEXT {
            pNext: &mut shader_untyped_pointers as *mut _ as *mut c_void,
            descriptorHeap: VK_TRUE, // we use descriptor heaps exclusively
            descriptorHeapCaptureReplay: VK_FALSE,
            ..
        };
        let mut fragment_shader_interlock_features = VkPhysicalDeviceFragmentShaderInterlockFeaturesEXT {
            pNext: &mut descriptor_heap_features as *mut _ as *mut c_void,
            fragmentShaderPixelInterlock: VK_TRUE, // nice-to-have for experimentation
            ..
        };
        let mut maintenance5_features = VkPhysicalDeviceMaintenance5FeaturesKHR {
            pNext: &mut fragment_shader_interlock_features as *mut _ as *mut c_void,
            maintenance5: VK_TRUE,
            ..Default::default()
        };
        let mut mutable_descriptor_type_features = VkPhysicalDeviceMutableDescriptorTypeFeaturesEXT {
            pNext: &mut maintenance5_features as *mut _ as *mut c_void,
            mutableDescriptorType: VK_TRUE, // TODO not sure this is needed anymore with descriptor_heap
            ..Default::default()
        };
        let mut mesh_shader_features = VkPhysicalDeviceMeshShaderFeaturesEXT {
            pNext: &mut mutable_descriptor_type_features as *mut _ as *mut c_void,
            taskShader: VK_TRUE, // it's the future
            meshShader: VK_TRUE,
            ..Default::default()
        };
        // don't bother with static state in pipelines
        let mut ext_dynamic_state = VkPhysicalDeviceExtendedDynamicState3FeaturesEXT {
            pNext: &mut mesh_shader_features as *mut _ as *mut c_void,
            extendedDynamicState3TessellationDomainOrigin: VK_TRUE,
            extendedDynamicState3DepthClampEnable: VK_TRUE,
            extendedDynamicState3PolygonMode: VK_TRUE,
            extendedDynamicState3RasterizationSamples: VK_TRUE,
            extendedDynamicState3SampleMask: VK_TRUE,
            extendedDynamicState3AlphaToCoverageEnable: VK_TRUE,
            extendedDynamicState3AlphaToOneEnable: VK_TRUE,
            extendedDynamicState3LogicOpEnable: VK_TRUE,
            extendedDynamicState3ColorBlendEnable: VK_TRUE,
            extendedDynamicState3ColorBlendEquation: VK_TRUE,
            extendedDynamicState3ColorWriteMask: VK_TRUE,
            extendedDynamicState3RasterizationStream: VK_TRUE,
            extendedDynamicState3ConservativeRasterizationMode: VK_TRUE,
            extendedDynamicState3ExtraPrimitiveOverestimationSize: VK_TRUE,
            extendedDynamicState3DepthClipEnable: VK_TRUE,
            extendedDynamicState3SampleLocationsEnable: VK_TRUE,
            extendedDynamicState3ColorBlendAdvanced: VK_TRUE,
            extendedDynamicState3ProvokingVertexMode: VK_TRUE,
            extendedDynamicState3LineRasterizationMode: VK_TRUE,
            extendedDynamicState3LineStippleEnable: VK_TRUE,
            extendedDynamicState3DepthClipNegativeOneToOne: VK_TRUE,
            extendedDynamicState3ViewportWScalingEnable: VK_TRUE,
            extendedDynamicState3ViewportSwizzle: VK_TRUE,
            extendedDynamicState3CoverageToColorEnable: VK_TRUE,
            extendedDynamicState3CoverageToColorLocation: VK_TRUE,
            extendedDynamicState3CoverageModulationMode: VK_TRUE,
            extendedDynamicState3CoverageModulationTableEnable: VK_TRUE,
            extendedDynamicState3CoverageModulationTable: VK_TRUE,
            extendedDynamicState3CoverageReductionMode: VK_TRUE,
            extendedDynamicState3RepresentativeFragmentTestEnable: VK_TRUE,
            extendedDynamicState3ShadingRateImageEnable: VK_TRUE,
            ..
        };
        let mut vk13_features = VkPhysicalDeviceVulkan13Features {
            pNext: &mut ext_dynamic_state as *mut _ as *mut c_void,
            synchronization2: VK_TRUE,
            dynamicRendering: VK_TRUE, // we use dynamic rendering exclusively
            // we expose a constant subgroup size of 32 to simplify the implementation of algorithms that depend on subgroups
            subgroupSizeControl: VK_TRUE,
            ..Default::default()
        };
        let mut vk12_features = VkPhysicalDeviceVulkan12Features {
            pNext: &mut vk13_features as *mut _ as *mut c_void,
            descriptorIndexing: VK_TRUE,
            descriptorBindingVariableDescriptorCount: VK_TRUE,
            descriptorBindingPartiallyBound: VK_TRUE,
            descriptorBindingUpdateUnusedWhilePending: VK_TRUE,
            shaderUniformBufferArrayNonUniformIndexing: VK_TRUE,
            shaderStorageBufferArrayNonUniformIndexing: VK_TRUE,
            shaderSampledImageArrayNonUniformIndexing: VK_TRUE,
            shaderStorageImageArrayNonUniformIndexing: VK_TRUE,
            runtimeDescriptorArray: VK_TRUE,
            bufferDeviceAddress: VK_TRUE,
            bufferDeviceAddressCaptureReplay: VK_TRUE,
            timelineSemaphore: VK_TRUE,
            storageBuffer8BitAccess: VK_TRUE,
            storagePushConstant8: VK_TRUE,
            shaderInt8: VK_TRUE,
            scalarBlockLayout: VK_TRUE,
            hostQueryReset: VK_TRUE,
            ..Default::default()
        };
        let mut vk11_features = VkPhysicalDeviceVulkan11Features {
            pNext: &mut vk12_features as *mut _ as *mut c_void,
            shaderDrawParameters: VK_TRUE,
            storageBuffer16BitAccess: VK_TRUE,
            storagePushConstant16: VK_TRUE,
            ..Default::default()
        };
        let mut features2 = VkPhysicalDeviceFeatures2 {
            pNext: &mut vk11_features as *mut _ as *mut c_void,
            features: VkPhysicalDeviceFeatures {
                tessellationShader: VK_TRUE,
                fillModeNonSolid: VK_TRUE,
                samplerAnisotropy: VK_TRUE,
                shaderInt16: VK_TRUE,
                shaderInt64: VK_TRUE,
                shaderStorageImageExtendedFormats: VK_TRUE,
                fragmentStoresAndAtomics: VK_TRUE,
                depthClamp: VK_TRUE,
                multiDrawIndirect: VK_TRUE,
                independentBlend: VK_TRUE,
                ..Default::default()
            },
            ..Default::default()
        };
        let device_extensions = DEVICE_EXTENSIONS.map(|s| s.as_ptr());
        let device_create_info = VkDeviceCreateInfo {
            pNext: &mut features2 as *mut _ as *mut c_void,
            flags: Default::default(),
            queueCreateInfoCount: device_queue_create_infos.len() as u32,
            pQueueCreateInfos: device_queue_create_infos.as_ptr(),
            enabledExtensionCount: device_extensions.len() as u32,
            ppEnabledExtensionNames: device_extensions.as_ptr(),
            pEnabledFeatures: ptr::null(),
            ..
        };
        // ------ END SHOPPING LIST ------

        // ------ Create device ------
        let device = instance
            .fns
            .CreateDevice(phy.physical_device, &device_create_info, ptr::null_mut())
            .expect("failed to create Vulkan device");
        Self::from_existing(phy.physical_device, device, graphics_queue_family)
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////
// MISC
////////////////////////////////////////////////////////////////////////////////////////////////

impl Device {
    /// Returns the underlying raw vulkan device (via `ash::Device`).
    pub fn raw(&self) -> &ash::Device {
        &self.raw
    }

    /// Allocates memory, or panic trying.
    ///
    /// This is used internally for resource creation since we don't expose memory allocation errors to the user.
    pub(crate) fn allocate_memory_or_panic(
        &self,
        create_desc: &AllocationCreateDesc,
    ) -> gpu_allocator::vulkan::Allocation {
        self.allocator.lock().unwrap().allocate(create_desc).expect("failed to allocate device memory")
    }

    pub(crate) fn get_last_completed_frame_index(&self) -> u64 {
        unsafe {
            let mut value = self.vk.GetSemaphoreCounterValue(self.vkd, self.thread_safe.frame_timeline).unwrap();
            if value == u64::MAX {
                // We've likely lost the device.
                panic!("GetSemaphoreCounterValue returned an invalid value, possible device lost");
            }
            value
        }
    }

    /// Schedules a function call.
    ///
    /// The function will be called once the GPU has finished processing commands up to and
    /// including the specified frame index.
    pub fn call_later(&self, after_frame_completed_index: u64, f: impl FnOnce(&Self) + Send + Sync + 'static) {
        if after_frame_completed_index <= self.get_last_completed_frame_index() {
            trace!("GPU: immediate call_later for frame_index={after_frame_completed_index}");
            f(self);
        } else {
            // otherwise move it to the deferred deletion list
            let mut deletion_queue = self.deletion_queue.lock().unwrap();
            let pos = deletion_queue
                .binary_search_by_key(&after_frame_completed_index, |e| e.frame_index)
                .unwrap_or_else(|p| p);
            deletion_queue
                .insert(pos, DeleteQueueEntry { frame_index: after_frame_completed_index, deleter: Some(Box::new(f)) });
        }
    }

    /// Schedules a function (destructor) to be called after the current frame is complete.
    pub(crate) fn delete_after_current_frame(&self, deleter: impl FnOnce(&Self) + Send + Sync + 'static) {
        let current_frame_index = self.frame_index.load(Relaxed);
        self.call_later(current_frame_index, move |device| {
            deleter(device);
        })
    }

    pub(crate) unsafe fn free_memory(&self, allocation: &mut ResourceAllocation) {
        match mem::replace(allocation, ResourceAllocation::External) {
            ResourceAllocation::Allocation { allocation } => {
                self.allocator.lock().unwrap().free(allocation).expect("failed to free memory")
            }
            ResourceAllocation::DeviceMemory { device_memory } => unsafe {
                self.raw.free_memory(device_memory, None);
            },
            ResourceAllocation::None => {
                // nothing to do
            }
            ResourceAllocation::External => {
                unreachable!()
            }
        }
    }

    fn end_frame(&self) -> u64 {
        // Terminate pending command buffers.
        flush();

        // /!\ we are in frame N /!\
        // Fetch and increment frame index, and signal it in on the timeline.
        let frame_index = self.frame_index.fetch_add(1, Relaxed);
        signal(vk::Semaphore::from_raw(self.thread_safe.frame_timeline.0), frame_index);

        // /!\ we are now in frame N+1 /!\
        // Reclaim resources of completed frames.
        let last_completed_frame_index = self.get_last_completed_frame_index();

        // process all completed submissions
        let mut ss = self.submission_state.lock().unwrap();
        loop {
            if ss.active_submissions.is_empty() {
                break;
            }
            if ss.active_submissions.front().unwrap().frame_index > last_completed_frame_index {
                break;
            };
            let _sub = ss.active_submissions.pop_front().unwrap();
        }

        let mut deletion_queue = self.deletion_queue.lock().unwrap();
        // *** This invokes all delayed destructors for resources which are no longer in use by the GPU.
        deletion_queue.retain_mut(|DeleteQueueEntry { frame_index, deleter }| {
            if *frame_index > last_completed_frame_index {
                return true;
            }
            let deleter = deleter.take().unwrap();
            deleter(self);
            false
        });
        frame_index
    }

    /// Creates a new, or returns an existing, binary semaphore that is in the unsignaled state,
    /// or for which we've submitted a wait operation on this queue and that will eventually be unsignaled.
    pub fn get_or_create_semaphore(&self) -> VkSemaphore {
        // Try to recycle one
        if let Some(semaphore) = self.semaphores.lock().unwrap().pop() {
            return semaphore;
        }
        // Otherwise create a new one
        unsafe {
            let create_info = VkSemaphoreCreateInfo { .. };
            self.vk.CreateSemaphore(self.vkd, &create_info, ptr::null()).unwrap()
        }
    }

    /// Recycles a binary semaphore.
    ///
    /// There must be a pending wait operation on the semaphore, or it must be in the unsignaled state.
    pub(crate) unsafe fn recycle_binary_semaphore(&self, binary_semaphore: VkSemaphore) {
        self.semaphores.lock().unwrap().push(binary_semaphore);
    }

    pub(crate) fn register_sampler(&self, info: &SamplerParams) -> SamplerHandle {
        let info_hashable = SamplerParamsHashable::from(*info);
        if let Some(sampler) = self.sampler_cache.lock().unwrap().get(&info_hashable) {
            return SamplerHandle::new(*sampler);
        }
        let create_info = VkSamplerCreateInfo {
            flags: Default::default(),
            magFilter: info.mag_filter,
            minFilter: info.min_filter,
            mipmapMode: info.mipmap_mode,
            addressModeU: info.address_mode_u,
            addressModeV: info.address_mode_v,
            addressModeW: info.address_mode_w,
            mipLodBias: info.mip_lod_bias,
            anisotropyEnable: info.anisotropy_enable.into(),
            maxAnisotropy: info.max_anisotropy,
            compareEnable: info.compare_enable.into(),
            compareOp: info.compare_op.into(),
            minLod: info.min_lod,
            maxLod: info.max_lod,
            borderColor: info.border_color,
            ..Default::default()
        };
        let sampler = self.allocate_sampler_descriptor(&create_info);
        self.sampler_cache.lock().unwrap().insert(info_hashable, sampler);
        SamplerHandle::new(sampler)
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////>
// SHADERS, PIPELINES & LAYOUTS
////////////////////////////////////////////////////////////////////////////////////////////////

macro_rules! make_stage {
    ($sh:expr, $stage_flags:expr, $module:ident, $p_next:ident, $entry_point_name:ident) => {{
        $entry_point_name = CString::new($sh.entry_point).unwrap();
        $module = VkShaderModuleCreateInfo {
            pNext: &$p_next as *const _ as *const c_void,
            codeSize: $sh.code.len() * 4,
            pCode: $sh.code.as_ptr(),
            ..
        };
        VkPipelineShaderStageCreateInfo {
            pNext: &$module as *const _ as *const c_void,
            stage: $stage_flags,
            pName: $entry_point_name.as_ptr(),
            ..
        }
    }};
    () => {};
}

impl Device {
    /// Creates a compute pipeline.
    pub(crate) fn create_compute_pipeline(
        &self,
        create_info: ComputePipelineCreateInfo,
    ) -> Result<ComputePipeline, Error> {
        let mut push_constants_size = create_info.push_constants_size;
        push_constants_size = push_constants_size.max(create_info.shader.push_constants_size);
        let req_subgroup_size = VkPipelineShaderStageRequiredSubgroupSizeCreateInfo {
            requiredSubgroupSize: SUBGROUP_SIZE,
            pNext: ptr::null_mut(),
            ..
        };
        let _module;
        let _entry_point_name;
        let compute_stage =
            make_stage!(create_info.shader, VK_SHADER_STAGE_COMPUTE_BIT, _module, req_subgroup_size, _entry_point_name);
        let pipeline_create_flags =
            VkPipelineCreateFlags2CreateInfoKHR { flags: VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT, .. };
        let cpci = VkComputePipelineCreateInfo {
            pNext: &pipeline_create_flags as *const _ as *const c_void,
            flags: 0,
            stage: compute_stage,
            layout: Default::default(),
            basePipelineIndex: 0,
            ..
        };
        let pipeline = unsafe {
            let mut pipeline = VkPipeline::null();
            vkcheck!(self.vk.CreateComputePipelines(
                self.vkd,
                VkPipelineCache::null(),
                1,
                &cpci,
                ptr::null(),
                &mut pipeline
            ));
            pipeline
        };
        Ok(ComputePipeline { pipeline: vk::Pipeline::from_raw(pipeline.0), reflection: create_info.shader.refl_params })
    }

    /// Creates a graphics pipeline.
    pub(crate) fn create_graphics_pipeline(
        &self,
        create_info: GraphicsPipelineCreateInfo,
    ) -> Result<GraphicsPipeline, Error> {
        let mut push_constants_size = create_info.push_constants_size;
        match create_info.pre_rasterization_shaders {
            PreRasterizationShaders::PrimitiveShading { vertex } => {
                push_constants_size = push_constants_size.max(vertex.push_constants_size);
            }
            PreRasterizationShaders::MeshShading { mesh, task } => {
                push_constants_size = push_constants_size.max(mesh.push_constants_size);
                if let Some(task) = task {
                    push_constants_size = push_constants_size.max(task.push_constants_size);
                }
            }
        }
        push_constants_size = push_constants_size.max(create_info.fragment.shader.push_constants_size);

        // ------ Dynamic states ------
        // TODO: this could be a static property of the pipeline interface
        // FIXME don't allocate there
        let mut dynamic_states = vec![
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR,
            VK_DYNAMIC_STATE_DEPTH_BIAS,
            VK_DYNAMIC_STATE_DEPTH_BIAS_ENABLE,
        ];
        if matches!(create_info.pre_rasterization_shaders, PreRasterizationShaders::PrimitiveShading { .. }) {
            dynamic_states.push(VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY);
        }
        let dynamic_state_create_info = VkPipelineDynamicStateCreateInfo {
            dynamicStateCount: dynamic_states.len() as u32,
            pDynamicStates: dynamic_states.as_ptr(),
            ..
        };

        // ------ Vertex state ------
        let vertex_input = create_info.vertex_input;
        let vertex_attribute_count = vertex_input.attributes.len();
        let vertex_buffer_count = vertex_input.buffers.len();
        let mut vertex_attribute_descriptions = Vec::with_capacity(vertex_attribute_count);
        let mut vertex_binding_descriptions = Vec::with_capacity(vertex_buffer_count);
        for attribute in vertex_input.attributes.iter() {
            vertex_attribute_descriptions.push(VkVertexInputAttributeDescription {
                location: attribute.location,
                binding: attribute.binding,
                // VULKAN-MIGRATION
                format: unsafe { mem::transmute(attribute.format) },
                offset: attribute.offset,
            });
        }
        for desc in vertex_input.buffers.iter() {
            vertex_binding_descriptions.push(VkVertexInputBindingDescription {
                binding: desc.binding,
                stride: desc.stride,
                // VULKAN-MIGRATION
                inputRate: unsafe { mem::transmute(desc.input_rate) },
            });
        }
        let vertex_input_state = VkPipelineVertexInputStateCreateInfo {
            vertexBindingDescriptionCount: vertex_buffer_count as u32,
            pVertexBindingDescriptions: vertex_binding_descriptions.as_ptr(),
            vertexAttributeDescriptionCount: vertex_attribute_count as u32,
            pVertexAttributeDescriptions: vertex_attribute_descriptions.as_ptr(),
            ..
        };
        let input_assembly_state = VkPipelineInputAssemblyStateCreateInfo {
            topology: VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, // ignored, specified dynamically
            primitiveRestartEnable: VK_FALSE,
            ..
        };

        // ------ Shader stages ------
        let req_subgroup_size =
            VkPipelineShaderStageRequiredSubgroupSizeCreateInfo { requiredSubgroupSize: SUBGROUP_SIZE, .. };
        let mut stages = Vec::new();
        // those variables are referenced by VkGraphicsPipelineCreateInfo
        // put them here so that they live at least until vkCreateGraphicsPipelines is called
        let vertex_entry_point;
        let task_entry_point;
        let mesh_entry_point;
        let fragment_entry_point;
        let vertex_module;
        let task_module;
        let mesh_module;
        let fragment_module;
        let mut stage_reflection = vec![];
        match create_info.pre_rasterization_shaders {
            PreRasterizationShaders::PrimitiveShading { vertex } => {
                stages.push(make_stage!(
                    vertex,
                    VK_SHADER_STAGE_VERTEX_BIT,
                    vertex_module,
                    req_subgroup_size,
                    vertex_entry_point
                ));
                stage_reflection.push((ShaderStage::Vertex, vertex.refl_params));
            }
            PreRasterizationShaders::MeshShading { mesh, task } => {
                if let Some(task) = task {
                    stages.push(make_stage!(
                        task,
                        VK_SHADER_STAGE_TASK_BIT_EXT,
                        task_module,
                        req_subgroup_size,
                        task_entry_point
                    ));
                    stage_reflection.push((ShaderStage::Task, task.refl_params));
                }
                stages.push(make_stage!(
                    mesh,
                    VK_SHADER_STAGE_MESH_BIT_EXT,
                    mesh_module,
                    req_subgroup_size,
                    mesh_entry_point
                ));
                stage_reflection.push((ShaderStage::Mesh, mesh.refl_params));
            }
        };
        stages.push(make_stage!(
            create_info.fragment.shader,
            VK_SHADER_STAGE_FRAGMENT_BIT,
            fragment_module,
            req_subgroup_size,
            fragment_entry_point
        ));
        stage_reflection.push((ShaderStage::Fragment, create_info.fragment.shader.refl_params));

        // ------ attachments ------
        let attachment_states: Vec<_> = create_info
            .fragment
            .color_targets
            .iter()
            .map(|target| match target.blend_equation {
                None => VkPipelineColorBlendAttachmentState {
                    blendEnable: VK_FALSE,
                    srcColorBlendFactor: 0,
                    dstColorBlendFactor: 0,
                    colorBlendOp: 0,
                    srcAlphaBlendFactor: 0,
                    dstAlphaBlendFactor: 0,
                    alphaBlendOp: 0,
                    colorWriteMask: unsafe { mem::transmute(target.color_write_mask) },
                },
                Some(blend_equation) => VkPipelineColorBlendAttachmentState {
                    blendEnable: VK_TRUE,
                    // VULKAN-MIGRATION
                    srcColorBlendFactor: unsafe { mem::transmute(blend_equation.src_color_blend_factor) },
                    dstColorBlendFactor: unsafe { mem::transmute(blend_equation.dst_color_blend_factor) },
                    colorBlendOp: unsafe { mem::transmute(blend_equation.color_blend_op) },
                    srcAlphaBlendFactor: unsafe { mem::transmute(blend_equation.src_alpha_blend_factor) },
                    dstAlphaBlendFactor: unsafe { mem::transmute(blend_equation.dst_alpha_blend_factor) },
                    alphaBlendOp: unsafe { mem::transmute(blend_equation.alpha_blend_op) },
                    colorWriteMask: unsafe { mem::transmute(target.color_write_mask) },
                },
            })
            .collect();

        // ------ misc ------
        let conservative_rasterization_state = VkPipelineRasterizationConservativeStateCreateInfoEXT {
            conservativeRasterizationMode: unsafe {
                mem::transmute(create_info.rasterization.conservative_rasterization_mode)
            },
            extraPrimitiveOverestimationSize: 0.0,
            ..
        };
        let rasterization_state = VkPipelineRasterizationStateCreateInfo {
            pNext: &conservative_rasterization_state as *const _ as *const _,
            depthClampEnable: if create_info.rasterization.depth_clamp_enable { VK_TRUE } else { VK_FALSE },
            rasterizerDiscardEnable: 0,
            polygonMode: unsafe { mem::transmute(create_info.rasterization.polygon_mode) },
            cullMode: unsafe { mem::transmute(create_info.rasterization.cull_mode) },
            frontFace: unsafe { mem::transmute(create_info.rasterization.front_face) },
            depthBiasEnable: vk::FALSE,
            depthBiasConstantFactor: 0.0,
            depthBiasClamp: 0.0,
            depthBiasSlopeFactor: 0.0,
            lineWidth: 1.0,
            ..
        };
        let rasterization_samples = match create_info.fragment.multisample.count {
            1 => VK_SAMPLE_COUNT_1_BIT,
            2 => VK_SAMPLE_COUNT_2_BIT,
            4 => VK_SAMPLE_COUNT_4_BIT,
            8 => VK_SAMPLE_COUNT_8_BIT,
            16 => VK_SAMPLE_COUNT_16_BIT,
            32 => VK_SAMPLE_COUNT_32_BIT,
            64 => VK_SAMPLE_COUNT_64_BIT,
            _ => panic!("invalid multisample count {}", create_info.fragment.multisample.count),
        };
        let multisample_state = VkPipelineMultisampleStateCreateInfo {
            rasterizationSamples: rasterization_samples,
            sampleShadingEnable: VK_FALSE,
            minSampleShading: 0.0,
            pSampleMask: ptr::null(),
            alphaToCoverageEnable: create_info.fragment.multisample.alpha_to_coverage_enabled.into(),
            alphaToOneEnable: VK_FALSE,
            ..
        };
        let color_blend_state = VkPipelineColorBlendStateCreateInfo {
            flags: Default::default(),
            logicOpEnable: VK_FALSE,
            logicOp: Default::default(),
            attachmentCount: attachment_states.len() as u32,
            pAttachments: attachment_states.as_ptr(),
            blendConstants: create_info.fragment.blend_constants,
            ..
        };
        let depth_stencil_state = if let Some(ds) = create_info.depth_stencil {
            VkPipelineDepthStencilStateCreateInfo {
                flags: Default::default(),
                depthTestEnable: (ds.depth_compare_op != vk::CompareOp::ALWAYS).into(),
                depthWriteEnable: ds.depth_write_enable.into(),
                // VULKAN-MIGRATION
                depthCompareOp: unsafe { mem::transmute(ds.depth_compare_op) },
                stencilTestEnable: ds.stencil_state.is_enabled().into(),
                front: unsafe { mem::transmute(ds.stencil_state.front.to_vk_stencil_op_state()) },
                back: unsafe { mem::transmute(ds.stencil_state.back.to_vk_stencil_op_state()) },
                depthBoundsTestEnable: VK_FALSE,
                minDepthBounds: 0.0,
                maxDepthBounds: 0.0,
                ..
            }
        } else {
            VkPipelineDepthStencilStateCreateInfo {
                flags: 0,
                depthTestEnable: 0,
                depthWriteEnable: 0,
                depthCompareOp: 0,
                depthBoundsTestEnable: 0,
                stencilTestEnable: 0,
                front: VkStencilOpState {
                    failOp: 0,
                    passOp: 0,
                    depthFailOp: 0,
                    compareOp: 0,
                    compareMask: 0,
                    writeMask: 0,
                    reference: 0,
                },
                back: VkStencilOpState {
                    failOp: 0,
                    passOp: 0,
                    depthFailOp: 0,
                    compareOp: 0,
                    compareMask: 0,
                    writeMask: 0,
                    reference: 0,
                },
                minDepthBounds: 0.0,
                maxDepthBounds: 0.0,
                ..
            }
        };
        let color_attachment_formats =
            create_info.fragment.color_targets.iter().map(|target| target.format).collect::<Vec<_>>();
        let depth_attachment_format = create_info.depth_stencil.map(|ds| ds.format).unwrap_or(vk::Format::UNDEFINED);
        let stencil_attachment_format = if is_depth_and_stencil_format(depth_attachment_format) {
            depth_attachment_format
        } else {
            vk::Format::UNDEFINED
        };
        let rendering_info = VkPipelineRenderingCreateInfo {
            viewMask: 0,
            colorAttachmentCount: color_attachment_formats.len() as u32,
            pColorAttachmentFormats: unsafe { mem::transmute(color_attachment_formats.as_ptr()) },
            depthAttachmentFormat: unsafe { mem::transmute(depth_attachment_format) },
            stencilAttachmentFormat: unsafe { mem::transmute(stencil_attachment_format) },
            ..
        };
        let pipeline_create_flags = VkPipelineCreateFlags2CreateInfoKHR {
            pNext: &rendering_info as *const _ as *const c_void,
            flags: VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT,
            ..
        };
        let pipeline_create_info = VkGraphicsPipelineCreateInfo {
            pNext: &pipeline_create_flags as *const _ as *const _,
            flags: 0,
            stageCount: stages.len() as u32,
            pStages: stages.as_ptr(),
            pVertexInputState: &vertex_input_state,
            pInputAssemblyState: &input_assembly_state,
            pTessellationState: &VkPipelineTessellationStateCreateInfo { patchControlPoints: 0, .. },
            pViewportState: &VkPipelineViewportStateCreateInfo { viewportCount: 1, scissorCount: 1, .. },
            pRasterizationState: &rasterization_state,
            pMultisampleState: &multisample_state,
            pDepthStencilState: &depth_stencil_state,
            pColorBlendState: &color_blend_state,
            pDynamicState: &dynamic_state_create_info,
            layout: Default::default(),
            renderPass: Default::default(),
            subpass: 0,
            basePipelineHandle: Default::default(),
            basePipelineIndex: 0,
            ..
        };

        let pipeline = unsafe {
            let mut pipeline = VkPipeline::null();
            vkcheck!(self.vk.CreateGraphicsPipelines(
                self.vkd,
                VkPipelineCache::null(),
                1,
                &pipeline_create_info,
                ptr::null(),
                &mut pipeline
            ));
            pipeline
        };
        Ok(GraphicsPipeline { pipeline: vk::Pipeline::from_raw(pipeline.0), stage_reflection })
    }
}

/// Waits for the GPU to complete all submitted work.
pub fn wait_idle() {
    unsafe { Device::instance().raw.device_wait_idle().unwrap() }
}

/// Waits for the specified frame.
pub fn wait_for_frame(frame_index: FrameIndex, timeout: Option<Duration>) {
    unsafe {
        let device = Device::instance();
        let wait_info = VkSemaphoreWaitInfo {
            semaphoreCount: 1,
            pSemaphores: &device.thread_safe.frame_timeline,
            pValues: &frame_index,
            ..
        };
        vkcheck!(device.vk.WaitSemaphores(
            device.vkd,
            &wait_info,
            timeout.map(|d| d.as_nanos() as u64).unwrap_or(u64::MAX)
        ));
    }
}

/// Waits for frame `get_frame_index() - nth_prev` to complete.
pub fn wait_for_previous_frame(nth_prev: usize, timeout: Option<Duration>) {
    let frame_index = get_frame_index();
    if frame_index < nth_prev as u64 {
        return;
    }
    let target_frame_index = frame_index - nth_prev as u64;
    wait_for_frame(target_frame_index, timeout);
}

/// Ends the frame.
///
/// The next frame starts automatically.
/// This cleans up expired resources.
///
/// This doesn't need to correspond to actual VSync frames.
/// However, it should be called periodically to free resources that are no longer
/// used by the GPU, and per-frame is a good frequency.
/// Otherwise, tasks scheduled with `call_later` or `delete_later` will never be executed.
///
/// # Safety
///
/// It is undefined behavior to call this function concurrently with functions that modify internal
/// thread-local state, such as [`alloc_temp`](crate::alloc_temp). Those functions are marked as
/// such in their documentation.
///
/// # Return value
///
/// The index of the frame that was just ended.
pub unsafe fn end_frame() -> FrameIndex {
    Device::instance().end_frame()
}

/// Assigns a debug name to an object represented by its raw vulkan handle
///
/// # Arguments
/// * `handle` - the handle to the object
/// * `name` - the name to associate with the object
///
/// # Safety
/// * This function internally calls `vkSetDebugUtilsObjectNameEXT`, which requires that the access
///   to the object is externally synchronized: only the calling thread may access the object
///   while this function is executing.
/// * The handle must be a valid vulkan object handle.
pub unsafe fn set_debug_name_raw<H: vk::Handle>(handle: H, name: impl AsRef<str>) {
    let device = Device::instance();
    let object_name = CString::new(name.as_ref()).unwrap();

    unsafe {
        // SAFETY: TODO
        device
            .ext
            .debug_utils
            .set_debug_utils_object_name(&vk::DebugUtilsObjectNameInfoEXT {
                object_type: H::TYPE,
                object_handle: handle.as_raw(),
                p_object_name: object_name.as_ptr(),
                ..Default::default()
            })
            .unwrap();
    }
}

/// Assigns a debug name to a vulkan object.
///
/// # Safety
///
/// This function internally calls `vkSetDebugUtilsObjectNameEXT`, which requires that the access
/// to the object is externally synchronized: only the calling thread may access the object
/// while this function is executing.
pub unsafe fn set_debug_name<Object: VulkanObject>(object: &Object, name: impl AsRef<str>) {
    unsafe {
        set_debug_name_raw(object.handle(), name);
    }
}

/// Returns the current frame index.
#[inline(never)]
pub fn get_frame_index() -> FrameIndex {
    Device::instance().frame_index.load(Relaxed)
}

/// Returns the index of the most recently completed frame.
#[inline(never)]
pub fn get_last_completed_frame_index() -> FrameIndex {
    Device::instance().get_last_completed_frame_index()
}

/// Returns the `VkPhysicalDeviceProperties` of the physical device used by the global device.
#[inline(never)]
pub fn get_physical_device_properties() -> vk::PhysicalDeviceProperties {
    Device::instance().thread_safe.physical_device_properties
}

/// Returns the name of the physical device used by the global device.
#[inline(never)]
pub fn get_physical_device_name() -> String {
    let properties = get_physical_device_properties();
    let device_name = unsafe { CStr::from_ptr(properties.device_name.as_ptr()) };
    device_name.to_string_lossy().into_owned()
}

/// Returns the device UUID of the physical device.
#[inline(never)]
pub fn get_device_uuid() -> [u8; 16] {
    Device::instance().thread_safe.physical_device_id_properties.device_uuid
}

/// Returns the device LUID of the physical device.
#[inline(never)]
pub fn get_device_luid() -> Option<[u8; 8]> {
    let id_properties = &Device::instance().thread_safe.physical_device_id_properties;
    if id_properties.device_luid_valid == vk::TRUE { Some(id_properties.device_luid) } else { None }
}

/// Returns the timestamp period in nanoseconds.
#[inline(never)]
pub fn get_timestamp_period() -> f32 {
    let properties = get_physical_device_properties();
    properties.limits.timestamp_period
}

#[inline(never)]
pub fn register_sampler(params: &SamplerParams) -> SamplerHandle {
    Device::instance().register_sampler(params)
}

/// Returns a pair (device, system) of calibrated timestamps.
///
/// The first timestamp is a system-specific timestamp value, and the second is a closely corresponding
/// timestamp on the device (the same kind returned by [`write_timestamp`](crate::write_timestamp)).
///
/// # Platform-specific notes
/// - On Windows, the system timestamp is a QueryPerformanceCounter value
/// - On Linux, it's a value in the CLOCK_MONOTONIC time domain returned by clock_gettime(CLOCK_MONOTONIC).
#[inline(never)]
pub fn get_calibrated_timestamp_pair() -> (u64, u64) {
    let device = Device::instance();

    // Create the query pool for timestamps.
    let (timestamps, _) = unsafe {
        device
            .ext
            .calibrated_timestamps
            .get_calibrated_timestamps(&[
                vk::CalibratedTimestampInfoKHR { time_domain: vk::TimeDomainKHR::DEVICE, ..Default::default() },
                #[cfg(windows)]
                vk::CalibratedTimestampInfoKHR {
                    time_domain: vk::TimeDomainKHR::QUERY_PERFORMANCE_COUNTER,
                    ..Default::default()
                },
                #[cfg(unix)]
                vk::CalibratedTimestampInfoKHR {
                    time_domain: vk::TimeDomainKHR::CLOCK_MONOTONIC,
                    ..Default::default()
                },
            ])
            .expect("vkGetCalibratedTimestamps failed")
    };

    let device_timestamp = timestamps[0];
    let system_timestamp = timestamps[1];
    (device_timestamp, system_timestamp)
}
