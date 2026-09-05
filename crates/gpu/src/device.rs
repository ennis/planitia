//! Abstractions over a vulkan device & queues.
//mod bindless;
mod descriptor_heap;

use crate::device::descriptor_heap::DescriptorHeaps;
use crate::instance::vk_khr_surface;
use crate::platform::PlatformExtensions;
use crate::{
    BufferAddressRange, BufferUsage, ComputePipeline, ComputePipelineCreateInfo, DescriptorSetLayout, Error,
    FrameIndex, GraphicsPipeline, GraphicsPipelineCreateInfo, PreRasterizationShaders, Ptr, SUBGROUP_SIZE,
    SamplerParams, SamplerParamsHashable, ShaderReflection, VulkanObject, get_vulkan_entry, get_vulkan_instance,
    is_depth_and_stencil_format, signal,
};
use ash::vk;
use gpu::device::descriptor_heap::SamplerDescriptorHandle;
use gpu::flush;
use gpu_allocator::vulkan::AllocationCreateDesc;
use gpu_types::{SamplerHandle, ShaderStage};
use log::{debug, error, info, trace, warn};
use slotmap::{SlotMap, new_key_type};
use std::collections::{HashMap, VecDeque};
use std::ffi::{CStr, CString, c_void};
use std::ops::Range;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::{Arc, LazyLock, Mutex};
use std::time::Duration;
use std::{fmt, mem, ptr};
use vulkan_headers::vulkan::vulkan as vk2;
use vulkan_headers::vulkan::vulkan::{
    VK_FALSE, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_HEAP_FEATURES_EXT, VK_TRUE,
    VkPhysicalDeviceDescriptorHeapFeaturesEXT, VkPhysicalDeviceShaderUntypedPointersFeaturesKHR,
};
use vulkan_headers::vulkan::vulkan_core::VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT;

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

pub(crate) struct ExtDescriptorHeap {
    pub(crate) cmd_bind_resource_heap: vk2::NonNullPFN_vkCmdBindResourceHeapEXT,
    pub(crate) cmd_bind_sampler_heap: vk2::NonNullPFN_vkCmdBindSamplerHeapEXT,
    pub(crate) cmd_push_data: vk2::NonNullPFN_vkCmdPushDataEXT,
    pub(crate) cmd_get_physical_descriptor_size: vk2::NonNullPFN_vkGetPhysicalDeviceDescriptorSizeEXT,
    pub(crate) write_resource_descriptors: vk2::NonNullPFN_vkWriteResourceDescriptorsEXT,
    pub(crate) write_sampler_descriptors: vk2::NonNullPFN_vkWriteSamplerDescriptorsEXT,
}

impl ExtDescriptorHeap {
    pub(crate) unsafe fn load(entry: &ash::Entry, instance: &ash::Instance) -> Self {
        let get_proc_addr = |name: &CStr| {
            let addr = entry.get_instance_proc_addr(instance.handle(), name.as_ptr());
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
                cmd_get_physical_descriptor_size: mem::transmute(get_proc_addr(
                    c"vkGetPhysicalDeviceDescriptorSizeEXT",
                )),
                write_resource_descriptors: mem::transmute(get_proc_addr(c"vkWriteResourceDescriptorsEXT")),
                write_sampler_descriptors: mem::transmute(get_proc_addr(c"vkWriteSamplerDescriptorsEXT")),
            }
        }
    }
}

/// Device extensions.
pub(crate) struct DeviceExtensions {
    pub(crate) swapchain: ash::khr::swapchain::Device,
    //pub(crate) ext_shader_object: ash::ext::,
    pub(crate) push_descriptor: ash::khr::push_descriptor::Device,
    pub(crate) calibrated_timestamps: ash::khr::calibrated_timestamps::Device,
    pub(crate) mesh_shader: ash::ext::mesh_shader::Device,
    pub(crate) _ext_extended_dynamic_state3: ash::ext::extended_dynamic_state3::Device,
    pub(crate) debug_utils: ash::ext::debug_utils::Device,
    pub(crate) descriptor_heap: ExtDescriptorHeap,
}

/// Device state that is unconditionally safe to access from multiple threads, even though
/// the fields themselves may not be Send or Sync.
pub(crate) struct DeviceThreadSafeState {
    pub(crate) physical_device_memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub(crate) physical_device_id_properties: vk::PhysicalDeviceIDProperties<'static>,
    pub(crate) descriptor_heap_properties: vk2::VkPhysicalDeviceDescriptorHeapPropertiesEXT,
    _physical_device_descriptor_buffer_properties: vk::PhysicalDeviceDescriptorBufferPropertiesEXT<'static>,
    physical_device_properties: vk::PhysicalDeviceProperties,
    /// Timeline used to track completion of frames.
    /// It is incremented and signalled on each frame completion (see `poll`).
    // SAFETY: we're never using this as an externally-synchronized command parameter.
    pub(crate) frame_timeline: vk::Semaphore,
    // SAFETY: we're never using this as an externally-synchronized command parameter.
    pub(crate) physical_device: vk::PhysicalDevice,
}

unsafe impl Send for DeviceThreadSafeState {}
unsafe impl Sync for DeviceThreadSafeState {}

/// Submission-related device state locked during command buffer submission.
pub(crate) struct DeviceSubmissionState {
    pub(crate) queue: vk::Queue,
    /// Sorted by create_ticket, not by order of submission.
    pub(crate) active_submissions: VecDeque<ActiveSubmission>,
}

pub struct Device {
    /// Underlying vulkan device
    pub(crate) raw: ash::Device,
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
    pub(crate) semaphores: Mutex<Vec<vk::Semaphore>>,
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

pub(crate) fn get_vk_sample_count(count: u32) -> vk::SampleCountFlags {
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
    DeviceMemory { device_memory: vk::DeviceMemory },
    /// No memory is allocated for this resource.
    ///
    /// Currently, this is only used in zero-byte [`Buffer`s](crate::Buffer)
    /// so that we have something to put in the `allocation` field.
    None,
}

/// Chooses a swap chain surface format among a list of supported formats.
///
/// TODO there's only one supported format right now...
fn get_preferred_swapchain_surface_format(surface_formats: &[vk::SurfaceFormatKHR]) -> vk::SurfaceFormatKHR {
    surface_formats
        .iter()
        .find_map(|&fmt| {
            if fmt.format == vk::Format::B8G8R8A8_SRGB && fmt.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR {
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
unsafe fn create_device_with_surface(present_surface: Option<vk::SurfaceKHR>) -> Result<Device, DeviceCreateError> {
    let device = Device::with_surface(present_surface)?;
    Ok(device)
}

/// Creates a `Device`. A physical device is chosen automatically.
fn create_device() -> Result<Device, DeviceCreateError> {
    unsafe { create_device_with_surface(None) }
}

struct PhysicalDeviceAndProperties {
    physical_device: vk::PhysicalDevice,
    properties: vk::PhysicalDeviceProperties,
    //features: vk::PhysicalDeviceFeatures,
}

/// Chooses a present mode among a list of supported modes.
pub(super) fn get_preferred_present_mode(available_present_modes: &[vk::PresentModeKHR]) -> vk::PresentModeKHR {
    if available_present_modes.contains(&vk::PresentModeKHR::MAILBOX) {
        vk::PresentModeKHR::MAILBOX
    } else if available_present_modes.contains(&vk::PresentModeKHR::IMMEDIATE) {
        vk::PresentModeKHR::IMMEDIATE
    } else {
        vk::PresentModeKHR::FIFO
    }
}

/// Computes the preferred swap extent.
pub(super) fn get_preferred_swap_extent(
    framebuffer_size: (u32, u32),
    capabilities: &vk::SurfaceCapabilitiesKHR,
) -> vk::Extent2D {
    if capabilities.current_extent.width != u32::MAX {
        capabilities.current_extent
    } else {
        vk::Extent2D {
            width: framebuffer_size.0.clamp(capabilities.min_image_extent.width, capabilities.max_image_extent.width),
            height: framebuffer_size
                .1
                .clamp(capabilities.min_image_extent.height, capabilities.max_image_extent.height),
        }
    }
}

unsafe fn select_physical_device(instance: &ash::Instance) -> PhysicalDeviceAndProperties {
    let physical_devices = instance.enumerate_physical_devices().expect("failed to enumerate physical devices");
    if physical_devices.is_empty() {
        panic!("no device with vulkan support");
    }
    let mut selected_phy = None;
    let mut selected_phy_properties = Default::default();
    //let mut selected_phy_features = Default::default();
    for phy in physical_devices {
        let props = instance.get_physical_device_properties(phy);
        let _features = instance.get_physical_device_features(phy);
        if props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU {
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

unsafe fn find_queue_family(
    phy: vk::PhysicalDevice,
    vk_khr_surface: &ash::khr::surface::Instance,
    queue_families: &[vk::QueueFamilyProperties],
    flags: vk::QueueFlags,
    present_surface: Option<vk::SurfaceKHR>,
) -> u32 {
    let mut best_queue_family: Option<u32> = None;
    let mut best_flags = 0u32;
    let mut index = 0u32;
    for queue_family in queue_families {
        if queue_family.queue_flags.contains(flags) {
            // matches the intended usage
            // if present_surface != nullptr, check that it also supports presentation
            // to the given surface
            if let Some(surface) = present_surface {
                if !vk_khr_surface.get_physical_device_surface_support(phy, index, surface).unwrap() {
                    // does not support presentation, skip it
                    continue;
                }
            }
            if let Some(ref mut i) = best_queue_family {
                // there was already a queue for the specified usage,
                // change it only if it is more specialized.
                // to determine if it is more specialized, count number of bits (XXX sketchy?)
                if queue_family.queue_flags.as_raw().count_ones() < best_flags.count_ones() {
                    *i = index;
                    best_flags = queue_family.queue_flags.as_raw();
                }
            } else {
                best_queue_family = Some(index);
                best_flags = queue_family.queue_flags.as_raw();
            }
        }
        index += 1;
    }
    best_queue_family.expect("could not find a compatible queue")
}

const DEVICE_EXTENSIONS: &[&str] = &[
    "VK_KHR_swapchain",
    "VK_KHR_maintenance5",
    "VK_KHR_push_descriptor",
    "VK_EXT_extended_dynamic_state3",
    "VK_EXT_mesh_shader",
    "VK_EXT_conservative_rasterization",
    "VK_EXT_fragment_shader_interlock",
    "VK_EXT_shader_image_atomic_int64",
    "VK_KHR_calibrated_timestamps",
    "VK_EXT_descriptor_heap",
    "VK_KHR_shader_untyped_pointers",
    "VK_EXT_mutable_descriptor_type",
];

////////////////////////////////////////////////////////////////////////////////////////////////
// INITIALIZATION
////////////////////////////////////////////////////////////////////////////////////////////////

impl Device {
    /// Returns the global device instance.
    #[inline(never)]
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
        memory_properties: vk::MemoryPropertyFlags,
    ) -> Option<u32> {
        for i in 0..self.thread_safe.physical_device_memory_properties.memory_type_count {
            if memory_type_bits & (1 << i) != 0
                && self.thread_safe.physical_device_memory_properties.memory_types[i as usize]
                    .property_flags
                    .contains(memory_properties)
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
        required_memory_properties: vk::MemoryPropertyFlags,
        preferred_memory_properties: vk::MemoryPropertyFlags,
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
        physical_device: vk::PhysicalDevice,
        device: vk::Device,
        graphics_queue_family_index: u32,
    ) -> Result<Device, DeviceCreateError> {
        let entry = get_vulkan_entry();
        let instance = get_vulkan_instance();
        let device = ash::Device::load(instance.fp_v1_0(), device);
        let queue = device.get_device_queue(graphics_queue_family_index, 0);
        let timeline = {
            let timeline_create_info = vk::SemaphoreTypeCreateInfo {
                semaphore_type: vk::SemaphoreType::TIMELINE,
                initial_value: 0,
                ..Default::default()
            };
            let semaphore_create_info = vk::SemaphoreCreateInfo {
                p_next: &timeline_create_info as *const _ as *const c_void,
                ..Default::default()
            };
            device.create_semaphore(&semaphore_create_info, None).expect("failed to create timeline semaphore")
        };
        let mut allocator = {
            let allocator_create_desc = gpu_allocator::vulkan::AllocatorCreateDesc {
                physical_device,
                debug_settings: Default::default(),
                device: device.clone(),     // not cheap!
                instance: instance.clone(), // not cheap!
                buffer_device_address: true,
                allocation_sizes: Default::default(),
            };
            gpu_allocator::vulkan::Allocator::new(&allocator_create_desc).expect("failed to create GPU allocator")
        };
        let mut physical_device_descriptor_buffer_properties =
            vk::PhysicalDeviceDescriptorBufferPropertiesEXT::default();
        // TODO: replace this once ash is updated
        let mut descriptor_heap_properties = vk2::VkPhysicalDeviceDescriptorHeapPropertiesEXT {
            sType: vk2::VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_HEAP_PROPERTIES_EXT,
            pNext: &mut physical_device_descriptor_buffer_properties as *mut _ as *mut c_void,
            samplerHeapAlignment: 0,
            resourceHeapAlignment: 0,
            maxSamplerHeapSize: 0,
            maxResourceHeapSize: 0,
            minSamplerHeapReservedRange: 0,
            minSamplerHeapReservedRangeWithEmbedded: 0,
            minResourceHeapReservedRange: 0,
            samplerDescriptorSize: 0,
            imageDescriptorSize: 0,
            bufferDescriptorSize: 0,
            samplerDescriptorAlignment: 0,
            imageDescriptorAlignment: 0,
            bufferDescriptorAlignment: 0,
            maxPushDataSize: 0,
            imageCaptureReplayOpaqueDataSize: 0,
            maxDescriptorHeapEmbeddedSamplers: 0,
            samplerYcbcrConversionCount: 0,
            sparseDescriptorHeaps: 0,
            protectedDescriptorHeaps: 0,
        };
        let mut physical_device_id_properties = vk::PhysicalDeviceIDProperties {
            p_next: &mut descriptor_heap_properties as *mut _ as *mut c_void,
            ..Default::default()
        };
        let mut physical_device_properties = vk::PhysicalDeviceProperties2 {
            p_next: &mut physical_device_id_properties as *mut _ as *mut c_void,
            ..Default::default()
        };
        instance.get_physical_device_properties2(physical_device, &mut physical_device_properties);

        // Extensions
        let khr_swapchain = ash::khr::swapchain::Device::new(instance, &device);
        let khr_push_descriptor = ash::khr::push_descriptor::Device::new(instance, &device);
        let khr_calibrated_timestamps = ash::khr::calibrated_timestamps::Device::new(instance, &device);
        let ext_extended_dynamic_state3 = ash::ext::extended_dynamic_state3::Device::new(instance, &device);
        let ext_mesh_shader = ash::ext::mesh_shader::Device::new(instance, &device);
        let physical_device_memory_properties = instance.get_physical_device_memory_properties(physical_device);
        let ext_debug_utils = ash::ext::debug_utils::Device::new(instance, &device);
        let platform_extensions = PlatformExtensions::load(entry, instance, &device);
        let ext_descriptor_heap = ExtDescriptorHeap::load(entry, instance);
        let descriptor_heaps = DescriptorHeaps::new(&mut allocator, &device, &descriptor_heap_properties);

        // ------ info dump ------
        let device_name = CStr::from_ptr(physical_device_properties.properties.device_name.as_ptr()).to_string_lossy();
        info!("gpu: using device {device_name}",);
        info!(
            "    deviceType: {:?}  deviceID: {:04x}  vendorID: {:04x}",
            physical_device_properties.properties.device_type,
            physical_device_properties.properties.device_id,
            physical_device_properties.properties.vendor_id
        );
        info!("    pipelineCacheUUID: {:02x?}", physical_device_properties.properties.pipeline_cache_uuid);
        info!(
            "    apiVersion: {}.{}.{}   driverVersion: {}",
            vk::api_version_major(physical_device_properties.properties.api_version),
            vk::api_version_minor(physical_device_properties.properties.api_version),
            vk::api_version_patch(physical_device_properties.properties.api_version),
            physical_device_properties.properties.driver_version
        );
        if physical_device_id_properties.device_luid_valid == vk::TRUE {
            info!("    deviceLUID: {:02x?}", physical_device_id_properties.device_luid);
        }
        info!("    Timestamp information:");
        info!("        timestampPeriod: {}", physical_device_properties.properties.limits.timestamp_period);

        Ok(Device {
            raw: device,
            ext: DeviceExtensions {
                swapchain: khr_swapchain,
                push_descriptor: khr_push_descriptor,
                calibrated_timestamps: khr_calibrated_timestamps,
                mesh_shader: ext_mesh_shader,
                _ext_extended_dynamic_state3: ext_extended_dynamic_state3,
                debug_utils: ext_debug_utils,
                descriptor_heap: ext_descriptor_heap,
            },
            platform_extensions,
            thread_safe: DeviceThreadSafeState {
                physical_device_memory_properties,
                physical_device_id_properties,
                descriptor_heap_properties,
                _physical_device_descriptor_buffer_properties: physical_device_descriptor_buffer_properties,
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
    pub unsafe fn get_surface_formats(&self, surface: vk::SurfaceKHR) -> Vec<vk::SurfaceFormatKHR> {
        vk_khr_surface().get_physical_device_surface_formats(self.thread_safe.physical_device, surface).unwrap()
    }

    /// Returns one supported surface format. Use if you don't care about the format of your swapchain.
    pub unsafe fn get_preferred_surface_format(&self, surface: vk::SurfaceKHR) -> vk::SurfaceFormatKHR {
        let surface_formats = self.get_surface_formats(surface);
        get_preferred_swapchain_surface_format(&surface_formats)
    }

    /// Creates a new `Device` that can render to the specified `present_surface` if one is specified.
    ///
    /// Also creates queues as requested.
    pub unsafe fn with_surface(present_surface: Option<vk::SurfaceKHR>) -> Result<Device, DeviceCreateError> {
        let instance = get_vulkan_instance();
        let vk_khr_surface = vk_khr_surface();
        let phy = select_physical_device(instance);
        let queue_family_properties = instance.get_physical_device_queue_family_properties(phy.physical_device);
        let graphics_queue_family = find_queue_family(
            phy.physical_device,
            &vk_khr_surface,
            &queue_family_properties,
            vk::QueueFlags::GRAPHICS,
            present_surface,
        );
        let queue_priorities = [1.0f32];
        let device_queue_create_infos = &[vk::DeviceQueueCreateInfo {
            flags: Default::default(),
            queue_family_index: graphics_queue_family,
            queue_count: 1,
            p_queue_priorities: queue_priorities.as_ptr(),
            ..Default::default()
        }];

        // ------ BEGIN SHOPPING LIST ------
        // TODO: this code should probably be generated by a JSON profile.
        //       on occasion, try the vulkan profiles library
        let mut shader_untyped_pointers = VkPhysicalDeviceShaderUntypedPointersFeaturesKHR {
            sType: vk2::VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_UNTYPED_POINTERS_FEATURES_KHR,
            pNext: ptr::null_mut(),
            shaderUntypedPointers: VK_TRUE,
        };
        let mut descriptor_heap_features = VkPhysicalDeviceDescriptorHeapFeaturesEXT {
            sType: VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_HEAP_FEATURES_EXT,
            pNext: &mut shader_untyped_pointers as *mut _ as *mut c_void,
            descriptorHeap: VK_TRUE, // we use descriptor heaps exclusively
            descriptorHeapCaptureReplay: VK_FALSE,
        };
        let mut fragment_shader_interlock_features = vk::PhysicalDeviceFragmentShaderInterlockFeaturesEXT {
            p_next: &mut descriptor_heap_features as *mut _ as *mut c_void,
            fragment_shader_pixel_interlock: vk::TRUE, // nice-to-have for experimentation
            ..Default::default()
        };
        let mut maintenance5_features = vk::PhysicalDeviceMaintenance5FeaturesKHR {
            p_next: &mut fragment_shader_interlock_features as *mut _ as *mut c_void,
            maintenance5: vk::TRUE,
            ..Default::default()
        };
        let mut mutable_descriptor_type_features = vk::PhysicalDeviceMutableDescriptorTypeFeaturesEXT {
            p_next: &mut maintenance5_features as *mut _ as *mut c_void,
            mutable_descriptor_type: vk::TRUE, // TODO not sure this is needed anymore with descriptor_heap
            ..Default::default()
        };
        let mut mesh_shader_features = vk::PhysicalDeviceMeshShaderFeaturesEXT {
            p_next: &mut mutable_descriptor_type_features as *mut _ as *mut c_void,
            task_shader: vk::TRUE, // it's the future
            mesh_shader: vk::TRUE,
            ..Default::default()
        };
        // don't bother with static state in pipelines
        let mut ext_dynamic_state = vk::PhysicalDeviceExtendedDynamicState3FeaturesEXT {
            p_next: &mut mesh_shader_features as *mut _ as *mut c_void,
            extended_dynamic_state3_tessellation_domain_origin: vk::TRUE,
            extended_dynamic_state3_depth_clamp_enable: vk::TRUE,
            extended_dynamic_state3_polygon_mode: vk::TRUE,
            extended_dynamic_state3_rasterization_samples: vk::TRUE,
            extended_dynamic_state3_sample_mask: vk::TRUE,
            extended_dynamic_state3_alpha_to_coverage_enable: vk::TRUE,
            extended_dynamic_state3_alpha_to_one_enable: vk::TRUE,
            extended_dynamic_state3_logic_op_enable: vk::TRUE,
            extended_dynamic_state3_color_blend_enable: vk::TRUE,
            extended_dynamic_state3_color_blend_equation: vk::TRUE,
            extended_dynamic_state3_color_write_mask: vk::TRUE,
            extended_dynamic_state3_rasterization_stream: vk::TRUE,
            extended_dynamic_state3_conservative_rasterization_mode: vk::TRUE,
            extended_dynamic_state3_extra_primitive_overestimation_size: vk::TRUE,
            extended_dynamic_state3_depth_clip_enable: vk::TRUE,
            extended_dynamic_state3_sample_locations_enable: vk::TRUE,
            extended_dynamic_state3_color_blend_advanced: vk::TRUE,
            extended_dynamic_state3_provoking_vertex_mode: vk::TRUE,
            extended_dynamic_state3_line_rasterization_mode: vk::TRUE,
            extended_dynamic_state3_line_stipple_enable: vk::TRUE,
            extended_dynamic_state3_depth_clip_negative_one_to_one: vk::TRUE,
            extended_dynamic_state3_viewport_w_scaling_enable: vk::TRUE,
            extended_dynamic_state3_viewport_swizzle: vk::TRUE,
            extended_dynamic_state3_coverage_to_color_enable: vk::TRUE,
            extended_dynamic_state3_coverage_to_color_location: vk::TRUE,
            extended_dynamic_state3_coverage_modulation_mode: vk::TRUE,
            extended_dynamic_state3_coverage_modulation_table_enable: vk::TRUE,
            extended_dynamic_state3_coverage_modulation_table: vk::TRUE,
            extended_dynamic_state3_coverage_reduction_mode: vk::TRUE,
            extended_dynamic_state3_representative_fragment_test_enable: vk::TRUE,
            extended_dynamic_state3_shading_rate_image_enable: vk::TRUE,
            ..Default::default()
        };
        let mut vk13_features = vk::PhysicalDeviceVulkan13Features {
            p_next: &mut ext_dynamic_state as *mut _ as *mut c_void,
            synchronization2: vk::TRUE,
            dynamic_rendering: vk::TRUE, // we use dynamic rendering exclusively
            // we expose a constant subgroup size of 32 to simplify the implementation of algorithms that depend on subgroups
            subgroup_size_control: vk::TRUE,
            ..Default::default()
        };
        let mut vk12_features = vk::PhysicalDeviceVulkan12Features {
            p_next: &mut vk13_features as *mut _ as *mut c_void,
            descriptor_indexing: vk::TRUE,
            descriptor_binding_variable_descriptor_count: vk::TRUE,
            descriptor_binding_partially_bound: vk::TRUE,
            descriptor_binding_update_unused_while_pending: vk::TRUE,
            shader_uniform_buffer_array_non_uniform_indexing: vk::TRUE,
            shader_storage_buffer_array_non_uniform_indexing: vk::TRUE,
            shader_sampled_image_array_non_uniform_indexing: vk::TRUE,
            shader_storage_image_array_non_uniform_indexing: vk::TRUE,
            runtime_descriptor_array: vk::TRUE,
            buffer_device_address: vk::TRUE,
            buffer_device_address_capture_replay: vk::TRUE,
            timeline_semaphore: vk::TRUE,
            storage_buffer8_bit_access: vk::TRUE,
            storage_push_constant8: vk::TRUE,
            shader_int8: vk::TRUE,
            scalar_block_layout: vk::TRUE,
            host_query_reset: vk::TRUE,
            ..Default::default()
        };
        let mut vk11_features = vk::PhysicalDeviceVulkan11Features {
            p_next: &mut vk12_features as *mut _ as *mut c_void,
            shader_draw_parameters: vk::TRUE,
            storage_buffer16_bit_access: vk::TRUE,
            storage_push_constant16: vk::TRUE,
            ..Default::default()
        };
        let mut features2 = vk::PhysicalDeviceFeatures2 {
            p_next: &mut vk11_features as *mut _ as *mut c_void,
            features: vk::PhysicalDeviceFeatures {
                tessellation_shader: vk::TRUE,
                fill_mode_non_solid: vk::TRUE,
                sampler_anisotropy: vk::TRUE,
                shader_int16: vk::TRUE,
                shader_int64: vk::TRUE,
                shader_storage_image_extended_formats: vk::TRUE,
                fragment_stores_and_atomics: vk::TRUE,
                depth_clamp: vk::TRUE,
                multi_draw_indirect: vk::TRUE,
                independent_blend: vk::TRUE,
                ..Default::default()
            },
            ..Default::default()
        };
        // Convert extension strings into C-strings
        let c_device_extensions: Vec<_> = DEVICE_EXTENSIONS
            .iter()
            .chain(PlatformExtensions::names().iter())
            .map(|&s| CString::new(s).unwrap())
            .collect();
        let device_extensions: Vec<_> = c_device_extensions.iter().map(|s| s.as_ptr()).collect();
        let device_create_info = vk::DeviceCreateInfo {
            p_next: &mut features2 as *mut _ as *mut c_void,
            flags: Default::default(),
            queue_create_info_count: device_queue_create_infos.len() as u32,
            p_queue_create_infos: device_queue_create_infos.as_ptr(),
            enabled_extension_count: device_extensions.len() as u32,
            pp_enabled_extension_names: device_extensions.as_ptr(),
            p_enabled_features: ptr::null(),
            ..Default::default()
        };
        // ------ END SHOPPING LIST ------

        // ------ Create device ------
        let device: ash::Device = instance
            .create_device(phy.physical_device, &device_create_info, None)
            .expect("could not create vulkan device");
        Self::from_existing(phy.physical_device, device.handle(), graphics_queue_family)
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
            let value = self.raw.get_semaphore_counter_value(self.thread_safe.frame_timeline).unwrap();
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
        flush().unwrap();

        // /!\ we are in frame N /!\
        // Fetch and increment frame index, and signal it in on the timeline.
        let frame_index = self.frame_index.fetch_add(1, Relaxed);
        signal(self.thread_safe.frame_timeline, frame_index);

        // /!\ we are now in frame N+1 /!\
        // Reclaim resources of completed frames.
        let last_completed_frame_index = unsafe {
            self.raw
                .get_semaphore_counter_value(self.thread_safe.frame_timeline)
                .expect("get_semaphore_counter_value failed")
        };
        if last_completed_frame_index == u64::MAX {
            // This means "device lost".
            panic!("GetSemaphoreCounterValue returned an invalid value");
        }
        //trace!("GPU: cleaning up to submission {last_completed_submission_index}");

        // process all completed submissions
        //let mut free_timestamp_query_pools = self.free_timestamp_query_pools.lock().unwrap();
        let mut ss = self.submission_state.lock().unwrap();
        loop {
            if ss.active_submissions.is_empty() {
                break;
            }
            if ss.active_submissions.front().unwrap().frame_index > last_completed_frame_index {
                break;
            };
            let _sub = ss.active_submissions.pop_front().unwrap();
            // read timestamp query results
            //unsafe {
            //    if sub.timestamp_query_count > 0 {
            //        let mut timestamp_results = vec![0u64; sub.timestamp_query_count as usize];
            //        self.raw
            //            .get_query_pool_results(
            //                sub.timestamp_query_pool,
            //                0,
            //                &mut timestamp_results[..],
            //                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
            //            )
            //            .expect("vkGetQueryPoolResults failed");
            //        self.raw.reset_query_pool(sub.timestamp_query_pool, 0, sub.timestamp_query_count);
            //        // Invoke callbacks with the results
            //        for (i, cb) in sub.timestamp_callbacks.into_iter().enumerate() {
            //            (cb)(timestamp_results[i]);
            //        }
            //    }
            //}
            //// recycle query pools
            //free_timestamp_query_pools.push(sub.timestamp_query_pool);
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
    pub fn get_or_create_semaphore(&self) -> vk::Semaphore {
        // Try to recycle one
        if let Some(semaphore) = self.semaphores.lock().unwrap().pop() {
            return semaphore;
        }
        // Otherwise create a new one
        unsafe {
            let create_info = vk::SemaphoreCreateInfo { ..Default::default() };
            self.raw.create_semaphore(&create_info, None).unwrap()
        }
    }

    /// Recycles a binary semaphore.
    ///
    /// There must be a pending wait operation on the semaphore, or it must be in the unsignaled state.
    pub(crate) unsafe fn recycle_binary_semaphore(&self, binary_semaphore: vk::Semaphore) {
        self.semaphores.lock().unwrap().push(binary_semaphore);
    }

    pub(crate) fn register_sampler(&self, info: &SamplerParams) -> SamplerHandle {
        let info_hashable = SamplerParamsHashable::from(*info);
        if let Some(sampler) = self.sampler_cache.lock().unwrap().get(&info_hashable) {
            return SamplerHandle::new(*sampler);
        }
        let create_info = vk::SamplerCreateInfo {
            flags: Default::default(),
            mag_filter: info.mag_filter,
            min_filter: info.min_filter,
            mipmap_mode: info.mipmap_mode,
            address_mode_u: info.address_mode_u,
            address_mode_v: info.address_mode_v,
            address_mode_w: info.address_mode_w,
            mip_lod_bias: info.mip_lod_bias,
            anisotropy_enable: info.anisotropy_enable.into(),
            max_anisotropy: info.max_anisotropy,
            compare_enable: info.compare_enable.into(),
            compare_op: info.compare_op.into(),
            min_lod: info.min_lod,
            max_lod: info.max_lod,
            border_color: info.border_color,
            ..Default::default()
        };
        let sampler = self.allocate_sampler_descriptor(&create_info);
        self.sampler_cache.lock().unwrap().insert(info_hashable, sampler);
        SamplerHandle::new(sampler)
    }

    /*pub(crate) fn get_or_create_timestamp_query_pool(&self) -> vk::QueryPool {
        let free = &mut self.free_timestamp_query_pools.lock().unwrap();
        if let Some(pool) = free.pop() {
            pool
        } else {
            let create_info = vk::QueryPoolCreateInfo {
                flags: Default::default(),
                query_type: vk::QueryType::TIMESTAMP,
                query_count: MAX_TIMESTAMP_QUERY_COUNT,
                pipeline_statistics: Default::default(),
                ..Default::default()
            };
            unsafe {
                let pool = self.raw.create_query_pool(&create_info, None).unwrap();
                self.raw.reset_query_pool(pool, 0, MAX_TIMESTAMP_QUERY_COUNT);
                pool
            }
        }
    }*/
}

////////////////////////////////////////////////////////////////////////////////////////////////>
// SHADERS, PIPELINES & LAYOUTS
////////////////////////////////////////////////////////////////////////////////////////////////

/*
struct ShaderModuleGuard<'a> {
    device: &'a Device,
    module: vk::ShaderModule,
}

impl<'a> Drop for ShaderModuleGuard<'a> {
    fn drop(&mut self) {
        unsafe {
            self.device.raw.destroy_shader_module(self.module, None);
        }
    }
}*/

/*
/// Helper to create PipelineShaderStageCreateInfo
fn create_stage<'a>(
    device: &'a Device,
    p_next: *const c_void,
    stage: vk::ShaderStageFlags,
    code: &'a [u32],
    entry_point: &CStr,
) -> Result<(vk::PipelineShaderStageCreateInfo<'static>, ShaderModuleGuard<'a>), Error> {
    let create_info = vk::ShaderModuleCreateInfo {
        flags: Default::default(),
        code_size: code.len() * 4,
        p_code: code.as_ptr(),
        ..Default::default()
    };
    let module = unsafe { device.raw.create_shader_module(&create_info, None)? };
    let stage_create_info = vk::PipelineShaderStageCreateInfo {
        p_next,
        flags: Default::default(),
        stage,
        module,
        p_name: entry_point.as_ptr(),
        p_specialization_info: ptr::null(),
        ..Default::default()
    };
    Ok((stage_create_info, ShaderModuleGuard { device, module }))
}*/

macro_rules! make_stage {
    ($sh:expr, $stage_flags:expr, $module:ident, $p_next:ident, $entry_point_name:ident) => {{
        $entry_point_name = CString::new($sh.entry_point).unwrap();
        $module = vk::ShaderModuleCreateInfo {
            p_next: &$p_next as *const _ as *const c_void,
            code_size: $sh.code.len() * 4,
            p_code: $sh.code.as_ptr(),
            ..Default::default()
        };
        vk::PipelineShaderStageCreateInfo {
            p_next: &$module as *const _ as *const c_void,
            stage: $stage_flags,
            p_name: $entry_point_name.as_ptr(),
            ..Default::default()
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
        let req_subgroup_size = vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo {
            required_subgroup_size: SUBGROUP_SIZE,
            p_next: ptr::null_mut(),
            ..Default::default()
        };
        let _module;
        let _entry_point_name;
        let compute_stage = make_stage!(
            create_info.shader,
            vk::ShaderStageFlags::COMPUTE,
            _module,
            req_subgroup_size,
            _entry_point_name
        );
        let pipeline_create_flags = vk::PipelineCreateFlags2CreateInfoKHR {
            flags: vk::PipelineCreateFlags2KHR::from_raw(VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT),
            ..Default::default()
        };
        let cpci = vk::ComputePipelineCreateInfo {
            p_next: &pipeline_create_flags as *const _ as *const c_void,
            flags: vk::PipelineCreateFlags::empty(),
            stage: compute_stage,
            layout: Default::default(),
            ..Default::default()
        };
        let pipeline = unsafe {
            match self.raw.create_compute_pipelines(vk::PipelineCache::null(), &[cpci], None) {
                Ok(pipelines) => pipelines[0],
                Err(e) => {
                    return Err(Error::Vulkan(e.1));
                }
            }
        };
        Ok(ComputePipeline { pipeline, reflection: create_info.shader.refl_params })
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
            vk::DynamicState::VIEWPORT,
            vk::DynamicState::SCISSOR,
            vk::DynamicState::DEPTH_BIAS,
            vk::DynamicState::DEPTH_BIAS_ENABLE,
        ];
        if matches!(create_info.pre_rasterization_shaders, PreRasterizationShaders::PrimitiveShading { .. }) {
            dynamic_states.push(vk::DynamicState::PRIMITIVE_TOPOLOGY);
        }
        let dynamic_state_create_info = vk::PipelineDynamicStateCreateInfo {
            dynamic_state_count: dynamic_states.len() as u32,
            p_dynamic_states: dynamic_states.as_ptr(),
            ..Default::default()
        };

        // ------ Vertex state ------
        let vertex_input = create_info.vertex_input;
        let vertex_attribute_count = vertex_input.attributes.len();
        let vertex_buffer_count = vertex_input.buffers.len();
        let mut vertex_attribute_descriptions = Vec::with_capacity(vertex_attribute_count);
        let mut vertex_binding_descriptions = Vec::with_capacity(vertex_buffer_count);
        for attribute in vertex_input.attributes.iter() {
            vertex_attribute_descriptions.push(vk::VertexInputAttributeDescription {
                location: attribute.location,
                binding: attribute.binding,
                format: attribute.format,
                offset: attribute.offset,
            });
        }
        for desc in vertex_input.buffers.iter() {
            vertex_binding_descriptions.push(vk::VertexInputBindingDescription {
                binding: desc.binding,
                stride: desc.stride,
                input_rate: desc.input_rate.into(),
            });
        }
        let vertex_input_state = vk::PipelineVertexInputStateCreateInfo {
            vertex_binding_description_count: vertex_buffer_count as u32,
            p_vertex_binding_descriptions: vertex_binding_descriptions.as_ptr(),
            vertex_attribute_description_count: vertex_attribute_count as u32,
            p_vertex_attribute_descriptions: vertex_attribute_descriptions.as_ptr(),
            ..Default::default()
        };
        let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo {
            topology: vk::PrimitiveTopology::TRIANGLE_LIST, // ignored, specified dynamically
            primitive_restart_enable: vk::FALSE,
            ..Default::default()
        };

        // ------ Shader stages ------
        let req_subgroup_size = vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo {
            required_subgroup_size: SUBGROUP_SIZE,
            p_next: ptr::null_mut(),
            ..Default::default()
        };
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
                    vk::ShaderStageFlags::VERTEX,
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
                        vk::ShaderStageFlags::TASK_EXT,
                        task_module,
                        req_subgroup_size,
                        task_entry_point
                    ));
                    stage_reflection.push((ShaderStage::Task, task.refl_params));
                }
                stages.push(make_stage!(
                    mesh,
                    vk::ShaderStageFlags::MESH_EXT,
                    mesh_module,
                    req_subgroup_size,
                    mesh_entry_point
                ));
                stage_reflection.push((ShaderStage::Mesh, mesh.refl_params));
            }
        };
        stages.push(make_stage!(
            create_info.fragment.shader,
            vk::ShaderStageFlags::FRAGMENT,
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
                None => vk::PipelineColorBlendAttachmentState {
                    blend_enable: vk::FALSE,
                    color_write_mask: target.color_write_mask.into(),
                    ..Default::default()
                },
                Some(blend_equation) => vk::PipelineColorBlendAttachmentState {
                    blend_enable: vk::TRUE,
                    src_color_blend_factor: blend_equation.src_color_blend_factor.into(),
                    dst_color_blend_factor: blend_equation.dst_color_blend_factor.into(),
                    color_blend_op: blend_equation.color_blend_op.into(),
                    src_alpha_blend_factor: blend_equation.src_alpha_blend_factor.into(),
                    dst_alpha_blend_factor: blend_equation.dst_alpha_blend_factor.into(),
                    alpha_blend_op: blend_equation.alpha_blend_op.into(),
                    color_write_mask: target.color_write_mask.into(),
                },
            })
            .collect();

        // ------ misc ------
        let conservative_rasterization_state = vk::PipelineRasterizationConservativeStateCreateInfoEXT {
            conservative_rasterization_mode: create_info.rasterization.conservative_rasterization_mode.into(),
            ..Default::default()
        };
        let rasterization_state = vk::PipelineRasterizationStateCreateInfo {
            p_next: &conservative_rasterization_state as *const _ as *const _,
            depth_clamp_enable: create_info.rasterization.depth_clamp_enable.into(),
            rasterizer_discard_enable: 0,
            polygon_mode: create_info.rasterization.polygon_mode.into(),
            cull_mode: create_info.rasterization.cull_mode.into(),
            front_face: create_info.rasterization.front_face.into(),
            depth_bias_enable: vk::FALSE,
            depth_bias_constant_factor: 0.0,
            depth_bias_clamp: 0.0,
            depth_bias_slope_factor: 0.0,
            line_width: 1.0,
            ..Default::default()
        };
        let multisample_state = vk::PipelineMultisampleStateCreateInfo {
            rasterization_samples: vk::SampleCountFlags::TYPE_1,
            sample_shading_enable: vk::FALSE,
            min_sample_shading: 0.0,
            p_sample_mask: ptr::null(),
            alpha_to_coverage_enable: create_info.fragment.multisample.alpha_to_coverage_enabled.into(),
            alpha_to_one_enable: vk::FALSE,
            ..Default::default()
        };
        let color_blend_state = vk::PipelineColorBlendStateCreateInfo {
            flags: Default::default(),
            logic_op_enable: vk::FALSE,
            logic_op: Default::default(),
            attachment_count: attachment_states.len() as u32,
            p_attachments: attachment_states.as_ptr(),
            blend_constants: create_info.fragment.blend_constants,
            ..Default::default()
        };
        let depth_stencil_state = if let Some(ds) = create_info.depth_stencil {
            vk::PipelineDepthStencilStateCreateInfo {
                flags: Default::default(),
                depth_test_enable: (ds.depth_compare_op != vk::CompareOp::ALWAYS).into(),
                depth_write_enable: ds.depth_write_enable.into(),
                depth_compare_op: ds.depth_compare_op,
                stencil_test_enable: ds.stencil_state.is_enabled().into(),
                front: ds.stencil_state.front.into(),
                back: ds.stencil_state.back.into(),
                depth_bounds_test_enable: vk::FALSE,
                min_depth_bounds: 0.0,
                max_depth_bounds: 0.0,
                ..Default::default()
            }
        } else {
            Default::default()
        };
        let color_attachment_formats =
            create_info.fragment.color_targets.iter().map(|target| target.format).collect::<Vec<_>>();
        let depth_attachment_format = create_info.depth_stencil.map(|ds| ds.format).unwrap_or(vk::Format::UNDEFINED);
        let stencil_attachment_format = if is_depth_and_stencil_format(depth_attachment_format) {
            depth_attachment_format
        } else {
            vk::Format::UNDEFINED
        };
        let rendering_info = vk::PipelineRenderingCreateInfo {
            view_mask: 0,
            color_attachment_count: color_attachment_formats.len() as u32,
            p_color_attachment_formats: color_attachment_formats.as_ptr(),
            depth_attachment_format,
            stencil_attachment_format,
            ..Default::default()
        };
        let pipeline_create_flags = vk::PipelineCreateFlags2CreateInfoKHR {
            p_next: &rendering_info as *const _ as *const c_void,
            flags: vk::PipelineCreateFlags2KHR::from_raw(vk2::VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT),
            ..Default::default()
        };
        let pipeline_create_info = vk::GraphicsPipelineCreateInfo {
            p_next: &pipeline_create_flags as *const _ as *const _,
            flags: vk::PipelineCreateFlags::empty(),
            stage_count: stages.len() as u32,
            p_stages: stages.as_ptr(),
            p_vertex_input_state: &vertex_input_state,
            p_input_assembly_state: &input_assembly_state,
            p_tessellation_state: &Default::default(),
            p_viewport_state: &vk::PipelineViewportStateCreateInfo {
                viewport_count: 1,
                scissor_count: 1,
                ..Default::default()
            },
            p_rasterization_state: &rasterization_state,
            p_multisample_state: &multisample_state,
            p_depth_stencil_state: &depth_stencil_state,
            p_color_blend_state: &color_blend_state,
            p_dynamic_state: &dynamic_state_create_info,
            layout: Default::default(),
            render_pass: Default::default(),
            subpass: 0,
            base_pipeline_handle: Default::default(),
            base_pipeline_index: 0,
            ..Default::default()
        };

        let pipeline = unsafe {
            match self.raw.create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_create_info], None) {
                Ok(pipelines) => pipelines[0],
                Err(e) => {
                    return Err(Error::Vulkan(e.1));
                }
            }
        };
        Ok(GraphicsPipeline { pipeline, stage_reflection })
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
        let wait_info = vk::SemaphoreWaitInfo {
            semaphore_count: 1,
            p_semaphores: &device.thread_safe.frame_timeline,
            p_values: &frame_index,
            ..Default::default()
        };
        device.raw.wait_semaphores(&wait_info, timeout.map(|d| d.as_nanos() as u64).unwrap_or(u64::MAX)).unwrap();
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
pub fn get_frame_index() -> FrameIndex {
    Device::instance().frame_index.load(Relaxed)
}

/// Returns the index of the most recently completed frame.
pub fn get_last_completed_frame_index() -> FrameIndex {
    unsafe {
        let device = Device::instance();
        device
            .raw
            .get_semaphore_counter_value(device.thread_safe.frame_timeline)
            .expect("get_semaphore_counter_value failed")
    }
}

/// Returns the `VkPhysicalDeviceProperties` of the physical device used by the global device.
pub fn get_physical_device_properties() -> vk::PhysicalDeviceProperties {
    Device::instance().thread_safe.physical_device_properties
}

/// Returns the name of the physical device used by the global device.
pub fn get_physical_device_name() -> String {
    let properties = get_physical_device_properties();
    let device_name = unsafe { CStr::from_ptr(properties.device_name.as_ptr()) };
    device_name.to_string_lossy().into_owned()
}

/// Returns the device UUID of the physical device.
pub fn get_device_uuid() -> [u8; 16] {
    Device::instance().thread_safe.physical_device_id_properties.device_uuid
}

/// Returns the device LUID of the physical device.
pub fn get_device_luid() -> Option<[u8; 8]> {
    let id_properties = &Device::instance().thread_safe.physical_device_id_properties;
    if id_properties.device_luid_valid == vk::TRUE { Some(id_properties.device_luid) } else { None }
}

/// Returns the timestamp period in nanoseconds.
pub fn get_timestamp_period() -> f32 {
    let properties = get_physical_device_properties();
    properties.limits.timestamp_period
}

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
