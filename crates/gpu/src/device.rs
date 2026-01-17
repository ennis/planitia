//! Abstractions over a vulkan device & queues.
mod bindless;
mod upload_buffer;

use crate::device::bindless::BindlessDescriptorTable;
use crate::device::upload_buffer::{UploadBuffer, UPLOAD_BUFFER_CHUNK_SIZE};
use crate::instance::vk_khr_surface;
use crate::platform::PlatformExtensions;
use crate::{
    get_vulkan_entry, get_vulkan_instance, is_depth_and_stencil_format, BufferCreateInfo, BufferUntyped, BufferUsage,
    CommandPool, ComputePipeline, ComputePipelineCreateInfo, DescriptorSetLayout, Error, GraphicsPipeline,
    GraphicsPipelineCreateInfo, PreRasterizationShaders, Ptr, Sampler, SamplerCreateInfo, SamplerCreateInfoHashable,
    VulkanObject, SUBGROUP_SIZE,
};
use ash::vk;
use gpu_allocator::vulkan::AllocationCreateDesc;
use log::{debug, error, trace};
use slotmap::SlotMap;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::ffi::{c_void, CStr, CString};
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::{Arc, LazyLock, Mutex};
use std::{fmt, mem, ptr};
////////////////////////////////////////////////////////////////////////////////////////////////////

/// Size of the global descriptor heaps (in number of descriptors).
const DESCRIPTOR_TABLE_SIZE: usize = 4096;

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Device extensions.
pub(crate) struct DeviceExtensions {
    pub(crate) khr_swapchain: ash::khr::swapchain::Device,
    //pub(crate) ext_shader_object: ash::ext::,
    pub(crate) khr_push_descriptor: ash::khr::push_descriptor::Device,
    pub(crate) ext_mesh_shader: ash::ext::mesh_shader::Device,
    pub(crate) _ext_extended_dynamic_state3: ash::ext::extended_dynamic_state3::Device,
    pub(crate) ext_debug_utils: ash::ext::debug_utils::Device,
}

/// Device state that is unconditionally safe to access from multiple threads, even though
/// the fields themselves may not be Send or Sync.
pub(crate) struct DeviceThreadSafeState {
    pub(crate) physical_device_memory_properties: vk::PhysicalDeviceMemoryProperties,
    _physical_device_descriptor_buffer_properties: vk::PhysicalDeviceDescriptorBufferPropertiesEXT<'static>,
    _physical_device_properties: vk::PhysicalDeviceProperties2<'static>,

    // SAFETY: we're never using this as an externally-synchronized command parameter.
    pub(crate) timeline: vk::Semaphore,
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

    // Pending writes not yet made visible.
    //pub(crate) writes: MemoryAccess,
    // Last access type tracked per resource. Used mostly to track image layouts.
    // Get rid of this once GENERAL layouts with no performance penalty are widely supported.
    //pub(crate) access_per_resource: SecondaryMap<ResourceId, MemoryAccess>,
}

pub(crate) struct ResourceState {
    //pub(crate) last_submission_index: u64,
}

pub(crate) struct DeviceDescriptorIndexTable {
    pub(crate) resource_descriptor_indices: SlotMap<ResourceDescriptorIndex, ()>,
    pub(crate) sampler_descriptor_indices: SlotMap<SamplerDescriptorIndex, ()>,
}

pub struct Device {
    /// Underlying vulkan device
    pub(crate) raw: ash::Device,

    /// Common device extensions.
    pub(crate) extensions: DeviceExtensions,
    /// Platform-specific extension functions
    pub(crate) platform_extensions: PlatformExtensions,
    pub(crate) allocator: Mutex<gpu_allocator::vulkan::Allocator>,

    /// Global upload buffer.
    upload_buffer: Mutex<UploadBuffer>,

    // main graphics queue
    pub(crate) queue_family: u32,

    pub(crate) thread_safe: DeviceThreadSafeState,
    pub(crate) submission_state: Mutex<DeviceSubmissionState>,
    pub(crate) resources: Mutex<SlotMap<ResourceId, ResourceState>>,
    pub(crate) descriptor_indices: Mutex<DeviceDescriptorIndexTable>,
    pub(crate) descriptor_table: BindlessDescriptorTable,

    // semaphores ready for reuse
    pub(crate) semaphores: Mutex<Vec<vk::Semaphore>>,

    // Command pools per queue and thread.
    free_command_pools: Mutex<Vec<CommandPool>>,

    /// Index of the next submission not yet created.
    pub(crate) next_create_ticket: AtomicU64,

    /// Index of the next submission to be submitted.
    pub(crate) next_timeline_value: AtomicU64,

    /// All command buffers with create_index <= than this value have completed execution.
    ///
    /// There might be some command buffers with a higher create_index that have also completed,
    /// but this is the highest value for which we can guarantee that all lower-indexed command
    /// buffers have completed.
    pub(crate) completed_tickets: AtomicU64,

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

    pub(crate) sampler_cache: Mutex<HashMap<SamplerCreateInfoHashable, Sampler>>,
}

impl fmt::Debug for Device {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("DeviceInner").finish_non_exhaustive()
    }
}

pub(crate) struct ActiveSubmission {
    pub(crate) create_ticket: u64,
    pub(crate) timeline_value: u64,
    pub(crate) command_pools: Vec<CommandPool>,
}

struct DeleteQueueEntry {
    create_ticket: u64,
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

slotmap::new_key_type! {
    /// Identifies a GPU resource for tracking.
    pub struct ResourceId;
    /// Identifies an image resource (sampled or storage image) in a bindless descriptor heap.
    pub(crate) struct ResourceDescriptorIndex;
    /// Identifies a sampler in a bindless sampler descriptor heap.
    pub struct SamplerDescriptorIndex;
    /// Identifies a resource group.
    pub struct GroupId;

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
pub enum ResourceAllocation {
    /// We don't own the memory for this resource.
    #[default]
    External,
    /// We allocated a block of memory exclusively for this resource.
    Allocation {
        allocation: gpu_allocator::vulkan::Allocation,
    },
    /// The memory for this resource was imported or exported from/to an external handle.
    DeviceMemory { device_memory: vk::DeviceMemory },
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
            width: framebuffer_size
                .0
                .clamp(capabilities.min_image_extent.width, capabilities.max_image_extent.width),
            height: framebuffer_size.1.clamp(
                capabilities.min_image_extent.height,
                capabilities.max_image_extent.height,
            ),
        }
    }
}

unsafe fn select_physical_device(instance: &ash::Instance) -> PhysicalDeviceAndProperties {
    let physical_devices = instance
        .enumerate_physical_devices()
        .expect("failed to enumerate physical devices");
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
    // TODO fallbacks

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
                if !vk_khr_surface
                    .get_physical_device_surface_support(phy, index, surface)
                    .unwrap()
                {
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
    "VK_KHR_push_descriptor",
    "VK_EXT_extended_dynamic_state3",
    "VK_EXT_mesh_shader",
    "VK_EXT_conservative_rasterization",
    "VK_EXT_fragment_shader_interlock",
    "VK_EXT_shader_image_atomic_int64",
    "VK_EXT_mutable_descriptor_type", //"VK_EXT_descriptor_buffer",
];

////////////////////////////////////////////////////////////////////////////////////////////////
// INITIALIZATION
////////////////////////////////////////////////////////////////////////////////////////////////

impl Device {
    /// Returns the global device.
    pub fn global() -> &'static Device {
        static DEVICE: LazyLock<&'static Device> =
            LazyLock::new(|| Box::leak(Box::new(create_device().expect("failed to create the GPU device"))));
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

    /*/// Returns whether this device is compatible for presentation on the specified surface.
    ///
    /// More precisely, it checks that the graphics queue created for this device can present to the given surface.
    pub unsafe fn is_compatible_for_presentation(&self, surface: vk::SurfaceKHR) -> bool {
        vk_khr_surface()
            .get_physical_device_surface_support(self.inner.physical_device, self.graphics_queue().1, surface)
            .unwrap()
    }*/

    // TODO: enabled features?

    /// Creates a new `Device` from an existing vulkan device.
    ///
    /// The device should have been created with at least one graphics queue.
    ///
    /// # Arguments
    ///
    /// * `physical_device` - the physical device that the device was created on
    /// * `device` - the vulkan device handle
    /// * `graphics_queue_family_index` - queue family index of the main graphics queue
    pub unsafe fn from_existing(
        physical_device: vk::PhysicalDevice,
        device: vk::Device,
        graphics_queue_family_index: u32,
    ) -> Result<Device, DeviceCreateError> {
        let entry = get_vulkan_entry();
        let instance = get_vulkan_instance();
        let device = ash::Device::load(instance.fp_v1_0(), device);

        // fetch the graphics queue
        let queue = device.get_device_queue(graphics_queue_family_index, 0);

        // create timeline semaphore
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
            device
                .create_semaphore(&semaphore_create_info, None)
                .expect("failed to queue timeline semaphore")
        };

        // Create the GPU memory allocator
        let allocator_create_desc = gpu_allocator::vulkan::AllocatorCreateDesc {
            physical_device,
            debug_settings: Default::default(),
            device: device.clone(),     // not cheap!
            instance: instance.clone(), // not cheap!
            buffer_device_address: true,
            allocation_sizes: Default::default(),
        };

        let allocator =
            gpu_allocator::vulkan::Allocator::new(&allocator_create_desc).expect("failed to create GPU allocator");

        // Extensions
        let khr_swapchain = ash::khr::swapchain::Device::new(instance, &device);
        let khr_push_descriptor = ash::khr::push_descriptor::Device::new(instance, &device);
        let ext_extended_dynamic_state3 = ash::ext::extended_dynamic_state3::Device::new(instance, &device);
        let ext_mesh_shader = ash::ext::mesh_shader::Device::new(instance, &device);
        //let vk_ext_descriptor_buffer = ash::extensions::ext::DescriptorBuffer::new(instance, &device);
        let physical_device_memory_properties = instance.get_physical_device_memory_properties(physical_device);
        let ext_debug_utils = ash::ext::debug_utils::Device::new(instance, &device);
        let platform_extensions = PlatformExtensions::load(entry, instance, &device);

        let mut physical_device_descriptor_buffer_properties =
            vk::PhysicalDeviceDescriptorBufferPropertiesEXT::default();
        let mut physical_device_properties = vk::PhysicalDeviceProperties2 {
            p_next: &mut physical_device_descriptor_buffer_properties as *mut _ as *mut c_void,
            ..Default::default()
        };

        instance.get_physical_device_properties2(physical_device, &mut physical_device_properties);

        // Create global descriptor tables
        let descriptor_table = BindlessDescriptorTable::new(&device, DESCRIPTOR_TABLE_SIZE);

        Ok(Device {
            raw: device,
            extensions: DeviceExtensions {
                khr_swapchain,
                khr_push_descriptor,
                ext_mesh_shader,
                _ext_extended_dynamic_state3: ext_extended_dynamic_state3,
                ext_debug_utils,
            },
            platform_extensions,
            thread_safe: DeviceThreadSafeState {
                physical_device_memory_properties,
                _physical_device_descriptor_buffer_properties: physical_device_descriptor_buffer_properties,
                _physical_device_properties: physical_device_properties,
                timeline,
                physical_device,
            },
            submission_state: Mutex::new(DeviceSubmissionState {
                queue,
                active_submissions: VecDeque::new(),
            }),
            queue_family: graphics_queue_family_index,
            allocator: Mutex::new(allocator),
            resources: Mutex::new(SlotMap::with_key()),
            descriptor_indices: Mutex::new(DeviceDescriptorIndexTable {
                resource_descriptor_indices: Default::default(),
                sampler_descriptor_indices: Default::default(),
            }),
            descriptor_table,
            sampler_cache: Mutex::new(Default::default()),
            free_command_pools: Mutex::new(Default::default()),
            next_create_ticket: AtomicU64::new(1),
            next_timeline_value: AtomicU64::new(1),
            semaphores: Default::default(),
            deletion_queue: Mutex::new(Vec::new()),
            upload_buffer: Mutex::new(UploadBuffer::new(BufferUsage::UNIFORM)),
            completed_tickets: AtomicU64::new(0),
        })
    }

    /// Creates a new `Device`, automatically choosing a suitable physical device.
    pub fn new() -> Result<Device, DeviceCreateError> {
        unsafe { Self::with_surface(None) }
    }

    /// Returns the list of supported swapchain formats for the given surface.
    pub unsafe fn get_surface_formats(&self, surface: vk::SurfaceKHR) -> Vec<vk::SurfaceFormatKHR> {
        vk_khr_surface()
            .get_physical_device_surface_formats(self.thread_safe.physical_device, surface)
            .unwrap()
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

        debug!(
            "selected physical device: {:?}",
            CStr::from_ptr(phy.properties.device_name.as_ptr())
        );

        // ------ Setup device create info ------
        let queue_priorities = [1.0f32];
        let device_queue_create_infos = &[vk::DeviceQueueCreateInfo {
            flags: Default::default(),
            queue_family_index: graphics_queue_family,
            queue_count: 1,
            p_queue_priorities: queue_priorities.as_ptr(),
            ..Default::default()
        }];

        let mut fragment_shader_interlock_features = vk::PhysicalDeviceFragmentShaderInterlockFeaturesEXT {
            //p_next: &mut timeline_features as *mut _ as *mut c_void,
            fragment_shader_pixel_interlock: vk::TRUE,
            ..Default::default()
        };

        /*let mut descriptor_buffer_features = vk::PhysicalDeviceDescriptorBufferFeaturesEXT {
            p_next: &mut fragment_shader_interlock_features as *mut _ as *mut c_void,
            descriptor_buffer: vk::TRUE,
            ..Default::default()
        };*/

        let mut mutable_descriptor_type_features = vk::PhysicalDeviceMutableDescriptorTypeFeaturesEXT {
            p_next: &mut fragment_shader_interlock_features as *mut _ as *mut c_void,
            mutable_descriptor_type: vk::TRUE,
            ..Default::default()
        };

        let mut mesh_shader_features = vk::PhysicalDeviceMeshShaderFeaturesEXT {
            p_next: &mut mutable_descriptor_type_features as *mut _ as *mut c_void,
            task_shader: vk::TRUE,
            mesh_shader: vk::TRUE,
            ..Default::default()
        };

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
            dynamic_rendering: vk::TRUE,
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

    /// Allocates a new resource ID for tracking a resource.
    pub(crate) fn allocate_resource_id(&self) -> ResourceId {
        self.resources.lock().unwrap().insert(ResourceState {
            //last_submission_index: 0,
        })
    }

    /// Releases a resource ID that is no longer used.
    pub(crate) fn free_resource_id(&self, resource_id: ResourceId) {
        self.resources.lock().unwrap().remove(resource_id);
    }

    /// Releases a resource heap index that is no longer used.
    pub(crate) fn free_resource_heap_index(&self, index: ResourceDescriptorIndex) {
        self.descriptor_indices
            .lock()
            .unwrap()
            .resource_descriptor_indices
            .remove(index);
    }

    /// Allocates memory, or panic trying.
    ///
    /// This is used internally for resource creation since we don't expose memory allocation errors to the user.
    pub(crate) fn allocate_memory_or_panic(
        &self,
        create_desc: &AllocationCreateDesc,
    ) -> gpu_allocator::vulkan::Allocation {
        self.allocator
            .lock()
            .unwrap()
            .allocate(create_desc)
            .expect("failed to allocate device memory")
    }

    fn ticket_completed(&self, ticket: u64) -> bool {
        let last_completed_submission_index =
            unsafe { self.raw.get_semaphore_counter_value(self.thread_safe.timeline).unwrap() };

        if last_completed_submission_index == u64::MAX {
            error!("GetSemaphoreCounterValue returned UINT64_MAX");
            return false;
        }

        // find corresponding submission for ticket
        let submission_state = self.submission_state.lock().unwrap();
        for s in submission_state.active_submissions.iter() {
            if s.create_ticket == ticket {
                return s.timeline_value <= last_completed_submission_index;
            }
        }

        ticket <= last_completed_submission_index
    }

    /// Schedules a function call.
    ///
    /// The function will be called once the GPU has finished processing commands up to and
    /// including the specified submission index.
    pub fn call_later(&self, ticket: u64, f: impl FnOnce(&Self) + Send + Sync + 'static) {
        // if the command buffer has already completed execution, call the function right away
        if ticket <= self.completed_tickets.load(Relaxed) {
            trace!("GPU: immediate call_later for ticket={ticket})");
            f(self);
        } else {
            // otherwise move it to the deferred deletion list
            let mut deletion_queue = self.deletion_queue.lock().unwrap();
            let pos = deletion_queue
                .binary_search_by_key(&ticket, |e| e.create_ticket)
                .unwrap_or_else(|p| p);
            deletion_queue.insert(
                pos,
                DeleteQueueEntry {
                    create_ticket: ticket,
                    deleter: Some(Box::new(f)),
                },
            );
        }
    }

    /// Schedules a resource for deletion after the current submission is complete.
    pub(crate) fn delete_resource_after_current_submission(
        &self,
        resource_id: ResourceId,
        deleter: impl FnOnce(&Self) + Send + Sync + 'static,
    ) {
        // See comments in `delete_after_current_submission`.
        let last_create_ticket = self.next_create_ticket.load(Relaxed) - 1;
        self.call_later(last_create_ticket, move |device| {
            //trace!("GPU: deleting tracked resource {:?}", resource_id);
            Self::global().free_resource_id(resource_id);
            deleter(device);
        })
    }

    /// Schedules a function (destructor) to be called after the current submission is complete.
    pub(crate) fn delete_after_current_submission(&self, deleter: impl FnOnce(&Self) + Send + Sync + 'static) {
        let last_ticket = self.next_create_ticket.load(Relaxed) - 1;
        self.call_later(last_ticket, move |device| {
            deleter(device);
        })
    }

    pub(crate) unsafe fn free_memory(&self, allocation: &mut ResourceAllocation) {
        match mem::replace(allocation, ResourceAllocation::External) {
            ResourceAllocation::Allocation { allocation } => self
                .allocator
                .lock()
                .unwrap()
                .free(allocation)
                .expect("failed to free memory"),
            ResourceAllocation::DeviceMemory { device_memory } => unsafe {
                self.raw.free_memory(device_memory, None);
            },
            ResourceAllocation::External => {
                unreachable!()
            }
        }
    }

    /// Uploads data to GPU memory via this device's upload buffer.
    ///
    /// The returned pointer is guaranteed to be valid for the current submission.
    pub fn upload<T: Copy>(&self, data: &[T]) -> Ptr<T> {
        let mut upload_buffer = self.upload_buffer.lock().unwrap();
        upload_buffer.allocate_slice(data)
    }

    /// Uploads data to GPU memory via this device's upload buffer.
    ///
    /// The returned pointer is guaranteed to be valid for the current submission.
    pub fn upload_one<T: Copy>(&self, data: &T) -> Ptr<T> {
        let mut upload_buffer = self.upload_buffer.lock().unwrap();
        upload_buffer.allocate(data)
    }

    /// Schedules deletion of full upload buffers.
    fn retire_upload_buffers(&self) {
        let mut upload_buffer = self.upload_buffer.lock().unwrap();
        let full = mem::take(&mut upload_buffer.full);
        if !full.is_empty() {
            trace!(
                "deleting {} full upload buffers ({} MB)",
                full.len(),
                full.len() * UPLOAD_BUFFER_CHUNK_SIZE / (1024 * 1024)
            );
            for _buffer in full {
                // nothing to do, the drop impl does what we want
            }
        }
    }

    fn maintain(&self) {
        self.retire_upload_buffers();
        let last_completed_submission_index = unsafe {
            self.raw
                .get_semaphore_counter_value(self.thread_safe.timeline)
                .expect("get_semaphore_counter_value failed")
        };
        if last_completed_submission_index == u64::MAX {
            error!("GetSemaphoreCounterValue returned UINT64_MAX");
            return;
        }

        //trace!("GPU: cleaning up to submission {last_completed_submission_index}");

        // process all completed submissions
        let mut free_command_pools = self.free_command_pools.lock().unwrap();
        let mut submission_state = self.submission_state.lock().unwrap();
        loop {
            let Some(submission) = submission_state.active_submissions.front() else {
                break;
            };
            if submission.timeline_value > last_completed_submission_index {
                break;
            }
            let submission = submission_state.active_submissions.pop_front().unwrap();
            for mut command_pool in submission.command_pools {
                // SAFETY: command buffers are not in use anymore
                unsafe {
                    command_pool.reset(&self.raw);
                }
                free_command_pools.push(command_pool);
            }
            // update completed tickets
            self.completed_tickets.store(submission.create_ticket, Relaxed);
        }

        let mut deletion_queue = self.deletion_queue.lock().unwrap();
        let completed_tickets = self.completed_tickets.load(Relaxed);

        // *** This invokes all delayed destructors for resources which are no longer in use by the GPU.
        deletion_queue.retain_mut(|DeleteQueueEntry { create_ticket, deleter }| {
            if *create_ticket > completed_tickets {
                return true;
            }
            let deleter = deleter.take().unwrap();
            deleter(self);
            false
        });
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

    pub(crate) fn create_sampler(&self, info: &SamplerCreateInfo) -> Sampler {
        let info_hashable = SamplerCreateInfoHashable::from(*info);
        if let Some(sampler) = self.sampler_cache.lock().unwrap().get(&info_hashable) {
            return sampler.clone();
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

        let sampler = unsafe {
            self.raw
                .create_sampler(&create_info, None)
                .expect("failed to create sampler")
        };

        let descriptor_index = unsafe { self.create_global_sampler_descriptor(sampler) };
        let sampler = Sampler {
            descriptor_index,
            sampler,
        };
        self.sampler_cache
            .lock()
            .unwrap()
            .insert(info_hashable, sampler.clone());
        sampler
    }

    pub(crate) fn get_or_create_command_pool(&self, queue_family: u32) -> CommandPool {
        let free_command_pools = &mut self.free_command_pools.lock().unwrap();
        let index = free_command_pools
            .iter()
            .position(|pool| pool.queue_family == queue_family);
        if let Some(index) = index {
            free_command_pools.swap_remove(index)
        } else {
            unsafe { CommandPool::new(&self.raw, queue_family) }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////>
// SHADERS, PIPELINES & LAYOUTS
////////////////////////////////////////////////////////////////////////////////////////////////

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
}

/// Helper to create PipelineShaderStageCreateInfo
fn create_stage<'a>(
    device: &'a Device,
    p_next: *const c_void,
    stage: vk::ShaderStageFlags,
    code: &[u32],
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
}

impl Device {
    /// FIXME: this should be a constructor of `DescriptorSetLayout`, because now we have two
    /// functions with very similar names (`create_descriptor_set_layout` and `create_descriptor_set_layout_from_handle`)
    /// that have totally different semantics (one returns a raw vulkan handle, the other returns a RAII wrapper `DescriptorSetLayout`).
    pub fn create_descriptor_set_layout_from_handle(&self, handle: vk::DescriptorSetLayout) -> DescriptorSetLayout {
        DescriptorSetLayout {
            last_submission_index: Some(Arc::new(Default::default())),
            handle,
        }
    }

    pub fn create_push_descriptor_set_layout(
        &self,
        bindings: &[vk::DescriptorSetLayoutBinding],
    ) -> DescriptorSetLayout {
        let create_info = vk::DescriptorSetLayoutCreateInfo {
            flags: vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR,
            binding_count: bindings.len() as u32,
            p_bindings: bindings.as_ptr(),
            ..Default::default()
        };
        let handle = unsafe {
            self.raw
                .create_descriptor_set_layout(&create_info, None)
                .expect("failed to create descriptor set layout")
        };
        self.create_descriptor_set_layout_from_handle(handle)
    }

    /// Creates a pipeline layout object.
    fn create_pipeline_layout(
        &self,
        bind_point: vk::PipelineBindPoint,
        descriptor_set_layouts: &[DescriptorSetLayout],
        push_constants_size: usize,
    ) -> vk::PipelineLayout {
        let layout_handles: Vec<_> = if descriptor_set_layouts.is_empty() {
            // Empty set layouts means use the universal bindless layouts
            vec![self.descriptor_table.layout]
        } else {
            descriptor_set_layouts.iter().map(|layout| layout.handle).collect()
        };

        let pc_range = if push_constants_size != 0 {
            Some(match bind_point {
                vk::PipelineBindPoint::GRAPHICS => vk::PushConstantRange {
                    stage_flags: vk::ShaderStageFlags::ALL_GRAPHICS
                        | vk::ShaderStageFlags::MESH_EXT
                        | vk::ShaderStageFlags::TASK_EXT,
                    offset: 0,
                    size: push_constants_size as u32,
                },
                vk::PipelineBindPoint::COMPUTE => vk::PushConstantRange {
                    stage_flags: vk::ShaderStageFlags::COMPUTE,
                    offset: 0,
                    size: push_constants_size as u32,
                },
                _ => unimplemented!(),
            })
        } else {
            None
        };
        let pc_range = pc_range.as_slice();

        let create_info = vk::PipelineLayoutCreateInfo {
            set_layout_count: layout_handles.len() as u32,
            p_set_layouts: layout_handles.as_ptr(),
            push_constant_range_count: pc_range.len() as u32,
            p_push_constant_ranges: pc_range.as_ptr(),
            ..Default::default()
        };

        let pipeline_layout = unsafe {
            self.raw
                .create_pipeline_layout(&create_info, None)
                .expect("failed to create pipeline layout")
        };

        pipeline_layout
    }

    pub(crate) fn create_compute_pipeline(
        &self,
        create_info: ComputePipelineCreateInfo,
    ) -> Result<ComputePipeline, Error> {
        let pipeline_layout = self.create_pipeline_layout(
            vk::PipelineBindPoint::COMPUTE,
            create_info.set_layouts,
            create_info.push_constants_size,
        );

        let is_bindless = create_info.set_layouts.is_empty();

        let req_subgroup_size = vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo {
            required_subgroup_size: SUBGROUP_SIZE,
            p_next: ptr::null_mut(),
            ..Default::default()
        };

        let entry_point = CString::new(create_info.shader.entry_point).unwrap();
        let (compute_stage, _module) = create_stage(
            &self,
            &req_subgroup_size as *const _ as *const c_void,
            vk::ShaderStageFlags::COMPUTE,
            &create_info.shader.code,
            &entry_point,
        )?;

        let cpci = vk::ComputePipelineCreateInfo {
            flags: vk::PipelineCreateFlags::empty(),
            stage: compute_stage,
            layout: pipeline_layout,
            ..Default::default()
        };

        let pipeline = unsafe {
            match self
                .raw
                .create_compute_pipelines(vk::PipelineCache::null(), &[cpci], None)
            {
                Ok(pipelines) => pipelines[0],
                Err(e) => {
                    return Err(Error::Vulkan(e.1));
                }
            }
        };

        Ok(ComputePipeline {
            pipeline,
            pipeline_layout,
            _descriptor_set_layouts: create_info.set_layouts.to_vec(),
            bindless: is_bindless,
        })
    }

    /// Creates a graphics pipeline.
    pub(crate) fn create_graphics_pipeline(
        &self,
        create_info: GraphicsPipelineCreateInfo,
    ) -> Result<GraphicsPipeline, Error> {
        let pipeline_layout = self.create_pipeline_layout(
            vk::PipelineBindPoint::GRAPHICS,
            create_info.set_layouts,
            create_info.push_constants_size,
        );

        let bindless = create_info.set_layouts.is_empty();

        // ------ Dynamic states ------

        // TODO: this could be a static property of the pipeline interface
        let mut dynamic_states = vec![
            vk::DynamicState::VIEWPORT,
            vk::DynamicState::SCISSOR,
            vk::DynamicState::DEPTH_BIAS,
            vk::DynamicState::DEPTH_BIAS_ENABLE,
        ];

        if matches!(
            create_info.pre_rasterization_shaders,
            PreRasterizationShaders::PrimitiveShading { .. }
        ) {
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
        let p_next = &req_subgroup_size as *const _ as *const c_void;

        let mut stages = Vec::new();

        // those variables are referenced by VkGraphicsPipelineCreateInfo
        // put them here so that they live at least until vkCreateGraphicsPipelines is called
        let vertex_entry_point;
        let task_entry_point;
        let mesh_entry_point;
        let fragment_entry_point;
        let _vertex_module;
        let _task_module;
        let _mesh_module;
        let _fragment_module;

        match create_info.pre_rasterization_shaders {
            PreRasterizationShaders::PrimitiveShading { vertex } => {
                vertex_entry_point = CString::new(vertex.entry_point).unwrap();
                let (stage, module) = create_stage(
                    &self,
                    p_next,
                    vk::ShaderStageFlags::VERTEX,
                    &vertex.code,
                    &vertex_entry_point,
                )?;
                _vertex_module = module;
                stages.push(stage);
            }
            PreRasterizationShaders::MeshShading { mesh, task } => {
                if let Some(task) = task {
                    task_entry_point = CString::new(task.entry_point).unwrap();
                    let (stage, module) = create_stage(
                        &self,
                        p_next,
                        vk::ShaderStageFlags::TASK_EXT,
                        &task.code,
                        &task_entry_point,
                    )?;
                    _task_module = module;
                    stages.push(stage);
                }

                mesh_entry_point = CString::new(mesh.entry_point).unwrap();
                let (stage, module) = create_stage(
                    &self,
                    p_next,
                    vk::ShaderStageFlags::MESH_EXT,
                    &mesh.code,
                    &mesh_entry_point,
                )?;
                _mesh_module = module;
                stages.push(stage);
            }
        };

        fragment_entry_point = CString::new(create_info.fragment.shader.entry_point).unwrap();
        let (stage, module) = create_stage(
            &self,
            p_next,
            vk::ShaderStageFlags::FRAGMENT,
            &create_info.fragment.shader.code,
            &fragment_entry_point,
        )?;
        _fragment_module = module;
        stages.push(stage);

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

        let color_attachment_formats = create_info
            .fragment
            .color_targets
            .iter()
            .map(|target| target.format)
            .collect::<Vec<_>>();
        let depth_attachment_format = create_info
            .depth_stencil
            .map(|ds| ds.format)
            .unwrap_or(vk::Format::UNDEFINED);
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

        let pipeline_create_info = vk::GraphicsPipelineCreateInfo {
            p_next: &rendering_info as *const _ as *const _,
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
            layout: pipeline_layout,
            render_pass: Default::default(),
            subpass: 0,
            base_pipeline_handle: Default::default(),
            base_pipeline_index: 0,
            ..Default::default()
        };

        let pipeline = unsafe {
            match self
                .raw
                .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_create_info], None)
            {
                Ok(pipelines) => pipelines[0],
                Err(e) => {
                    return Err(Error::Vulkan(e.1));
                }
            }
        };

        Ok(GraphicsPipeline {
            pipeline,
            pipeline_layout,
            _descriptor_set_layouts: create_info.set_layouts.to_vec(),
            bindless,
        })
    }
}

/// Waits for the GPU to complete all submitted work.
pub fn wait_idle() {
    unsafe { Device::global().raw.device_wait_idle().unwrap() }
}

/// Cleanup expired resources.
///
/// This should be called periodically (e.g. per-frame) to free resources that are no longer
/// used by the GPU.
/// Otherwise, tasks scheduled with `call_later` or `delete_later` will never be executed.
pub fn maintain() {
    Device::global().maintain()
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
    let device = Device::global();
    let object_name = CString::new(name.as_ref()).unwrap();

    unsafe {
        // SAFETY: TODO
        device
            .extensions
            .ext_debug_utils
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
