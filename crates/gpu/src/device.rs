//! Abstractions over a vulkan device & queues.
mod bindless;

use crate::instance::{vk_ext_debug_utils, vk_khr_surface};
use crate::{
    aspects_for_format, get_vulkan_entry, get_vulkan_instance, is_depth_and_stencil_format, BufferInner, BufferUntyped,
    BufferUsage, CommandPool, CommandStream, ComputePipeline, ComputePipelineCreateInfo, DescriptorSetLayout, Error,
    Format, GraphicsPipeline, GraphicsPipelineCreateInfo, Image, ImageCreateInfo, ImageInner, ImageType, ImageUsage,
    ImageViewInfo, MemoryAccess, MemoryLocation, PreRasterizationShaders, Sampler, SamplerCreateInfo,
    SignaledSemaphore, Size3D, SwapChain, SwapchainImage, SwapchainImageInner, SUBGROUP_SIZE,
};
use std::cell::{Cell, RefCell};
use std::collections::{HashMap, VecDeque};
use std::ffi::{c_void, CString};
use std::rc::{Rc, Weak};
use std::sync::{Arc, Mutex};
use std::{fmt, ptr};

use crate::device::bindless::BindlessDescriptorTable;
use crate::platform::PlatformExtensions;
use ash::vk;
use gpu_allocator::vulkan::{AllocationCreateDesc, AllocationScheme};
use slotmap::{SecondaryMap, SlotMap};
use std::ffi::CStr;
use std::marker::PhantomData;
use std::mem;
use std::sync::atomic::AtomicU64;
use std::time::Duration;
use tracing::{debug, error};

////////////////////////////////////////////////////////////////////////////////////////////////////

// FIXME: Mutexes are useless here since this is wrapped in Rc and can't be sent across threads.
//        Just use RefCells
pub struct Device {
    /// Underlying vulkan device
    pub(crate) raw: ash::Device,

    /// Platform-specific extension functions
    pub(crate) platform_extensions: PlatformExtensions,
    physical_device: vk::PhysicalDevice,
    allocator: Mutex<gpu_allocator::vulkan::Allocator>,

    // main graphics queue
    pub(crate) queue_family: u32,
    pub(crate) queue: vk::Queue,
    pub(crate) timeline: vk::Semaphore,

    // semaphores ready for reuse
    semaphores: RefCell<Vec<vk::Semaphore>>,

    // --- Extensions ---
    vk_khr_swapchain: ash::extensions::khr::Swapchain,
    vk_ext_shader_object: ash::extensions::ext::ShaderObject,
    vk_khr_push_descriptor: ash::extensions::khr::PushDescriptor,
    vk_ext_mesh_shader: ash::extensions::ext::MeshShader,
    vk_ext_extended_dynamic_state3: ash::extensions::ext::ExtendedDynamicState3,
    //vk_ext_descriptor_buffer: ash::extensions::ext::DescriptorBuffer,

    // physical device properties
    pub(crate) physical_device_memory_properties: vk::PhysicalDeviceMemoryProperties,
    _physical_device_descriptor_buffer_properties: vk::PhysicalDeviceDescriptorBufferPropertiesEXT,
    physical_device_properties: vk::PhysicalDeviceProperties2,

    // We don't need to hold strong refs here, we just need an ID for them.
    pub(crate) image_ids: Mutex<SlotMap<ImageId, ()>>,
    pub(crate) sampler_ids: Mutex<SlotMap<SamplerId, ()>>,
    pub(crate) buffer_ids: Mutex<SlotMap<BufferId, ()>>,
    pub(crate) image_view_ids: Mutex<SlotMap<ImageViewId, ()>>,

    // Command pools per queue and thread.
    free_command_pools: Mutex<Vec<CommandPool>>,

    // Next submission index
    pub(crate) next_submission_index: AtomicU64,
    // Next expected submission index
    pub(crate) expected_submission_index: Cell<u64>,

    /// Resources that have a zero user reference count and that should be ready for deletion soon,
    /// but we're waiting for the GPU to finish using them.
    dropped_resources: Mutex<Vec<(u64, Box<dyn DeleteLater>)>>,
    pub(crate) tracker: Mutex<DeviceTracker>,
    sampler_cache: Mutex<HashMap<SamplerCreateInfo, Sampler>>,

    pub(crate) texture_descriptors: Mutex<BindlessDescriptorTable>,
    pub(crate) image_descriptors: Mutex<BindlessDescriptorTable>,
    pub(crate) sampler_descriptors: Mutex<BindlessDescriptorTable>,
    //image_handles: Mutex<SlotMap<I>>
}

impl fmt::Debug for Device {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("DeviceInner").finish_non_exhaustive()
    }
}

pub type RcDevice = Rc<Device>;
pub type WeakDevice = Weak<Device>;

pub(crate) struct ActiveSubmission {
    pub(crate) index: u64,
    pub(crate) command_pools: Vec<CommandPool>,
}

pub(crate) struct DeviceTracker {
    pub(crate) active_submissions: VecDeque<ActiveSubmission>,
    pub(crate) writes: MemoryAccess,
    pub(crate) images: SecondaryMap<ImageId, MemoryAccess>,
}

impl DeviceTracker {
    fn new() -> DeviceTracker {
        DeviceTracker {
            active_submissions: VecDeque::new(),
            writes: MemoryAccess::empty(),
            images: SecondaryMap::new(),
        }
    }
}

/// Dummy trait for `Device::delete_later`
trait DeleteLater {}
impl<T> DeleteLater for T {}

/// Helper struct for deleting vulkan objects.
///
/// TODO doc
pub struct Defer<F: FnMut()>(F);

impl<F> Drop for Defer<F>
where
    F: FnMut(),
{
    fn drop(&mut self) {
        self.0()
    }
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
    /// Identifies a GPU resource.
    pub struct ImageId;

    /// Identifies a GPU resource.
    pub struct BufferId;

    /// Identifies a GPU resource.
    pub struct ImageViewId;

    /// Identifies a resource group.
    pub struct GroupId;

    pub struct SamplerId;
}

impl ImageViewId {
    pub(crate) fn index(&self) -> u32 {
        (self.0.as_ffi() & 0xFFFF_FFFF) as u32
    }
}

impl SamplerId {
    pub(crate) fn index(&self) -> u32 {
        (self.0.as_ffi() & 0xFFFF_FFFF) as u32
    }
}

/// Describes how a resource got its memory.
#[derive(Debug)]
pub enum ResourceAllocation {
    /// We allocated a block of memory exclusively for this resource.
    Allocation {
        allocation: gpu_allocator::vulkan::Allocation,
    },
    /// The memory for this resource was imported or exported from/to an external handle.
    DeviceMemory { device_memory: vk::DeviceMemory },

    /// We don't own the memory for this resource.
    External,
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
pub unsafe fn create_device_with_surface(
    present_surface: Option<vk::SurfaceKHR>,
) -> Result<RcDevice, DeviceCreateError> {
    let device = Device::with_surface(present_surface)?;
    Ok(device)
}

/// Creates a `Device`. A physical device is chosen automatically.
pub fn create_device() -> Result<RcDevice, DeviceCreateError> {
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
    vk_khr_surface: &ash::extensions::khr::Surface,
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
    //"VK_EXT_descriptor_buffer",
];

impl Device {
    fn find_compatible_memory_type_internal(
        &self,
        memory_type_bits: u32,
        memory_properties: vk::MemoryPropertyFlags,
    ) -> Option<u32> {
        for i in 0..self.physical_device_memory_properties.memory_type_count {
            if memory_type_bits & (1 << i) != 0
                && self.physical_device_memory_properties.memory_types[i as usize]
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
    ) -> Result<RcDevice, DeviceCreateError> {
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
        let vk_khr_swapchain = ash::extensions::khr::Swapchain::new(instance, &device);
        let vk_ext_shader_object = ash::extensions::ext::ShaderObject::new(instance, &device);
        let vk_khr_push_descriptor = ash::extensions::khr::PushDescriptor::new(instance, &device);
        let vk_ext_extended_dynamic_state3 = ash::extensions::ext::ExtendedDynamicState3::new(instance, &device);
        let vk_ext_mesh_shader = ash::extensions::ext::MeshShader::new(instance, &device);
        //let vk_ext_descriptor_buffer = ash::extensions::ext::DescriptorBuffer::new(instance, &device);
        let physical_device_memory_properties = instance.get_physical_device_memory_properties(physical_device);
        let platform_extensions = PlatformExtensions::load(entry, instance, &device);

        let mut physical_device_descriptor_buffer_properties =
            vk::PhysicalDeviceDescriptorBufferPropertiesEXT::default();
        let mut physical_device_properties = vk::PhysicalDeviceProperties2 {
            p_next: &mut physical_device_descriptor_buffer_properties as *mut _ as *mut c_void,
            ..Default::default()
        };

        instance.get_physical_device_properties2(physical_device, &mut physical_device_properties);

        // Create global descriptor tables
        let texture_descriptors = BindlessDescriptorTable::new(&device, vk::DescriptorType::SAMPLED_IMAGE, 4096);
        let image_descriptors = BindlessDescriptorTable::new(&device, vk::DescriptorType::STORAGE_IMAGE, 4096);
        let sampler_descriptors = BindlessDescriptorTable::new(&device, vk::DescriptorType::SAMPLER, 4096);

        Ok(Rc::new(Device {
            raw: device,
            platform_extensions,
            physical_device,
            physical_device_properties,
            _physical_device_descriptor_buffer_properties: physical_device_descriptor_buffer_properties,
            physical_device_memory_properties,
            queue,
            timeline,
            queue_family: graphics_queue_family_index,
            allocator: Mutex::new(allocator),
            vk_khr_swapchain,
            vk_ext_shader_object,
            vk_khr_push_descriptor,
            vk_ext_mesh_shader,
            vk_ext_extended_dynamic_state3,
            //vk_ext_descriptor_buffer,
            image_ids: Mutex::new(Default::default()),
            sampler_ids: Mutex::new(Default::default()),
            buffer_ids: Mutex::new(Default::default()),
            tracker: Mutex::new(DeviceTracker::new()),
            sampler_cache: Mutex::new(Default::default()),
            //compiler,
            free_command_pools: Mutex::new(Default::default()),
            image_view_ids: Mutex::new(Default::default()),
            dropped_resources: Mutex::new(vec![]),
            next_submission_index: AtomicU64::new(1),
            expected_submission_index: Cell::new(1),
            texture_descriptors: Mutex::new(texture_descriptors),
            image_descriptors: Mutex::new(image_descriptors),
            sampler_descriptors: Mutex::new(sampler_descriptors),
            semaphores: Default::default(),
        }))
    }

    /// Creates a new `Device`, automatically choosing a suitable physical device.
    pub fn new() -> Result<RcDevice, DeviceCreateError> {
        unsafe { Self::with_surface(None) }
    }

    /// Creates a new `Device` that can render to the specified `present_surface` if one is specified.
    ///
    /// Also creates queues as requested.
    pub unsafe fn with_surface(present_surface: Option<vk::SurfaceKHR>) -> Result<RcDevice, DeviceCreateError> {
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
            "Selected physical device: {:?}",
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

        let mut mesh_shader_features = vk::PhysicalDeviceMeshShaderFeaturesEXT {
            p_next: &mut fragment_shader_interlock_features as *mut _ as *mut c_void,
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
            shader_int8: vk::TRUE,
            scalar_block_layout: vk::TRUE,
            ..Default::default()
        };

        let mut features2 = vk::PhysicalDeviceFeatures2 {
            p_next: &mut vk12_features as *mut _ as *mut c_void,
            features: vk::PhysicalDeviceFeatures {
                tessellation_shader: vk::TRUE,
                fill_mode_non_solid: vk::TRUE,
                sampler_anisotropy: vk::TRUE,
                shader_int16: vk::TRUE,
                shader_int64: vk::TRUE,
                shader_storage_image_extended_formats: vk::TRUE,
                fragment_stores_and_atomics: vk::TRUE,
                depth_clamp: vk::TRUE,
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

    /// Returns the physical device that this device was created on.
    pub fn physical_device(&self) -> vk::PhysicalDevice {
        self.physical_device
    }

    /// Returns the physical device properties.
    pub fn physical_device_properties(&self) -> &vk::PhysicalDeviceProperties2 {
        &self.physical_device_properties
    }

    /*pub fn create_command_stream(self: &Rc<Self>, queue_index: usize) -> CommandStream {
        CommandStream::new(self.clone(), command_pool, self.queues[queue_index].clone())
    }*/
}

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
) -> Result<(vk::PipelineShaderStageCreateInfo, ShaderModuleGuard<'a>), Error> {
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
    pub fn raw(&self) -> &ash::Device {
        &self.raw
    }

    /// Function pointers for VK_KHR_swapchain.
    pub fn khr_swapchain(&self) -> &ash::extensions::khr::Swapchain {
        &self.vk_khr_swapchain
    }

    /// Function pointers for VK_KHR_push_descriptor.
    pub fn khr_push_descriptor(&self) -> &ash::extensions::khr::PushDescriptor {
        &self.vk_khr_push_descriptor
    }

    pub fn ext_extended_dynamic_state3(&self) -> &ash::extensions::ext::ExtendedDynamicState3 {
        &self.vk_ext_extended_dynamic_state3
    }

    pub fn ext_mesh_shader(&self) -> &ash::extensions::ext::MeshShader {
        &self.vk_ext_mesh_shader
    }

    pub fn ext_shader_object(&self) -> &ash::extensions::ext::ShaderObject {
        &self.vk_ext_shader_object
    }

    /*pub fn ext_descriptor_buffer(&self) -> &ash::extensions::ext::DescriptorBuffer {
        &self.inner.vk_ext_descriptor_buffer
    }*/

    /// Helper function to associate a debug name to a vulkan object.
    ///
    /// # Arguments
    /// * `handle` - the handle to the object
    /// * `name` - the name to associate with the object
    ///
    /// # Safety
    /// The handle must be a valid vulkan object handle.
    pub unsafe fn set_object_name<H: vk::Handle>(&self, handle: H, name: &str) {
        let object_name = CString::new(name).unwrap();
        vk_ext_debug_utils()
            .set_debug_utils_object_name(
                self.raw.handle(),
                &vk::DebugUtilsObjectNameInfoEXT {
                    object_type: H::TYPE,
                    object_handle: handle.as_raw(),
                    p_object_name: object_name.as_ptr(),
                    ..Default::default()
                },
            )
            .unwrap();
    }

    /*/// Increments the submission index.
    pub(crate) fn next_submission_index(&self) -> u64 {
        self.next_submission_index
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            + 1
    }*/

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // COMMAND STREAMS
    ////////////////////////////////////////////////////////////////////////////////////////////////

    /// Creates a command stream used to submit commands to the GPU.
    ///
    /// Once finished, the command stream should be submitted to the GPU using
    /// `CommandStream::flush`.
    pub fn create_command_stream(self: &Rc<Self>) -> CommandStream {
        CommandStream::new(self.clone())
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // BUFFERS
    ////////////////////////////////////////////////////////////////////////////////////////////////

    /// Creates a new buffer resource.
    pub fn create_buffer(
        self: &Rc<Self>,
        usage: BufferUsage,
        memory_location: MemoryLocation,
        byte_size: u64,
    ) -> BufferUntyped {
        assert!(byte_size > 0, "buffer size must be greater than zero");

        // create the buffer object first
        let create_info = vk::BufferCreateInfo {
            flags: Default::default(),
            size: byte_size,
            usage: usage.to_vk_buffer_usage_flags() | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            queue_family_index_count: 0,
            p_queue_family_indices: ptr::null(),
            ..Default::default()
        };
        let handle = unsafe {
            self.raw
                .create_buffer(&create_info, None)
                .expect("failed to create buffer")
        };

        // get its memory requirements
        let mem_req = unsafe { self.raw.get_buffer_memory_requirements(handle) };

        let allocation_create_desc = AllocationCreateDesc {
            name: "",
            requirements: mem_req,
            location: memory_location,
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        };
        let allocation = self
            .allocator
            .lock()
            .unwrap()
            .allocate(&allocation_create_desc)
            .expect("failed to allocate device memory");
        unsafe {
            self.raw
                .bind_buffer_memory(handle, allocation.memory(), allocation.offset())
                .unwrap();
        }
        let mapped_ptr = allocation.mapped_ptr();
        let allocation = ResourceAllocation::Allocation { allocation };

        let device_address = unsafe {
            self.raw.get_buffer_device_address(&vk::BufferDeviceAddressInfo {
                buffer: handle,
                ..Default::default()
            })
        };

        let id = self.buffer_ids.lock().unwrap().insert(());
        BufferUntyped {
            inner: Some(Arc::new(BufferInner {
                device: self.clone(),
                id,
                last_submission_index: AtomicU64::new(0),
                allocation,
                handle,
                memory_location,
                device_address,
            })),
            handle,
            size: byte_size,
            usage,
            mapped_ptr,
            _marker: PhantomData,
        }
    }

    /*/// Registers an existing buffer resource.
    pub(crate) unsafe fn register_buffer(
        &self,
        allocation: ResourceAllocation,
        handle: vk::Buffer,
    ) -> Arc<BufferInner> {
        let device_address = self.get_buffer_device_address(&vk::BufferDeviceAddressInfo {
            buffer: handle,
            ..Default::default()
        });

        let mut buffer_ids = self.inner.buffer_ids.lock().unwrap();
        let id = buffer_ids.insert_with_key(|id| {
            Arc::new(BufferInner {
                device: self.clone(),
                id,
                user_ref_count: AtomicU32::new(1),
                last_submission_index: AtomicU64::new(0),
                allocation,
                group: None,
                handle,
                device_address,
            })
        });
        buffer_ids.get(id).unwrap().clone()
    }*/

    pub(crate) fn register_swapchain_image(
        self: &Rc<Self>,
        handle: vk::Image,
        format: vk::Format,
        width: u32,
        height: u32,
    ) -> Image {
        let id = self.image_ids.lock().unwrap().insert(());
        let (bindless_handle, default_view) = self.create_bindless_image_view(handle, ImageType::Image2D, format, 1, 1);
        Image {
            inner: Some(Arc::new(ImageInner {
                device: self.clone(),
                id,
                last_submission_index: AtomicU64::new(0),
                allocation: ResourceAllocation::External,
                handle,
                swapchain_image: true,
                default_view,
                bindless_handle,
            })),
            handle,
            usage: ImageUsage::TRANSFER_DST | ImageUsage::COLOR_ATTACHMENT,
            type_: ImageType::Image2D,
            format,
            size: Size3D {
                width,
                height,
                depth: 1,
            },
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // IMAGES
    ////////////////////////////////////////////////////////////////////////////////////////////////

    /// Creates the default image view for the image.
    pub(crate) fn create_bindless_image_view(
        &self,
        handle: vk::Image,
        type_: ImageType,
        format: Format,
        mip_levels: u32,
        array_layers: u32,
    ) -> (ImageViewId, vk::ImageView) {
        unsafe {
            let image_view = self
                .raw
                .create_image_view(
                    &vk::ImageViewCreateInfo {
                        flags: vk::ImageViewCreateFlags::empty(),
                        image: handle,
                        view_type: match type_ {
                            ImageType::Image1D => {
                                if array_layers > 1 {
                                    vk::ImageViewType::TYPE_1D_ARRAY
                                } else {
                                    vk::ImageViewType::TYPE_1D
                                }
                            }
                            ImageType::Image2D => {
                                if array_layers > 1 {
                                    vk::ImageViewType::TYPE_2D_ARRAY
                                } else {
                                    vk::ImageViewType::TYPE_2D
                                }
                            }
                            ImageType::Image3D => vk::ImageViewType::TYPE_3D,
                        },
                        format,
                        components: vk::ComponentMapping {
                            r: vk::ComponentSwizzle::IDENTITY,
                            g: vk::ComponentSwizzle::IDENTITY,
                            b: vk::ComponentSwizzle::IDENTITY,
                            a: vk::ComponentSwizzle::IDENTITY,
                        },
                        subresource_range: vk::ImageSubresourceRange {
                            aspect_mask: aspects_for_format(format),
                            base_mip_level: 0,
                            level_count: mip_levels,
                            base_array_layer: 0,
                            layer_count: array_layers,
                        },
                        ..Default::default()
                    },
                    None,
                )
                .expect("failed to create image view");

            let id = self.image_view_ids.lock().unwrap().insert(());
            // Update the global descriptor table.
            //if usage.contains(ImageUsage::SAMPLED) {
            self.write_global_texture_descriptor(id, image_view);
            //}
            //if usage.contains(ImageUsage::STORAGE) {
            self.write_global_storage_image_descriptor(id, image_view);
            //}
            (id, image_view)
        }
    }

    /// Creates a new image resource.
    ///
    /// Returns an `ImageInfo` struct containing the image resource ID and the vulkan image handle.
    ///
    /// # Notes
    /// The image might not have any device memory attached when this function returns.
    /// This is because graal may delay the allocation and binding of device memory until the end of the
    /// current frame (see `Context::end_frame`).
    ///
    /// # Examples
    ///
    pub fn create_image(self: &Rc<Self>, image_info: &ImageCreateInfo) -> Image {
        let create_info = vk::ImageCreateInfo {
            image_type: image_info.type_.into(),
            format: image_info.format,
            extent: vk::Extent3D {
                width: image_info.width,
                height: image_info.height,
                depth: image_info.depth,
            },
            mip_levels: image_info.mip_levels,
            array_layers: image_info.array_layers,
            samples: get_vk_sample_count(image_info.samples),
            tiling: vk::ImageTiling::OPTIMAL, // LINEAR tiling not used enough to be exposed
            usage: image_info.usage.into(),
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            queue_family_index_count: 0,
            p_queue_family_indices: ptr::null(),
            ..Default::default()
        };
        let handle = unsafe {
            self.raw
                .create_image(&create_info, None)
                .expect("failed to create image")
        };
        let mem_req = unsafe { self.raw.get_image_memory_requirements(handle) };

        let allocation_create_desc = AllocationCreateDesc {
            name: "",
            requirements: mem_req,
            location: image_info.memory_location,
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        };
        let allocation = self
            .allocator
            .lock()
            .unwrap()
            .allocate(&allocation_create_desc)
            .expect("failed to allocate device memory");
        unsafe {
            self.raw
                .bind_image_memory(handle, allocation.memory(), allocation.offset() as u64)
                .unwrap();
        }

        let id = self.image_ids.lock().unwrap().insert(());

        // create the bindless image view
        let (bindless_handle, default_view) = self.create_bindless_image_view(
            handle,
            image_info.type_,
            image_info.format,
            image_info.mip_levels,
            image_info.array_layers,
        );

        Image {
            handle,
            inner: Some(Arc::new(ImageInner {
                device: self.clone(),
                id,
                last_submission_index: AtomicU64::new(0),
                allocation: ResourceAllocation::Allocation { allocation },
                handle,
                swapchain_image: false,
                default_view,
                bindless_handle,
            })),
            usage: image_info.usage,
            type_: image_info.type_,
            format: image_info.format,
            size: Size3D {
                width: image_info.width,
                height: image_info.height,
                depth: image_info.depth,
            },
        }
    }

    /*pub fn create_image_view(&self, image: &Image, info: &ImageViewInfo) -> ImageView {
        // FIXME: support non-zero base mip level
        if info.subresource_range.base_mip_level != 0 {
            unimplemented!("non-zero base mip level");
        }

        let create_info = vk::ImageViewCreateInfo {
            flags: vk::ImageViewCreateFlags::empty(),
            image: image.handle,
            view_type: info.view_type,
            format: info.format,
            components: vk::ComponentMapping {
                r: info.component_mapping[0],
                g: info.component_mapping[1],
                b: info.component_mapping[2],
                a: info.component_mapping[3],
            },
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: info.subresource_range.aspect_mask,
                base_mip_level: info.subresource_range.base_mip_level,
                level_count: info.subresource_range.level_count,
                base_array_layer: info.subresource_range.base_array_layer,
                layer_count: info.subresource_range.layer_count,
            },
            ..Default::default()
        };

        // SAFETY: the device is valid, the create info is valid
        let handle = unsafe {
            self.raw
                .create_image_view(&create_info, None)
                .expect("failed to create image view")
        };

        let id = self.image_view_ids.lock().unwrap().insert(());

        // Update the global descriptor table
        let usage = image.usage();
        unsafe {
            if usage.contains(ImageUsage::SAMPLED) {
                self.write_global_texture_descriptor(id, handle);
            }
            if usage.contains(ImageUsage::STORAGE) {
                self.write_global_storage_image_descriptor(id, handle);
            }
        }

        ImageView {
            inner: Some(Arc::new(ImageViewInner {
                image: image.clone(),
                id,
                handle,
                last_submission_index: AtomicU64::new(0),
            })),
            handle,
            format: info.format,
            // TODO: size of mip level
            size: image.size,
        }
    }*/

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // RESOURCE GROUPS
    ////////////////////////////////////////////////////////////////////////////////////////////////

    /*/// Creates a resource group.
    ///
    /// Resource group hold a set of static resources that can be synchronized with as a group.
    /// This is useful for large sets of long-lived static resources, like texture maps,
    /// where it would be impractical to synchronize on each of them individually.
    pub fn create_resource_group(
        &self,
        dst_stage_mask: vk::PipelineStageFlags2,
        dst_access_mask: vk::AccessFlags2,
    ) -> GroupId {
        // resource groups are for read-only resources
        assert!(!is_write_access(dst_access_mask));
        self.inner.groups.lock().unwrap().insert(ResourceGroup {
            //wait: Default::default(),
            src_stage_mask: Default::default(),
            dst_stage_mask,
            src_access_mask: Default::default(),
            dst_access_mask,
        })
    }*/

    // TODO: instead of passing the submission index, get it via a trait method on T (GpuResource)
    pub fn delete_later<T: 'static>(&self, submission_index: u64, object: T) {
        let last_completed_submission_index = unsafe { self.raw.get_semaphore_counter_value(self.timeline).unwrap() };
        if submission_index <= last_completed_submission_index {
            // drop the object immediately if the submission has completed
            return;
        }

        // otherwise move it to the deferred deletion list
        self.dropped_resources
            .lock()
            .unwrap()
            .push((submission_index, Box::new(object)));
    }

    pub fn call_later(&self, submission_index: u64, f: impl FnMut() + 'static) {
        self.delete_later(submission_index, Defer(f))
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

    // Cleanup expired resources.
    pub fn cleanup(&self) {
        let last_completed_submission_index = unsafe {
            self.raw
                .get_semaphore_counter_value(self.timeline)
                .expect("get_semaphore_counter_value failed")
        };

        let mut tracker = self.tracker.lock().unwrap();
        /*let mut image_ids = self.inner.image_ids.lock().unwrap();
        let mut buffer_ids = self.inner.buffer_ids.lock().unwrap();
        let mut image_view_ids = self.inner.image_view_ids.lock().unwrap();*/
        let mut dropped_resources = self.dropped_resources.lock().unwrap();

        dropped_resources.retain(|(submission, _object)| *submission > last_completed_submission_index);

        // process all completed submissions, oldest to newest
        //let mut active_submissions = tracker.active_submissions.lock().unwrap();
        let mut free_command_pools = self.free_command_pools.lock().unwrap();

        loop {
            let Some(submission) = tracker.active_submissions.front() else {
                break;
            };
            if submission.index > last_completed_submission_index {
                break;
            }
            debug!("cleaning up submission {}", submission.index);
            let submission = tracker.active_submissions.pop_front().unwrap();
            for mut command_pool in submission.command_pools {
                // SAFETY: command buffers are not in use anymore
                unsafe {
                    command_pool.reset(&self.raw);
                }
                free_command_pools.push(command_pool);
            }
        }
    }

    /// Creates a swapchain object.
    pub unsafe fn create_swapchain(
        self: &Rc<Self>,
        surface: vk::SurfaceKHR,
        format: vk::SurfaceFormatKHR,
        width: u32,
        height: u32,
    ) -> SwapChain {
        let mut swapchain = SwapChain {
            handle: Default::default(),
            surface,
            images: vec![],
            format,
            width,
            height,
        };
        self.resize_swapchain(&mut swapchain, width, height);
        swapchain
    }

    /// Creates a new, or returns an existing, binary semaphore that is in the unsignaled state,
    /// or for which we've submitted a wait operation on this queue and that will eventually be unsignaled.
    pub fn get_or_create_semaphore(&self) -> vk::Semaphore {
        // Try to recycle one
        if let Some(semaphore) = self.semaphores.borrow_mut().pop() {
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
        self.semaphores.borrow_mut().push(binary_semaphore);
    }

    /// Acquires the next image in a swapchain.
    ///
    /// Returns the image and the semaphore that will be signaled when the image is available.
    pub unsafe fn acquire_next_swapchain_image<'a>(
        &self,
        swapchain: &'a SwapChain,
        timeout: Duration,
    ) -> Result<(SwapchainImage<'a>, SignaledSemaphore), vk::Result> {
        // We can't use `get_or_create_semaphore` because according to the spec the semaphore
        // passed to `vkAcquireNextImage` must not have any pending operations, whereas
        // `get_or_create_semaphore` only guarantees that a wait operation has been submitted
        // on the semaphore (not that the wait has completed).
        let ready = {
            let create_info = vk::SemaphoreCreateInfo { ..Default::default() };
            self.raw.create_semaphore(&create_info, None).unwrap()
        };

        let (index, _suboptimal) = match self.khr_swapchain().acquire_next_image(
            swapchain.handle,
            timeout.as_nanos() as u64,
            ready,
            vk::Fence::null(),
        ) {
            Ok(result) => result,
            Err(err) => {
                // delete the semaphore before returning
                self.raw.destroy_semaphore(ready, None);
                return Err(err);
            }
        };

        let img = SwapchainImage {
            swapchain: swapchain.handle,
            image: &swapchain.images[index as usize].image,
            index,
            render_finished: swapchain.images[index as usize].render_finished.clone(),
        };

        Ok((img, SignaledSemaphore(ready)))
    }

    /// Returns the list of supported swapchain formats for the given surface.
    pub unsafe fn get_surface_formats(&self, surface: vk::SurfaceKHR) -> Vec<vk::SurfaceFormatKHR> {
        vk_khr_surface()
            .get_physical_device_surface_formats(self.physical_device, surface)
            .unwrap()
    }

    /// Returns one supported surface format. Use if you don't care about the format of your swapchain.
    pub unsafe fn get_preferred_surface_format(&self, surface: vk::SurfaceKHR) -> vk::SurfaceFormatKHR {
        let surface_formats = self.get_surface_formats(surface);
        get_preferred_swapchain_surface_format(&surface_formats)
    }

    /// Resizes a swapchain.
    pub unsafe fn resize_swapchain(self: &Rc<Self>, swapchain: &mut SwapChain, width: u32, height: u32) {
        let phy = self.physical_device;
        let capabilities = vk_khr_surface()
            .get_physical_device_surface_capabilities(phy, swapchain.surface)
            .unwrap();
        /*let formats = self
        .vk_khr_surface
        .get_physical_device_surface_formats(phy, swapchain.surface)
        .unwrap();*/
        let present_modes = vk_khr_surface()
            .get_physical_device_surface_present_modes(phy, swapchain.surface)
            .unwrap();

        let present_mode = get_preferred_present_mode(&present_modes);
        let image_extent = get_preferred_swap_extent((width, height), &capabilities);
        let image_count =
            if capabilities.max_image_count > 0 && capabilities.min_image_count + 1 > capabilities.max_image_count {
                capabilities.max_image_count
            } else {
                capabilities.min_image_count + 1
            };

        let create_info = vk::SwapchainCreateInfoKHR {
            flags: Default::default(),
            surface: swapchain.surface,
            min_image_count: image_count,
            image_format: swapchain.format.format,
            image_color_space: swapchain.format.color_space,
            image_extent,
            image_array_layers: 1,
            // TODO: this should be a parameter
            image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST,
            image_sharing_mode: vk::SharingMode::EXCLUSIVE,
            queue_family_index_count: 0,
            p_queue_family_indices: ptr::null(),
            pre_transform: vk::SurfaceTransformFlagsKHR::IDENTITY,
            // TODO: this should be a parameter
            composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
            present_mode,
            clipped: vk::TRUE,
            old_swapchain: swapchain.handle,
            ..Default::default()
        };

        let new_handle = self.vk_khr_swapchain.create_swapchain(&create_info, None).unwrap();
        if swapchain.handle != vk::SwapchainKHR::null() {
            // FIXME the images may be in use, we should wait for the device to be idle
            self.vk_khr_swapchain.destroy_swapchain(swapchain.handle, None);
        }

        swapchain.handle = new_handle;
        swapchain.width = width;
        swapchain.height = height;

        // reset images & semaphores
        for SwapchainImageInner { render_finished, .. } in swapchain.images.drain(..) {
            self.recycle_binary_semaphore(render_finished);
        }
        swapchain.images = Vec::with_capacity(image_count as usize);

        let images = self.vk_khr_swapchain.get_swapchain_images(swapchain.handle).unwrap();
        for image in images {
            let render_finished = self.get_or_create_semaphore();
            swapchain.images.push(SwapchainImageInner {
                image: self.register_swapchain_image(image, swapchain.format.format, width, height),
                render_finished,
            });
        }
    }

    pub fn create_sampler(self: &Rc<Self>, info: &SamplerCreateInfo) -> Sampler {
        if let Some(sampler) = self.sampler_cache.lock().unwrap().get(info) {
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
            mip_lod_bias: info.mip_lod_bias.0,
            anisotropy_enable: info.anisotropy_enable.into(),
            max_anisotropy: info.max_anisotropy.0,
            compare_enable: info.compare_enable.into(),
            compare_op: info.compare_op.into(),
            min_lod: info.min_lod.0,
            max_lod: info.max_lod.0,
            border_color: info.border_color,
            ..Default::default()
        };

        let sampler = unsafe {
            self.raw
                .create_sampler(&create_info, None)
                .expect("failed to create sampler")
        };
        let id = self.sampler_ids.lock().unwrap().insert(());
        unsafe {
            self.write_global_sampler_descriptor(id, sampler);
        }
        let sampler = Sampler {
            device: Rc::downgrade(self),
            id,
            sampler,
        };
        self.sampler_cache.lock().unwrap().insert(info.clone(), sampler.clone());
        sampler
    }

    pub(crate) fn get_or_create_command_pool(&self, queue_family: u32) -> CommandPool {
        let free_command_pools = &mut self.free_command_pools.lock().unwrap();
        let index = free_command_pools
            .iter()
            .position(|pool| pool.queue_family == queue_family);
        if let Some(index) = index {
            return free_command_pools.swap_remove(index);
        } else {
            unsafe { CommandPool::new(&self.raw, queue_family) }
        }
    }

    /// FIXME: this should be a constructor of `DescriptorSetLayout`, because now we have two
    /// functions with very similar names (`create_descriptor_set_layout` and `create_descriptor_set_layout_from_handle`)
    /// that have totally different semantics (one returns a raw vulkan handle, the other returns a RAII wrapper `DescriptorSetLayout`).
    pub fn create_descriptor_set_layout_from_handle(
        self: &Rc<Self>,
        handle: vk::DescriptorSetLayout,
    ) -> DescriptorSetLayout {
        DescriptorSetLayout {
            device: self.clone(),
            last_submission_index: Some(Arc::new(Default::default())),
            handle,
        }
    }

    pub fn create_push_descriptor_set_layout(
        self: &Rc<Self>,
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
            vec![
                self.texture_descriptors.lock().unwrap().layout,
                self.image_descriptors.lock().unwrap().layout,
                self.sampler_descriptors.lock().unwrap().layout,
            ]
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

    pub fn create_compute_pipeline(
        self: &Rc<Self>,
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
            device: self.clone(),
            pipeline,
            pipeline_layout,
            _descriptor_set_layouts: create_info.set_layouts.to_vec(),
            bindless: is_bindless,
        })
    }

    /// Creates a graphics pipeline.
    pub fn create_graphics_pipeline(
        self: &Rc<Self>,
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
        let mut dynamic_states = vec![vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];

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
            device: self.clone(),
            pipeline,
            pipeline_layout,
            _descriptor_set_layouts: create_info.set_layouts.to_vec(),
            bindless,
        })
    }
}
