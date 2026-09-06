use crate::device::{ResourceAllocation, get_vk_sample_count};
use crate::{aspects_for_format, vk, vkcheck, CommandBuffer, Device, Image, ImageCreateInfo, Size3D};
//use ash::vk::{HANDLE, SECURITY_ATTRIBUTES};
use gpu_allocator::MemoryLocation;
use std::ffi::{OsStr, c_void};
use std::mem::MaybeUninit;
use std::ptr;
use std::rc::Rc;
use vulkan::*;

fn handle_name_to_wstr(name: Option<&str>) -> (Vec<u16>, *const u16) {
    use std::os::windows::ffi::OsStrExt;
    if let Some(name) = name {
        let mut w_name: Vec<u16> = OsStr::new(name).encode_wide().collect();
        w_name.push(0);
        let ptr = w_name.as_ptr();
        (w_name, ptr)
    } else {
        (Vec::new(), ptr::null())
    }
}

pub(crate) enum DedicatedAllocation {
    //Buffer(VkBuffer),
    Image(VkImage),
}

unsafe fn import_external_memory(
    device: &Device,
    memory_requirements: &VkMemoryRequirements,
    required_flags: VkMemoryPropertyFlags,
    preferred_flags: VkMemoryPropertyFlags,
    handle_type: VkExternalMemoryHandleTypeFlags,
    handle: HANDLE,
    handle_name: Option<&str>,
    dedicated: Option<DedicatedAllocation>,
) -> VkDeviceMemory {
    // TODO proper error handling
    let mut win32_handle_properties = vk::MemoryWin32HandlePropertiesKHR::default();
    device
        .platform_extensions
        .khr_external_memory_win32
        .get_memory_win32_handle_properties(handle_type, handle, &mut win32_handle_properties)
        .expect("vkGetMemoryWin32HandlePropertiesKHR failed");
    // find a memory type that both matches the resource requirement and the external handle requirements for importing
    let memory_type_bits = memory_requirements.memory_type_bits & win32_handle_properties.memory_type_bits;
    let memory_type_index = device
        .find_compatible_memory_type(memory_type_bits, required_flags, preferred_flags)
        .expect("could not find a compatible memory type for importing external memory");
    let (_, handle_name_wstr) = handle_name_to_wstr(handle_name);
    let mut dedicated_allocate_info: vk::MemoryDedicatedAllocateInfo;
    let mut p_dedicated_allocate_info = ptr::null();
    if let Some(dedicated) = dedicated {
        dedicated_allocate_info = vk::MemoryDedicatedAllocateInfo {
            image: Default::default(),
            buffer: Default::default(),
            ..Default::default()
        };
        match dedicated {
            //DedicatedAllocation::Buffer(buffer) => {
            //    dedicated_allocate_info.buffer = buffer;
            //}
            DedicatedAllocation::Image(image) => {
                dedicated_allocate_info.image = image;
            }
        }
        p_dedicated_allocate_info = &dedicated_allocate_info as *const _ as *const c_void;
    }
    let import_memory_win32_handle_info = vk::ImportMemoryWin32HandleInfoKHR {
        p_next: p_dedicated_allocate_info,
        handle_type,
        handle,
        name: handle_name_wstr,
        ..Default::default()
    };
    let memory_allocate_info = vk::MemoryAllocateInfo {
        p_next: &import_memory_win32_handle_info as *const _ as *const c_void,
        allocation_size: memory_requirements.size,
        memory_type_index,
        ..Default::default()
    };
    let device_memory = device.raw.allocate_memory(&memory_allocate_info, None).unwrap();
    device_memory
}

impl Device {
    pub unsafe fn create_imported_image_win32(
        &self,
        image_info: &ImageCreateInfo,
        required_memory_flags: vk::MemoryPropertyFlags,
        preferred_memory_flags: vk::MemoryPropertyFlags,
        win32_handle_type: vk::ExternalMemoryHandleTypeFlags,
        win32_handle: HANDLE,
        win32_handle_name: Option<&str>,
    ) -> Image {
        let external_memory_image_create_info =
            vk::ExternalMemoryImageCreateInfo { handle_types: win32_handle_type, ..Default::default() };
        let create_info = vk::ImageCreateInfo {
            p_next: &external_memory_image_create_info as *const _ as *const c_void,
            image_type: image_info.type_.to_vk_image_type(),
            format: image_info.format,
            extent: vk::Extent3D { width: image_info.width, height: image_info.height, depth: image_info.depth },
            mip_levels: image_info.mip_levels,
            array_layers: image_info.array_layers,
            samples: get_vk_sample_count(image_info.samples),
            tiling: VK_IMAGE_TILING_OPTIMAL,
            usage: image_info.usage.to_vk_image_usage_flags(),
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            queue_family_index_count: 0,
            p_queue_family_indices: ptr::null(),
            initial_layout: vk::ImageLayout::UNDEFINED,
            ..Default::default()
        };
        let handle = self.raw.create_image(&create_info, None).expect("failed to create image");
        let mem_req = self.raw.get_image_memory_requirements(handle);
        let device_memory = import_external_memory(
            self,
            &mem_req,
            required_memory_flags,
            preferred_memory_flags,
            win32_handle_type,
            win32_handle,
            win32_handle_name,
            Some(DedicatedAllocation::Image(handle)),
        );
        self.raw.bind_image_memory(handle, device_memory, 0).unwrap();
        let descriptors = self.register_image_descriptors(handle, &create_info);
        let attachment_view = self.create_attachment_image_view(handle, image_info.format);
        // transition image to GENERAL
        {
            let mut cmd = CommandBuffer::new();
            cmd.image_barrier(&vk::ImageMemoryBarrier2 {
                src_stage_mask: vk::PipelineStageFlags2::NONE,
                src_access_mask: vk::AccessFlags2::MEMORY_WRITE,
                dst_stage_mask: vk::PipelineStageFlags2::ALL_COMMANDS,
                dst_access_mask: vk::AccessFlags2::MEMORY_READ,
                old_layout: vk::ImageLayout::UNDEFINED,
                new_layout: vk::ImageLayout::GENERAL,
                src_queue_family_index: vk::QUEUE_FAMILY_EXTERNAL,
                dst_queue_family_index: self.queue_family,
                image: handle,
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: aspects_for_format(image_info.format),
                    base_mip_level: 0,
                    level_count: image_info.mip_levels,
                    base_array_layer: 0,
                    layer_count: image_info.array_layers,
                },
                ..Default::default()
            });
            crate::submit(cmd);
        }

        Image {
            handle,
            attachment_view,
            memory_location: MemoryLocation::Unknown,
            allocation: ResourceAllocation::DeviceMemory { device_memory },
            swapchain_image: false,
            descriptors,
            usage: image_info.usage,
            type_: image_info.type_,
            format: image_info.format,
            mip_levels: image_info.mip_levels,
            array_layers: image_info.array_layers,
            size: Size3D { width: image_info.width, height: image_info.height, depth: image_info.depth },
            samples: 0,
        }
    }

    pub unsafe fn create_exported_image_win32(
        &self,
        memory_location: MemoryLocation,
        image_info: &ImageCreateInfo,
        handle_type: VkExternalMemoryHandleTypeFlags,
        security_attributes: *const SECURITY_ATTRIBUTES,
        access_flags: u32,
        handle_name: Option<&str>,
    ) -> (Image, HANDLE) {
        let external_memory_image_create_info =
            VkExternalMemoryImageCreateInfo { handleTypes: handle_type, ..Default::default() };
        let create_info = VkImageCreateInfo {
            pNext: &external_memory_image_create_info as *const _ as *const c_void,
            imageType: image_info.type_.to_vk_image_type(),
            format: image_info.format,
            extent: VkExtent3D { width: image_info.width, height: image_info.height, depth: image_info.depth },
            mipLevels: image_info.mip_levels,
            arrayLayers: image_info.array_layers,
            samples: get_vk_sample_count(image_info.samples),
            tiling: VK_IMAGE_TILING_OPTIMAL,
            usage: image_info.usage.to_vk_image_usage_flags(),
            sharingMode: VK_SHARING_MODE_EXCLUSIVE,
            queueFamilyIndexCount: 0,
            pQueueFamilyIndices: ptr::null(),
            ..Default::default()
        };
        let handle = self.vk.CreateImage(self.vkd, &create_info, ptr::null()).expect("failed to create image");
        let mem_req = {
            let mut req = MaybeUninit::uninit();
            self.vk.GetImageMemoryRequirements(self.vkd, handle, req.as_mut_ptr());
            req.assume_init()
        };
        let (_, handle_name_wstr) = handle_name_to_wstr(handle_name);
        let (required_memory_properties, preferred_memory_properties) = match memory_location {
            MemoryLocation::Unknown => Default::default(),
            MemoryLocation::GpuOnly => (VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT),
            MemoryLocation::CpuToGpu => (
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                    | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
                    | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            ),
            MemoryLocation::GpuToCpu => (
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                    | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
                    | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            ),
        };
        let memory_type_index = self
            .find_compatible_memory_type(
                mem_req.memory_type_bits,
                required_memory_properties,
                preferred_memory_properties,
            )
            .expect("could not find a compatible memory type for exporting memory");
        let win32_handle_info = VkExportMemoryWin32HandleInfoKHR {
            pAttributes: security_attributes,
            dwAccess: access_flags,
            name: handle_name_wstr,
            ..Default::default()
        };
        let export_memory_allocate_info = VkExportMemoryAllocateInfo {
            pNext: &win32_handle_info as *const _ as *const c_void,
            handleTypes: handle_type,
            ..Default::default()
        };
        let memory_allocate_info = VkMemoryAllocateInfo {
            pNext: &export_memory_allocate_info as *const _ as *const c_void,
            allocationSize: mem_req.size,
            memoryTypeIndex: memory_type_index,
            ..Default::default()
        };
        let device_memory =
            self.vk.AllocateMemory(self.vkd, &memory_allocate_info, ptr::null()).expect("failed to allocate exported memory");
        // retrieve the win32 handle
        let get_win32_handle_info =
            VkMemoryGetWin32HandleInfoKHR { memory: device_memory, handleType: handle_type, ..Default::default() };
        // TODO proper error handling
        let win32_handle = self
            .platform_extensions
            .khr_external_memory_win32
            .GetMemoryWin32HandleKHR(self.vkd, &get_win32_handle_info)
            .unwrap();
        self.vk.BindImageMemory(self.vkd, handle, device_memory, 0).check();
        let descriptors = self.register_image_descriptors(handle, &create_info);
        let attachment_view = self.create_attachment_image_view(handle, image_info.format);
        let image = Image {
            handle,
            attachment_view,
            memory_location,
            allocation: ResourceAllocation::DeviceMemory { device_memory },
            swapchain_image: false,
            descriptors,
            usage: image_info.usage,
            type_: image_info.type_,
            format: image_info.format,
            mip_levels: image_info.mip_levels,
            array_layers: image_info.array_layers,
            size: Size3D { width: image_info.width, height: image_info.height, depth: image_info.depth },
            samples: image_info.samples,
        };
        (image, win32_handle)
    }

    pub unsafe fn create_exported_semaphore_win32(
        &self,
        handle_type: VkExternalSemaphoreHandleTypeFlags,
        security_attributes: *const SECURITY_ATTRIBUTES,
        access_flags: u32,
        handle_name: Option<&str>,
    ) -> (VkSemaphore, HANDLE) {
        let (_, handle_name_wstr) = handle_name_to_wstr(handle_name);
        let export_semaphore_win32_handle_info = VkExportSemaphoreWin32HandleInfoKHR {
            pAttributes: security_attributes,
            dwAccess: access_flags,
            name: handle_name_wstr,
            ..Default::default()
        };
        let export_semaphore_create_info = VkExportSemaphoreCreateInfo {
            pNext: &export_semaphore_win32_handle_info as *const _ as *const c_void,
            handleTypes: handle_type,
            ..
        };
        let semaphore_create_info = VkSemaphoreCreateInfo {
            pNext: &export_semaphore_create_info as *const _ as *const c_void,
            ..
        };
        let semaphore = self.vk.CreateSemaphore(self.vkd, &semaphore_create_info, ptr::null()).unwrap();
        let get_win32_handle_info = VkSemaphoreGetWin32HandleInfoKHR { semaphore, handleType: handle_type, ..Default::default() };
        let handle = self
            .platform_extensions
            .khr_external_semaphore_win32
            .GetSemaphoreWin32HandleKHR(self.vkd, &get_win32_handle_info)
            .unwrap();
        (semaphore, handle)
    }

    pub unsafe fn create_imported_semaphore_win32(
        &self,
        import_flags: VkSemaphoreImportFlags,
        handle_type: VkExternalSemaphoreHandleTypeFlags,
        handle: HANDLE,
        handle_name: Option<&str>,
    ) -> VkSemaphore {
        let (_, handle_name_wstr) = handle_name_to_wstr(handle_name);
        let is_timeline = match handle_type {
            VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_FENCE_BIT => true,
            _ => panic!("unsupported external semaphore type"),
        };
        let timeline_create_info = VkSemaphoreTypeCreateInfo {
            semaphoreType: VK_SEMAPHORE_TYPE_TIMELINE,
            initialValue: 0,
            ..
        };
        let semaphore_create_info = VkSemaphoreCreateInfo {
            pNext: if is_timeline { &timeline_create_info as *const _ as *const c_void } else { ptr::null() },
            ..
        };
        let semaphore = self.vk.CreateSemaphore(self.vkd, &semaphore_create_info, ptr::null()).unwrap();
        let import_semaphore_win32_handle_info = VkImportSemaphoreWin32HandleInfoKHR {
            semaphore,
            flags: import_flags, // ?????
            handleType,
            handle,
            name: handle_name_wstr,
            ..Default::default()
        };
        self.platform_extensions
            .khr_external_semaphore_win32
            .ImportSemaphoreWin32HandleKHR(&import_semaphore_win32_handle_info)
            .expect("vkImportSemaphoreWin32HandleKHR failed");
        semaphore
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

const PLATFORM_DEVICE_EXTENSIONS: &[&str] = &["VK_KHR_external_memory_win32", "VK_KHR_external_semaphore_win32"];

/// Windows-specific vulkan extensions
pub struct PlatformExtensions {
    pub khr_external_memory_win32: khr_external_memory_win32::DeviceDispatch,
    pub khr_external_semaphore_win32: khr_external_semaphore_win32::DeviceDispatch,
}

impl PlatformExtensions {
    pub(crate) fn names() -> &'static [&'static str] {
        PLATFORM_DEVICE_EXTENSIONS
    }

    pub(crate) fn load(_entry: &ash::Entry, instance: &ash::Instance, device: &ash::Device) -> PlatformExtensions {
        let khr_external_memory_win32 = ash::khr::external_memory_win32::Device::new(instance, device);
        let khr_external_semaphore_win32 = ash::khr::external_semaphore_win32::Device::new(instance, device);
        PlatformExtensions { khr_external_memory_win32, khr_external_semaphore_win32 }
    }
}
