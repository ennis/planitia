use crate::device::{ResourceAllocation, get_vk_sample_count};
use crate::{CommandBuffer, Device, Image, ImageCreateInfo, Size3D, aspects_for_format, vkcallnc};
//use ash::vk::{HANDLE, SECURITY_ATTRIBUTES};
use gpu::vkcall;
use gpu_allocator::MemoryLocation;
use std::ffi::{OsStr, c_void};
use std::ptr;
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
    let mut win32_handle_properties = VkMemoryWin32HandlePropertiesKHR { .. };
    vkcall!(device.fns.GetMemoryWin32HandlePropertiesKHR(
        device.vkd,
        handle_type,
        handle,
        &mut win32_handle_properties
    ));
    // find a memory type that both matches the resource requirement and the external handle requirements for importing
    let memory_type_bits = memory_requirements.memoryTypeBits & win32_handle_properties.memoryTypeBits;
    let memory_type_index = device
        .find_compatible_memory_type(memory_type_bits, required_flags, preferred_flags)
        .expect("could not find a compatible memory type for importing external memory");
    let (_, handle_name_wstr) = handle_name_to_wstr(handle_name);
    let mut dedicated_allocate_info: VkMemoryDedicatedAllocateInfo;
    let mut p_dedicated_allocate_info = ptr::null();
    if let Some(dedicated) = dedicated {
        dedicated_allocate_info =
            VkMemoryDedicatedAllocateInfo { image: Default::default(), buffer: Default::default(), .. };
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
    let import_memory_win32_handle_info = VkImportMemoryWin32HandleInfoKHR {
        pNext: p_dedicated_allocate_info,
        handleType: handle_type,
        handle,
        name: handle_name_wstr,
        ..
    };
    let memory_allocate_info = VkMemoryAllocateInfo {
        pNext: &import_memory_win32_handle_info as *const _ as *const c_void,
        allocationSize: memory_requirements.size,
        memoryTypeIndex: memory_type_index,
        ..
    };

    vkcall!(device.fns.AllocateMemory(device.vkd, &memory_allocate_info, ptr::null(), @out let device_memory));
    device_memory
}

impl Device {
    pub unsafe fn create_imported_image_win32(
        &self,
        image_info: &ImageCreateInfo,
        required_memory_flags: VkMemoryPropertyFlags,
        preferred_memory_flags: VkMemoryPropertyFlags,
        win32_handle_type: VkExternalMemoryHandleTypeFlags,
        win32_handle: HANDLE,
        win32_handle_name: Option<&str>,
    ) -> Image {
        let external_memory_image_create_info = VkExternalMemoryImageCreateInfo { handleTypes: win32_handle_type, .. };
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
            initialLayout: VK_IMAGE_LAYOUT_UNDEFINED,
            ..
        };
        vkcall!(self.fns.CreateImage(self.vkd, &create_info, ptr::null(), @out let handle));
        vkcallnc!(self.fns.GetImageMemoryRequirements(self.vkd, handle, @out let mem_req));
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
        vkcall!(self.fns.BindImageMemory(self.vkd, handle, device_memory, 0));
        let descriptors = self.register_image_descriptors(handle, &create_info);
        let attachment_view = self.create_attachment_image_view(handle, image_info.format);
        // transition image to GENERAL
        {
            let mut cmd = CommandBuffer::new();
            cmd.image_barrier(&VkImageMemoryBarrier2 {
                srcStageMask: 0,
                srcAccessMask: VK_ACCESS_2_MEMORY_WRITE_BIT,
                dstStageMask: VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                dstAccessMask: VK_ACCESS_2_MEMORY_READ_BIT,
                oldLayout: VK_IMAGE_LAYOUT_UNDEFINED,
                newLayout: VK_IMAGE_LAYOUT_GENERAL,
                srcQueueFamilyIndex: VK_QUEUE_FAMILY_EXTERNAL,
                dstQueueFamilyIndex: self.queue_family,
                image: handle,
                subresourceRange: VkImageSubresourceRange {
                    aspectMask: aspects_for_format(image_info.format),
                    baseMipLevel: 0,
                    levelCount: image_info.mip_levels,
                    baseArrayLayer: 0,
                    layerCount: image_info.array_layers,
                },
                ..
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
        let external_memory_image_create_info = VkExternalMemoryImageCreateInfo { handleTypes: handle_type, .. };
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
            ..
        };
        vkcall!(self.fns.CreateImage(self.vkd, &create_info, ptr::null(), @out let handle));
        vkcallnc!(self.fns.GetImageMemoryRequirements(self.vkd, handle, @out let mem_req));
        //let handle = self.fns.CreateImage(self.vkd, &create_info, ptr::null()).expect("failed to create image");
        //let mem_req = self.fns.GetImageMemoryRequirements(self.vkd, handle);
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
                mem_req.memoryTypeBits,
                required_memory_properties,
                preferred_memory_properties,
            )
            .expect("could not find a compatible memory type for exporting memory");
        let win32_handle_info = VkExportMemoryWin32HandleInfoKHR {
            pAttributes: security_attributes,
            dwAccess: access_flags,
            name: handle_name_wstr,
            ..
        };
        let export_memory_allocate_info = VkExportMemoryAllocateInfo {
            pNext: &win32_handle_info as *const _ as *const c_void,
            handleTypes: handle_type,
            ..
        };
        let memory_allocate_info = VkMemoryAllocateInfo {
            pNext: &export_memory_allocate_info as *const _ as *const c_void,
            allocationSize: mem_req.size,
            memoryTypeIndex: memory_type_index,
            ..
        };
        vkcall!(self.fns.AllocateMemory(self.vkd, &memory_allocate_info, ptr::null(), @out let device_memory));
        //let device_memory = self
        //    .fns
        //    .AllocateMemory(self.vkd, &memory_allocate_info, ptr::null())
        //    .expect("failed to allocate exported memory");
        // retrieve the win32 handle
        let get_win32_handle_info =
            VkMemoryGetWin32HandleInfoKHR { memory: device_memory, handleType: handle_type, .. };
        // TODO proper error handling
        vkcall!(self.fns.GetMemoryWin32HandleKHR(self.vkd, &get_win32_handle_info, @out let win32_handle));
        //let win32_handle = self
        //    .platform_extensions
        //    .khr_external_memory_win32
        //    .GetMemoryWin32HandleKHR(self.vkd, &get_win32_handle_info)
        //    .unwrap();
        vkcall!(self.fns.BindImageMemory(self.vkd, handle, device_memory, 0));
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
            ..
        };
        let export_semaphore_create_info = VkExportSemaphoreCreateInfo {
            pNext: &export_semaphore_win32_handle_info as *const _ as *const c_void,
            handleTypes: handle_type,
            ..
        };
        let semaphore_create_info =
            VkSemaphoreCreateInfo { pNext: &export_semaphore_create_info as *const _ as *const c_void, .. };
        vkcall!(self.fns.CreateSemaphore(self.vkd, &semaphore_create_info, ptr::null(), @out let semaphore));
        let get_win32_handle_info = VkSemaphoreGetWin32HandleInfoKHR { semaphore, handleType: handle_type, .. };
        vkcall!(self.fns.GetSemaphoreWin32HandleKHR(self.vkd, &get_win32_handle_info, @out let handle));
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
        let timeline_create_info =
            VkSemaphoreTypeCreateInfo { semaphoreType: VK_SEMAPHORE_TYPE_TIMELINE, initialValue: 0, .. };
        let semaphore_create_info = VkSemaphoreCreateInfo {
            pNext: if is_timeline { &timeline_create_info as *const _ as *const c_void } else { ptr::null() },
            ..
        };
        vkcall!(self.fns.CreateSemaphore(self.vkd, &semaphore_create_info, ptr::null(), @out let semaphore));
        let import_semaphore_win32_handle_info = VkImportSemaphoreWin32HandleInfoKHR {
            semaphore,
            flags: import_flags, // ?????
            handleType: handle_type,
            handle,
            name: handle_name_wstr,
            ..
        };
        vkcall!(
            self.fns.ImportSemaphoreWin32HandleKHR(self.vkd, &import_semaphore_win32_handle_info)
        );
        semaphore
    }
}

