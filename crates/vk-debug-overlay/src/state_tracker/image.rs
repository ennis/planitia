use crate::Device;
use crate::helper::HasPrivateData;
use ash::vk;
use ash::vk::Handle;

impl Device {
    pub unsafe fn hook_create_image_view(
        &self,
        device: VkDevice,
        p_create_info: *const VkImageViewCreateInfo<'_>,
        p_allocator: *const VkAllocationCallbacks<'_>,
        p_view: *mut VkImageView,
    ) -> VkResult {
        let result = (self.fp_v1_0().create_image_view)(device, p_create_info, p_allocator, p_view);
        if result == VK_SUCCESS {
            let view = *p_view;
            self.set_private_data(view, ImageViewInfo { format: (*p_create_info).format });
        }
        result
    }

    pub unsafe fn hook_destroy_image_view(
        &self,
        device: VkDevice,
        image_view: VkImageView,
        p_allocator: *const VkAllocationCallbacks<'_>,
    ) {
        self.take_private_data(image_view);
        (self.fp_v1_0().destroy_image_view)(device, image_view, p_allocator);
    }

    pub unsafe fn hook_create_image(
        &self,
        device: VkDevice,
        p_create_info: *const VkImageCreateInfo<'_>,
        p_allocator: *const VkAllocationCallbacks<'_>,
        p_image: *mut VkImage,
    ) -> VkResult {
        // Add TRANSFER_DST so that we can copy from images
        let mut create_info_copy = *p_create_info;
        create_info_copy.usage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;

        let result = (self.fp_v1_0().create_image)(device, &create_info_copy, p_allocator, p_image);

        if result == VK_SUCCESS {
            let image = *p_image;
            let image_info = ImageInfo {
                name: format!("Image_{:016x}", image.as_raw()),
                format: create_info_copy.format,
                usage: (*p_create_info).usage,
                ty_: create_info_copy.image_type,
                size: create_info_copy.extent,
                mip_count: create_info_copy.mip_levels,
                layer_count: create_info_copy.array_layers,
                samples: create_info_copy.samples,
            };
            self.set_private_data(image, image_info);
        }
        result
    }

    pub unsafe fn hook_destroy_image(
        &self,
        device: VkDevice,
        image: VkImage,
        p_allocator: *const VkAllocationCallbacks<'_>,
    ) {
        self.take_private_data(image);
        (self.fp_v1_0().destroy_image)(device, image, p_allocator);
    }

    pub unsafe fn hook_bind_image_memory(
        &self,
        device: VkDevice,
        image: VkImage,
        memory: VkDeviceMemory,
        memory_offset: VkDeviceSize,
    ) -> VkResult {
        (self.fp_v1_0().bind_image_memory)(device, image, memory, memory_offset)
    }

    pub unsafe fn hook_bind_image_memory_2(
        &self,
        device: VkDevice,
        bind_info_count: u32,
        p_bind_infos: *const VkBindImageMemoryInfo<'_>,
    ) -> VkResult {
        (self.fp_v1_1().bind_image_memory2)(device, bind_info_count, p_bind_infos)
    }
}

#[derive(Clone, Debug)]
pub struct ImageInfo {
    pub name: String,
    pub format: VkFormat,
    pub usage: VkImageUsageFlags,
    pub ty_: VkImageType,
    pub size: VkExtent3D,
    pub mip_count: u32,
    pub layer_count: u32,
    pub samples: VkSampleCountFlags,
}

impl HasPrivateData for VkImage {
    type PrivateData = ImageInfo;
}

/// Private data associated with a VkImageView.
#[derive(Clone, Debug)]
pub struct ImageViewInfo {
    pub format: VkFormat,
}

impl HasPrivateData for VkImageView {
    type PrivateData = ImageViewInfo;
}
