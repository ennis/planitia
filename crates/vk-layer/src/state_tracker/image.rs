use ash::vk;
use crate::Device;
use crate::helper::HasPrivateData;

impl Device {
    pub unsafe fn hook_create_image_view(
        &self,
        device: vk::Device,
        p_create_info: *const vk::ImageViewCreateInfo<'_>,
        p_allocator: *const vk::AllocationCallbacks<'_>,
        p_view: *mut vk::ImageView,
    ) -> vk::Result {
        let result = (self.fp_v1_0().create_image_view)(device, p_create_info, p_allocator, p_view);
        if result == vk::Result::SUCCESS {
            let view = *p_view;
            self.set_private_data(view, ImageViewInfo { format: (*p_create_info).format });
        }
        result
    }

    pub unsafe fn hook_destroy_image_view(
        &self,
        device: vk::Device,
        image_view: vk::ImageView,
        p_allocator: *const vk::AllocationCallbacks<'_>,
    ) {
        self.take_private_data(image_view);
        (self.fp_v1_0().destroy_image_view)(device, image_view, p_allocator);
    }
}

/// Private data associated with a VkImageView.
pub struct ImageViewInfo {
    pub format: vk::Format,
}

impl HasPrivateData for vk::ImageView {
    type PrivateData = ImageViewInfo;
}
