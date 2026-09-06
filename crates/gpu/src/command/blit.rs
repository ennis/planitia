//! Blit commands
//!
//! Wrappers for clear/blit/copy commands on buffers and images.
use ash::vk;

use crate::{
    BufferRangeUntyped, BufferUntyped, ClearColorValue, CommandBuffer, Device, Image, ImageCopyBuffer, ImageCopyView,
    ImageSubresourceLayers, Rect3D, Size3D,
};
use vulkan::*;

impl CommandBuffer {
    pub fn fill_buffer(&mut self, range: &BufferRangeUntyped, data: u32) {
        let device = Device::instance();
        unsafe {
            // SAFETY: FFI call and parameters are valid
            device.vk.CmdFillBuffer(self.cmdbuf, range.buffer.handle(), range.byte_offset, range.byte_size, data);
        }
    }

    // TODO specify subresources
    pub fn clear_image(&mut self, image: &Image, clear_color_value: ClearColorValue) {
        let device = Device::instance();
        static COLOR_SUBRESOURCES: &[VkImageSubresourceRange] = &[VkImageSubresourceRange {
            aspectMask: VK_IMAGE_ASPECT_COLOR_BIT,
            baseMipLevel: 0,
            levelCount: vk::REMAINING_MIP_LEVELS,
            baseArrayLayer: 0,
            layerCount: vk::REMAINING_ARRAY_LAYERS,
        }];
        unsafe {
            // SAFETY: FFI call and parameters are valid
            device.vk.CmdClearColorImage(
                self.cmdbuf,
                image.handle(),
                VK_IMAGE_LAYOUT_GENERAL,
                &clear_color_value.into(),
                COLOR_SUBRESOURCES.len() as u32,
                COLOR_SUBRESOURCES.as_ptr(),
            );
        }
    }

    pub fn clear_depth_image(&mut self, image: &Image, depth: f32) {
        let device = Device::instance();
        static DEPTH_SUBRESOURCES: &[VkImageSubresourceRange] = &[VkImageSubresourceRange {
            aspectMask: VK_IMAGE_ASPECT_DEPTH_BIT,
            baseMipLevel: 0,
            levelCount: vk::REMAINING_MIP_LEVELS,
            baseArrayLayer: 0,
            layerCount: vk::REMAINING_ARRAY_LAYERS,
        }];
        unsafe {
            // SAFETY: FFI call and parameters are valid
            device.vk.CmdClearDepthStencilImage(
                self.cmdbuf,
                image.handle(),
                VK_IMAGE_LAYOUT_GENERAL,
                &VkClearDepthStencilValue { depth, stencil: 0 },
                DEPTH_SUBRESOURCES.len() as u32,
                DEPTH_SUBRESOURCES.as_ptr(),
            );
        }
    }

    pub fn copy_image_to_image(
        &mut self,
        source: ImageCopyView<'_>,
        destination: ImageCopyView<'_>,
        copy_size: VkExtent3D,
    ) {
        let device = Device::instance();
        // TODO: this is not required for multi-planar formats
        assert_eq!(source.aspect, destination.aspect);
        let regions = [VkImageCopy {
            srcSubresource: VkImageSubresourceLayers {
                aspectMask: source.aspect.to_aspect(source.image.format),
                mipLevel: source.mip_level,
                baseArrayLayer: 0,
                layerCount: 1,
            },
            srcOffset: source.origin.into(),
            dstSubresource: VkImageSubresourceLayers {
                aspectMask: destination.aspect.to_aspect(destination.image.format),
                mipLevel: destination.mip_level,
                baseArrayLayer: 0,
                layerCount: 1,
            },
            dstOffset: destination.origin.into(),
            extent: copy_size,
        }];
        // SAFETY: FFI call and parameters are valid
        unsafe {
            device.vk.CmdCopyImage(
                self.cmdbuf,
                source.image.handle(),
                VK_IMAGE_LAYOUT_GENERAL,
                destination.image.handle(),
                VK_IMAGE_LAYOUT_GENERAL,
                regions.len() as u32,
                regions.as_ptr(),
            );
        }
    }

    /// Copies data from one buffer to another.
    pub fn copy_buffer(
        &mut self,
        source: &BufferUntyped,
        src_offset: u64,
        destination: &BufferUntyped,
        dst_offset: u64,
        size: u64,
    ) {
        let device = Device::instance();
        assert!(src_offset + size <= source.byte_size());
        assert!(dst_offset + size <= destination.byte_size());
        // SAFETY: FFI call and parameters are valid
        unsafe {
            device.vk.CmdCopyBuffer(
                self.cmdbuf,
                source.handle(),
                destination.handle(),
                1,
                &VkBufferCopy { srcOffset: src_offset, dstOffset: dst_offset, size },
            );
        }
    }

    /// Copies data from a buffer to an image.
    ///
    /// TODO copy to layer other than 0
    pub fn copy_buffer_to_image(
        &mut self,
        source: ImageCopyBuffer<'_>,
        destination: ImageCopyView<'_>,
        copy_size: VkExtent3D,
    ) {
        let device = Device::instance();
        let regions = [VkBufferImageCopy {
            bufferOffset: source.layout.offset,
            bufferRowLength: source.layout.texel_row_length.unwrap_or(0),
            bufferImageHeight: source.layout.row_count.unwrap_or(0),
            imageSubresource: VkImageSubresourceLayers {
                aspectMask: destination.aspect.to_aspect(destination.image.format),
                mipLevel: destination.mip_level,
                baseArrayLayer: 0,
                layerCount: 1,
            },
            imageOffset: VkOffset3D { x: destination.origin.x, y: destination.origin.y, z: destination.origin.z },
            imageExtent: copy_size,
        }];
        // SAFETY: FFI call and parameters are valid
        unsafe {
            device.vk.CmdCopyBufferToImage(
                self.cmdbuf,
                source.buffer.handle(),
                destination.image.handle(),
                VK_IMAGE_LAYOUT_GENERAL,
                regions.len() as u32,
                regions.as_ptr(),
            );
        }
    }

    /// Copies data from an image to a buffer.
    pub fn copy_image_to_buffer(
        &mut self,
        source: ImageCopyView<'_>,
        destination: ImageCopyBuffer<'_>,
        copy_size: Size3D,
    ) {
        let device = Device::instance();
        let regions = [VkBufferImageCopy {
            bufferOffset: destination.layout.offset,
            bufferRowLength: destination.layout.texel_row_length.unwrap_or(0),
            bufferImageHeight: destination.layout.row_count.unwrap_or(0),
            imageSubresource: VkImageSubresourceLayers {
                aspectMask: source.aspect.to_aspect(source.image.format),
                mipLevel: source.mip_level,
                baseArrayLayer: 0,
                layerCount: 1,
            },
            imageOffset: source.origin.into(),
            imageExtent: copy_size.into(),
        }];
        // SAFETY: FFI call and parameters are valid
        unsafe {
            device.vk.CmdCopyImageToBuffer(
                self.cmdbuf,
                source.image.handle(),
                VK_IMAGE_LAYOUT_GENERAL,
                destination.buffer.handle(),
                regions.len() as u32,
                regions.as_ptr(),
            );
        }
    }

    pub fn blit_image(
        &mut self,
        src: &Image,
        src_subresource: ImageSubresourceLayers,
        src_region: Rect3D,
        dst: &Image,
        dst_subresource: ImageSubresourceLayers,
        dst_region: Rect3D,
        filter: VkFilter,
    ) {
        let device = Device::instance();
        let blits = [VkImageBlit {
            srcSubresource: VkImageSubresourceLayers {
                aspectMask: src_subresource.aspect.to_aspect(src.format),
                mipLevel: src_subresource.mip_level,
                baseArrayLayer: src_subresource.base_array_layer,
                layerCount: src_subresource.layer_count,
            },
            srcOffsets: [src_region.min.into(), src_region.max.into()],
            dstSubresource: VkImageSubresourceLayers {
                aspectMask: dst_subresource.aspect.to_aspect(dst.format),
                mipLevel: dst_subresource.mip_level,
                baseArrayLayer: dst_subresource.base_array_layer,
                layerCount: dst_subresource.layer_count,
            },
            dstOffsets: [dst_region.min.into(), dst_region.max.into()],
        }];
        // SAFETY: command buffer is OK, params OK
        unsafe {
            device.vk.CmdBlitImage(
                self.cmdbuf,
                src.handle(),
                VK_IMAGE_LAYOUT_GENERAL,
                dst.handle(),
                VK_IMAGE_LAYOUT_GENERAL,
                blits.len() as u32,
                blits.as_ptr(),
                filter,
            );
        }
    }
}
