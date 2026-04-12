//! Blit commands
//!
//! Wrappers for clear/blit/copy commands on buffers and images.
use ash::vk;

use crate::{
    BufferRangeUntyped, BufferUntyped, ClearColorValue, CommandBuffer, Device, Image, ImageCopyBuffer, ImageCopyView,
    ImageSubresourceLayers, Rect3D, Size3D,
};

fn full_subresource_range(aspects: vk::ImageAspectFlags) -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange {
        aspect_mask: aspects,
        base_mip_level: 0,
        level_count: vk::REMAINING_MIP_LEVELS,
        base_array_layer: 0,
        layer_count: vk::REMAINING_ARRAY_LAYERS,
    }
}

impl CommandBuffer {
    pub fn fill_buffer(&mut self, range: &BufferRangeUntyped, data: u32) {
        let cb = self.get_or_create_command_buffer();
        unsafe {
            // SAFETY: FFI call and parameters are valid
            Device::global()
                .raw
                .cmd_fill_buffer(cb, range.buffer.handle(), range.byte_offset, range.byte_size, data);
        }
    }

    // TODO specify subresources
    pub fn clear_image(&mut self, image: &Image, clear_color_value: ClearColorValue) {
        let cb = self.get_or_create_command_buffer();
        unsafe {
            // SAFETY: FFI call and parameters are valid
            Device::global().raw.cmd_clear_color_image(
                cb,
                image.handle(),
                vk::ImageLayout::GENERAL,
                &clear_color_value.into(),
                &[full_subresource_range(vk::ImageAspectFlags::COLOR)],
            );
        }
    }

    pub fn clear_depth_image(&mut self, image: &Image, depth: f32) {
        let cb = self.get_or_create_command_buffer();
        unsafe {
            // SAFETY: FFI call and parameters are valid
            Device::global().raw.cmd_clear_depth_stencil_image(
                cb,
                image.handle(),
                vk::ImageLayout::GENERAL,
                &vk::ClearDepthStencilValue { depth, stencil: 0 },
                &[full_subresource_range(vk::ImageAspectFlags::DEPTH)],
            );
        }
    }

    pub fn copy_image_to_image(
        &mut self,
        source: ImageCopyView<'_>,
        destination: ImageCopyView<'_>,
        copy_size: vk::Extent3D,
    ) {
        // TODO: this is not required for multi-planar formats
        assert_eq!(source.aspect, destination.aspect);

        let regions = [vk::ImageCopy {
            src_subresource: vk::ImageSubresourceLayers {
                aspect_mask: source.aspect.to_aspect(source.image.format),
                mip_level: source.mip_level,
                base_array_layer: 0,
                layer_count: 1,
            },
            src_offset: source.origin.into(),
            dst_subresource: vk::ImageSubresourceLayers {
                aspect_mask: destination.aspect.to_aspect(destination.image.format),
                mip_level: destination.mip_level,
                base_array_layer: 0,
                layer_count: 1,
            },
            dst_offset: destination.origin.into(),
            extent: copy_size,
        }];

        // SAFETY: FFI call and parameters are valid
        let cb = self.get_or_create_command_buffer();
        unsafe {
            Device::global().raw.cmd_copy_image(
                cb,
                source.image.handle(),
                vk::ImageLayout::GENERAL,
                destination.image.handle(),
                vk::ImageLayout::GENERAL,
                &regions,
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
        assert!(src_offset + size <= source.byte_size());
        assert!(dst_offset + size <= destination.byte_size());

        // SAFETY: FFI call and parameters are valid
        let cb = self.get_or_create_command_buffer();
        unsafe {
            Device::global().raw.cmd_copy_buffer(
                cb,
                source.handle(),
                destination.handle(),
                &[vk::BufferCopy {
                    src_offset,
                    dst_offset,
                    size,
                }],
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
        copy_size: vk::Extent3D,
    ) {
        let regions = [vk::BufferImageCopy {
            buffer_offset: source.layout.offset,
            buffer_row_length: source.layout.texel_row_length.unwrap_or(0),
            buffer_image_height: source.layout.row_count.unwrap_or(0),
            image_subresource: vk::ImageSubresourceLayers {
                aspect_mask: destination.aspect.to_aspect(destination.image.format),
                mip_level: destination.mip_level,
                base_array_layer: 0,
                layer_count: 1,
            },
            image_offset: vk::Offset3D {
                x: destination.origin.x,
                y: destination.origin.y,
                z: destination.origin.z,
            },
            image_extent: copy_size,
        }];

        // SAFETY: FFI call and parameters are valid
        let cb = self.get_or_create_command_buffer();
        unsafe {
            Device::global().raw.cmd_copy_buffer_to_image(
                cb,
                source.buffer.handle(),
                destination.image.handle(),
                vk::ImageLayout::GENERAL,
                &regions,
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
        let regions = [vk::BufferImageCopy {
            buffer_offset: destination.layout.offset,
            buffer_row_length: destination.layout.texel_row_length.unwrap_or(0),
            buffer_image_height: destination.layout.row_count.unwrap_or(0),
            image_subresource: vk::ImageSubresourceLayers {
                aspect_mask: source.aspect.to_aspect(source.image.format),
                mip_level: source.mip_level,
                base_array_layer: 0,
                layer_count: 1,
            },
            image_offset: source.origin.into(),
            image_extent: copy_size.into(),
        }];

        // SAFETY: FFI call and parameters are valid
        let cb = self.get_or_create_command_buffer();
        unsafe {
            Device::global().raw.cmd_copy_image_to_buffer(
                cb,
                source.image.handle(),
                vk::ImageLayout::GENERAL,
                destination.buffer.handle(),
                &regions,
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
        filter: vk::Filter,
    ) {
        let blits = [vk::ImageBlit {
            src_subresource: vk::ImageSubresourceLayers {
                aspect_mask: src_subresource.aspect.to_aspect(src.format),
                mip_level: src_subresource.mip_level,
                base_array_layer: src_subresource.base_array_layer,
                layer_count: src_subresource.layer_count,
            },
            src_offsets: [src_region.min.into(), src_region.max.into()],
            dst_subresource: vk::ImageSubresourceLayers {
                aspect_mask: dst_subresource.aspect.to_aspect(dst.format),
                mip_level: dst_subresource.mip_level,
                base_array_layer: dst_subresource.base_array_layer,
                layer_count: dst_subresource.layer_count,
            },
            dst_offsets: [dst_region.min.into(), dst_region.max.into()],
        }];

        // SAFETY: command buffer is OK, params OK
        let cb = self.get_or_create_command_buffer();
        unsafe {
            Device::global().raw.cmd_blit_image(
                cb,
                src.handle(),
                vk::ImageLayout::GENERAL,
                dst.handle(),
                vk::ImageLayout::GENERAL,
                &blits,
                filter,
            );
        }
    }
}
