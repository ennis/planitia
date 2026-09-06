use crate::device::get_vk_sample_count;
use crate::{
    BufferUntyped, ColorAttachment, CommandBuffer, DepthStencilAttachment, Device, Format, ResourceAllocation, Size3D,
    StorageImageHandle, TextureHandle, VulkanObject, aspects_for_format, upload_image_data,
};
use ash::vk;
use gpu::ImageCopyView;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{AllocationCreateDesc, AllocationScheme};
use gpu_types::{ImageAspect, ImageType, ImageUsage, Offset3D};
use slotmap::Key;
use std::{mem, ptr};
use vulkan::{VkImageDescriptorInfoEXT, VkImageViewCreateInfo, VK_IMAGE_LAYOUT_GENERAL, VkResourceDescriptorInfoEXT, VkResourceDescriptorDataEXT};

/// Information passed to `Image::new` to describe the image to be created.
#[derive(Copy, Clone, Debug)]
pub struct ImageCreateInfo {
    pub memory_location: MemoryLocation = MemoryLocation::GpuOnly,
    /// Dimensionality of the image.
    pub type_: ImageType = ImageType::Image2D,
    /// Image usage flags. Must include all intended uses of the image.
    pub usage: ImageUsage,
    /// Format of the image.
    pub format: Format,
    /// Size of the image.
    pub width: u32,
    pub height: u32 = 1,
    pub depth: u32 = 1,
    /// Number of mipmap levels. Note that the mipmaps contents must still be generated manually. Default is 1. 0 is *not* a valid value.
    pub mip_levels: u32 = 1,
    /// Number of array layers. Default is `1`. `0` is *not* a valid value.
    pub array_layers: u32 = 1,
    /// Number of samples. Default is `1`. `0` is *not* a valid value.
    pub samples: u32 = 1,
}

/// Image data stored in CPU-visible memory.
pub struct ImageBuffer {
    /// Host-mapped buffer containing the image data.
    pub data: BufferUntyped,
    pub format: Format,
    pub pitch: u32,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
}

/// Represents an image resource on the GPU.
#[derive(Debug)]
pub struct Image {
    pub(crate) handle: vk::Image,
    pub(crate) attachment_view: vk::ImageView,
    pub(crate) memory_location: MemoryLocation,
    pub(crate) allocation: ResourceAllocation,
    pub(crate) swapchain_image: bool,
    pub(crate) descriptors: ImageDescriptors,
    pub(crate) usage: ImageUsage,
    pub(crate) type_: ImageType,
    pub(crate) format: Format,
    pub(crate) mip_levels: u32,
    pub(crate) array_layers: u32,
    pub(crate) size: Size3D,
    pub(crate) samples: u32,
}

impl Drop for Image {
    fn drop(&mut self) {
        if !self.swapchain_image {
            let mut allocation = mem::take(&mut self.allocation);
            let handle = self.handle;
            let descriptors = self.descriptors;
            Device::instance().delete_after_current_frame(move |device| unsafe {
                //debug!("dropping image {:?} (handle: {:?})", id, handle);
                if descriptors.texture != u32::MAX {
                    device.free_resource_descriptor(descriptors.texture);
                }
                if descriptors.image != u32::MAX {
                    device.free_resource_descriptor(descriptors.image);
                }
                if descriptors.stencil_texture != u32::MAX {
                    device.free_resource_descriptor(descriptors.stencil_texture);
                }
                if descriptors.stencil_image != u32::MAX {
                    device.free_resource_descriptor(descriptors.stencil_image);
                }
                device.raw.destroy_image(handle, None);
                device.free_memory(&mut allocation);
            });
        }
    }
}

impl VulkanObject for Image {
    type Handle = vk::Image;

    fn handle(&self) -> vk::Image {
        self.handle
    }
}

impl Image {
    /// Creates a new image resource.
    pub fn new(image_info: ImageCreateInfo) -> Image {
        Device::instance().create_image(&image_info)
    }

    /// Shorthand for creating a 2D image suitable for sampling and storage uses with the specified properties.
    ///
    /// Equivalent to `Image::new` with `usage: ImageUsage::SAMPLED | ImageUsage::STORAGE`.
    pub fn new_texture(width: u32, height: u32, format: Format) -> Image {
        Self::new(ImageCreateInfo {
            type_: ImageType::Image2D,
            usage: ImageUsage::SAMPLED | ImageUsage::STORAGE,
            format,
            width,
            height,
            ..
        })
    }

    /// Creates a new image suitable for sampling and storage uses with the specified properties, and initializes it with the provided data.
    pub fn new_texture_with_data(width: u32, height: u32, format: Format, aspect: ImageAspect, data: &[u8]) -> Image {
        let image = Self::new_texture(width, height, format);
        upload_image_data(
            ImageCopyView { image: &image, mip_level: 0, origin: Offset3D::ZERO, aspect },
            Size3D::new(width, height, 1),
            data,
        );
        image
    }

    /// Shorthand for creating a 2D image suitable for use as a color attachment, and for sampling and storage, with the specified properties.
    ///
    /// Equivalent to `Image::new` with `usage: ImageUsage::SAMPLED | ImageUsage::STORAGE | ImageUsage::COLOR_ATTACHMENT`.
    pub fn new_color_attachment(width: u32, height: u32, format: Format) -> Image {
        Self::new(ImageCreateInfo {
            type_: ImageType::Image2D,
            usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::SAMPLED | ImageUsage::STORAGE,
            format,
            width,
            height,
            ..
        })
    }

    /// Shorthand for creating a 2D image suitable for use as a depth-stencil attachment, and for sampling and storage, with the specified properties.
    ///
    /// Equivalent to `Image::new` with `usage: SAMPLED | STORAGE | DEPTH_STENCIL_ATTACHMENT`.
    pub fn new_depth_stencil_attachment(width: u32, height: u32, format: Format) -> Image {
        Self::new(ImageCreateInfo {
            type_: ImageType::Image2D,
            usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT | ImageUsage::SAMPLED | ImageUsage::STORAGE,
            format,
            width,
            height,
            ..
        })
    }

    /// Returns the type (dimensionality) of the image.
    pub fn image_type(&self) -> ImageType {
        self.type_
    }

    /// Returns the format of the image.
    pub fn format(&self) -> Format {
        self.format
    }

    /// Returns the size in pixels of the image.
    pub fn size(&self) -> Size3D {
        self.size
    }

    /// Returns the width of the image.
    pub fn width(&self) -> u32 {
        self.size.width
    }

    /// Returns the height of the image.
    ///
    /// This is 1 for 1D images.
    pub fn height(&self) -> u32 {
        self.size.height
    }

    /// Returns the depth of the image.
    ///
    /// This is 1 for 1D & 2D images.
    pub fn depth(&self) -> u32 {
        self.size.depth
    }

    /// Returns the usage flags of the image.
    pub fn usage(&self) -> ImageUsage {
        self.usage
    }

    /// Returns the image handle.
    pub fn handle(&self) -> vk::Image {
        self.handle
    }

    // Returns the handle of the default image view.
    //pub fn view_handle(&self) -> vk::ImageView {
    //    self.descriptors.image_view
    //}

    /*    /// Returns a descriptor for sampling this image in a shader.
    pub fn texture_descriptor(&self, layout: vk::ImageLayout) -> Descriptor<'_> {
        Descriptor::SampledImage { image: self, layout }
    }

    /// Returns a descriptor for accessing this image as a storage image in a shader.
    pub fn storage_image_descriptor(&self, layout: vk::ImageLayout) -> Descriptor<'_> {
        Descriptor::StorageImage { image: self, layout }
    }*/

    /// Returns the bindless texture handle of this image view.
    pub fn texture_handle(&self) -> TextureHandle {
        TextureHandle::new(self.descriptors.texture)
    }

    /// Returns the bindless storage image handle of this image view.
    pub fn storage_handle(&self) -> StorageImageHandle {
        StorageImageHandle::new(self.descriptors.image)
    }

    /// Discards the contents of the image and resizes this image to the new dimensions.
    ///
    /// This effectively creates a new image and that replaces the old one.
    /// The contents of the existing image are discarded.
    /// Any existing descriptors or handles will become invalid, but those used in previous command
    /// buffer operations stay valid until those command buffers have finished executing.
    ///
    /// This function must be called only on images created with [`new`](Image::new).
    /// It will panic when called on images that refer to external storage, like swap chain images
    /// or images imported from external handles.
    /// Also, it cannot change the dimensionality of the current image.
    ///
    /// # Panics
    /// - when called on a swap chain image
    /// - when called on an imported image (via e.g. create_imported_image_win32).
    /// - when the specified dimensions do not match the current dimensionality
    ///   (e.g. depth != 1 when image_type is Image2D).
    pub fn resize_no_copy(&mut self, new_size: Size3D) {
        assert!(!self.swapchain_image, "cannot resize a swap chain image");
        assert!(
            !matches!(self.allocation, ResourceAllocation::External | ResourceAllocation::DeviceMemory { .. }),
            "cannot resize images created from external memory"
        );
        match self.type_ {
            ImageType::Image1D => assert_eq!(new_size.height, 1, "cannot change image dimensionality when resizing"),
            ImageType::Image2D => assert_eq!(new_size.depth, 1, "cannot change image dimensionality when resizing"),
            ImageType::Image3D => {}
        }
        *self = Self::new(ImageCreateInfo {
            memory_location: self.memory_location,
            type_: self.type_,
            usage: self.usage,
            format: self.format,
            width: new_size.width,
            height: new_size.height,
            depth: new_size.depth,
            mip_levels: self.mip_levels,
            array_layers: self.array_layers,
            samples: self.samples,
        });
    }

    /// Returns a [`ColorAttachment`] referencing this image, with the specified clear color.
    ///
    /// # Arguments
    /// - `clear_color`: color to clear to when beginning a render pass with this attachment. `None` to leave contents unchanged.
    pub fn as_color_attachment(&self, clear_color: impl Into<Option<[f64; 4]>>) -> ColorAttachment<'_> {
        ColorAttachment { image: self, clear: clear_color.into() }
    }

    /// Returns a [`DepthStencilAttachment`] referencing this image, with the specified clear depth and stencil values.
    ///
    /// # Arguments
    /// - `clear_depth`: depth value to clear to when beginning a render pass with this attachment. `None` to leave depth contents unchanged.
    /// - `clear_stencil`: stencil value to clear to when beginning a render pass with this attachment. `None` to leave stencil contents unchanged.
    pub fn as_depth_stencil_attachment(
        &self,
        clear_depth: impl Into<Option<f64>>,
        clear_stencil: impl Into<Option<u32>>,
    ) -> DepthStencilAttachment<'_> {
        DepthStencilAttachment { image: self, depth_clear: clear_depth.into(), stencil_clear: clear_stencil.into() }
    }
}

/*
#[derive(Debug, Copy, Clone)]
pub(crate) struct ImageResourceDescriptors {
    /// Index of the sampled image descriptor in the global descriptor heap.
    pub(crate) texture: ResourceDescriptorIndex,
    /// Index of the storage image descriptor in the global descriptor heap.
    pub(crate) storage: ResourceDescriptorIndex,
    pub(crate) stencil_texture: ResourceDescriptorIndex,
    pub(crate) stencil_storage: ResourceDescriptorIndex,
    pub(crate) image_view: vk::ImageView,
    pub(crate) stencil_view: vk::ImageView,
}*/

#[derive(Copy, Clone, Debug, Default)]
pub(crate) struct ImageDescriptors {
    /// Texture descriptor for sampling the image in shaders.
    pub(crate) texture: u32 = u32::MAX,
    /// Storage image descriptor for reading/writing the image in shaders.
    pub(crate) image: u32= u32::MAX,
    /// If the image has a stencil aspect, the descriptor to sample the stencil aspect in shaders.
    pub(crate) stencil_texture: u32= u32::MAX,
    /// If the image has a stencil aspect, the descriptor to read/write the stencil aspect in shaders.
    pub(crate) stencil_image: u32= u32::MAX,
}

/// Image creation
impl Device {
    //pub(crate) fn create_image_resource_descriptors(
    //    &self,
    //    handle: vk::Image,
    //    type_: ImageType,
    //    usage: vk::ImageUsageFlags,
    //    format: Format,
    //    mip_levels: u32,
    //    array_layers: u32,
    //) -> ImageResourceDescriptors {
    //    let create_aspect_view = |aspect: vk::ImageAspectFlags| unsafe {
    //        self.raw
    //            .create_image_view(
    //                &vk::ImageViewCreateInfo {
    //                    flags: vk::ImageViewCreateFlags::empty(),
    //                    image: handle,
    //                    view_type: match type_ {
    //                        ImageType::Image1D => {
    //                            if array_layers > 1 {
    //                                vk::ImageViewType::TYPE_1D_ARRAY
    //                            } else {
    //                                vk::ImageViewType::TYPE_1D
    //                            }
    //                        }
    //                        ImageType::Image2D => {
    //                            if array_layers > 1 {
    //                                vk::ImageViewType::TYPE_2D_ARRAY
    //                            } else {
    //                                vk::ImageViewType::TYPE_2D
    //                            }
    //                        }
    //                        ImageType::Image3D => vk::ImageViewType::TYPE_3D,
    //                    },
    //                    format,
    //                    components: vk::ComponentMapping {
    //                        r: vk::ComponentSwizzle::IDENTITY,
    //                        g: vk::ComponentSwizzle::IDENTITY,
    //                        b: vk::ComponentSwizzle::IDENTITY,
    //                        a: vk::ComponentSwizzle::IDENTITY,
    //                    },
    //                    subresource_range: vk::ImageSubresourceRange {
    //                        aspect_mask: aspect,
    //                        base_mip_level: 0,
    //                        level_count: mip_levels,
    //                        base_array_layer: 0,
    //                        layer_count: array_layers,
    //                    },
    //                    ..Default::default()
    //                },
    //                None,
    //            )
    //            .expect("failed to create image view")
    //    };
    //
    //    unsafe {
    //        let aspects = aspects_for_format(format);
    //        let mut main_view = vk::ImageView::null();
    //        let mut stencil_view = vk::ImageView::null();
    //        if aspects.contains(vk::ImageAspectFlags::COLOR) {
    //            main_view = create_aspect_view(vk::ImageAspectFlags::COLOR);
    //        }
    //        if aspects.contains(vk::ImageAspectFlags::DEPTH) {
    //            main_view = create_aspect_view(vk::ImageAspectFlags::DEPTH);
    //        }
    //        if aspects.contains(vk::ImageAspectFlags::STENCIL) {
    //            stencil_view = create_aspect_view(vk::ImageAspectFlags::STENCIL);
    //        }
    //
    //        let mut texture = ResourceDescriptorIndex::default();
    //        let mut storage = ResourceDescriptorIndex::default();
    //        let mut stencil_texture = ResourceDescriptorIndex::default();
    //        let mut stencil_storage = ResourceDescriptorIndex::default();
    //
    //        if usage.contains(vk::ImageUsageFlags::SAMPLED) {
    //            if !main_view.is_null() {
    //                texture = self.create_global_image_descriptor(
    //                    main_view,
    //                    vk::DescriptorType::SAMPLED_IMAGE,
    //                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
    //                )
    //            }
    //            if !stencil_view.is_null() {
    //                stencil_texture = self.create_global_image_descriptor(
    //                    stencil_view,
    //                    vk::DescriptorType::SAMPLED_IMAGE,
    //                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
    //                )
    //            }
    //        };
    //        if usage.contains(vk::ImageUsageFlags::STORAGE) {
    //            if !main_view.is_null() {
    //                storage = self.create_global_image_descriptor(
    //                    main_view,
    //                    vk::DescriptorType::STORAGE_IMAGE,
    //                    vk::ImageLayout::GENERAL,
    //                )
    //            }
    //            if !stencil_view.is_null() {
    //                stencil_storage = self.create_global_image_descriptor(
    //                    stencil_view,
    //                    vk::DescriptorType::STORAGE_IMAGE,
    //                    vk::ImageLayout::GENERAL,
    //                )
    //            }
    //        };
    //
    //        ImageResourceDescriptors {
    //            texture,
    //            storage,
    //            stencil_texture,
    //            stencil_storage,
    //            image_view: main_view,
    //            stencil_view,
    //        }
    //    }
    //}

    // Helper to transition to GENERAL layout during initialization and appease the validation layers.
    // The contents of the image will be undefined.
    pub(crate) unsafe fn transition_image_to_general(&self, image: vk::Image, aspect_mask: vk::ImageAspectFlags) {
        unsafe {
            let mut cmd = CommandBuffer::new();
            cmd.image_barrier(&vk::ImageMemoryBarrier2 {
                src_stage_mask: vk::PipelineStageFlags2::NONE,
                src_access_mask: vk::AccessFlags2::empty(),
                dst_stage_mask: vk::PipelineStageFlags2::ALL_COMMANDS,
                dst_access_mask: vk::AccessFlags2::MEMORY_READ,
                old_layout: vk::ImageLayout::UNDEFINED,
                new_layout: vk::ImageLayout::GENERAL,
                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                image,
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask,
                    base_mip_level: 0,
                    level_count: vk::REMAINING_MIP_LEVELS,
                    base_array_layer: 0,
                    layer_count: vk::REMAINING_ARRAY_LAYERS,
                },
                ..Default::default()
            });
            crate::submit(cmd);
        }
    }

    /// Creates the main image view for an image resource, suitable for use as a color or depth/stencil attachment.
    pub(crate) unsafe fn create_attachment_image_view(&self, image: vk::Image, format: vk::Format) -> vk::ImageView {
        unsafe {
            self.raw
                .create_image_view(
                    &vk::ImageViewCreateInfo {
                        flags: vk::ImageViewCreateFlags::empty(),
                        image,
                        view_type: vk::ImageViewType::TYPE_2D,
                        format,
                        components: vk::ComponentMapping {
                            r: vk::ComponentSwizzle::IDENTITY,
                            g: vk::ComponentSwizzle::IDENTITY,
                            b: vk::ComponentSwizzle::IDENTITY,
                            a: vk::ComponentSwizzle::IDENTITY,
                        },
                        subresource_range: vk::ImageSubresourceRange {
                            //  > When an image view of a depth/stencil image is used as a depth/stencil framebuffer attachment,
                            //    the aspectMask is ignored and both depth and stencil image subresources are used.
                            // (https://docs.vulkan.org/refpages/latest/refpages/source/VkImageSubresourceRange.html)
                            aspect_mask: aspects_for_format(format),
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        },
                        ..Default::default()
                    },
                    None,
                )
                .expect("failed to create image view")
        }
    }

    /// Creates a new image resource.
    pub(crate) fn create_image(&self, image_info: &ImageCreateInfo) -> Image {
        unsafe {
            let create_info = vk::ImageCreateInfo {
                image_type: image_info.type_.into(),
                format: image_info.format,
                extent: vk::Extent3D { width: image_info.width, height: image_info.height, depth: image_info.depth },
                mip_levels: image_info.mip_levels,
                array_layers: image_info.array_layers,
                samples: get_vk_sample_count(image_info.samples),
                tiling: vk::ImageTiling::OPTIMAL, // LINEAR tiling not used enough to be exposed
                usage: image_info.usage.into(),
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                queue_family_index_count: 0,
                p_queue_family_indices: ptr::null(),
                initial_layout: vk::ImageLayout::UNDEFINED,
                ..Default::default()
            };
            let handle = self.raw.create_image(&create_info, None).expect("failed to create image");
            let mem_req = self.raw.get_image_memory_requirements(handle);
            let allocation = self.allocate_memory_or_panic(&AllocationCreateDesc {
                name: "",
                requirements: mem_req,
                location: image_info.memory_location,
                linear: true,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            });
            self.raw.bind_image_memory(handle, allocation.memory(), allocation.offset() as u64).unwrap();
            let descriptors = self.register_image_descriptors(handle, &create_info);
            let attachment_view = self.create_attachment_image_view(handle, image_info.format);
            self.transition_image_to_general(handle, aspects_for_format(image_info.format));
            Image {
                handle,
                attachment_view,
                memory_location: image_info.memory_location,
                allocation: ResourceAllocation::Allocation { allocation },
                swapchain_image: false,
                descriptors,
                usage: image_info.usage,
                type_: image_info.type_,
                format: image_info.format,
                mip_levels: image_info.mip_levels,
                array_layers: image_info.array_layers,
                size: Size3D { width: image_info.width, height: image_info.height, depth: image_info.depth },
                samples: image_info.samples,
            }
        }
    }

    pub(crate) fn allocate_image_descriptor(
        &self,
        descriptor_type: vk::DescriptorType,
        view: &vk::ImageViewCreateInfo,
    ) -> u32 {
        let info = VkImageDescriptorInfoEXT {
            pView: view as *const _ as *const VkImageViewCreateInfo,
            layout: VK_IMAGE_LAYOUT_GENERAL,
            ..
        };
        self.allocate_resource_descriptor(&VkResourceDescriptorInfoEXT {
            r#type: descriptor_type.as_raw(),
            data: VkResourceDescriptorDataEXT { pImage: &info },
            ..
        })
    }

    pub(crate) fn register_image_descriptors(
        &self,
        handle: vk::Image,
        create_info: &vk::ImageCreateInfo,
    ) -> ImageDescriptors {
        let image_view_type = match create_info.image_type {
            vk::ImageType::TYPE_1D => {
                if create_info.array_layers > 1 {
                    vk::ImageViewType::TYPE_1D_ARRAY
                } else {
                    vk::ImageViewType::TYPE_1D
                }
            }
            vk::ImageType::TYPE_2D => {
                if create_info.array_layers > 1 {
                    vk::ImageViewType::TYPE_2D_ARRAY
                } else {
                    vk::ImageViewType::TYPE_2D
                }
            }
            vk::ImageType::TYPE_3D => vk::ImageViewType::TYPE_3D,
            _ => panic!("invalid image type"),
        };
        let view_for_aspect = |aspect: vk::ImageAspectFlags| {
            vk::ImageViewCreateInfo {
                flags: vk::ImageViewCreateFlags::empty(),
                image: handle,
                view_type: image_view_type,
                format: create_info.format,
                components: vk::ComponentMapping::default(), //  defaults to IDENTITY
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: aspect,
                    base_mip_level: 0,
                    level_count: create_info.mip_levels,
                    base_array_layer: 0,
                    layer_count: create_info.array_layers,
                },
                ..Default::default()
            }
        };
        let aspects = aspects_for_format(create_info.format);
        let main_view: vk::ImageViewCreateInfo; // color or depth aspect
        let stencil_view: vk::ImageViewCreateInfo; // stencil aspect
        let mut texture_descriptor = u32::MAX;
        let mut storage_descriptor = u32::MAX;
        let mut stencil_texture_descriptor = u32::MAX;
        let mut stencil_storage_descriptor = u32::MAX;
        if aspects.intersects(vk::ImageAspectFlags::COLOR | vk::ImageAspectFlags::DEPTH) {
            let main_aspect = if aspects.contains(vk::ImageAspectFlags::COLOR) {
                vk::ImageAspectFlags::COLOR
            } else {
                vk::ImageAspectFlags::DEPTH
            };
            main_view = view_for_aspect(main_aspect);
            if create_info.usage.contains(vk::ImageUsageFlags::SAMPLED) {
                // COLOR or DEPTH aspect, SAMPLED access
                texture_descriptor = self.allocate_image_descriptor(vk::DescriptorType::SAMPLED_IMAGE, &main_view);
            }
            if create_info.usage.contains(vk::ImageUsageFlags::STORAGE) {
                // COLOR or DEPTH aspect, STORAGE access
                storage_descriptor = self.allocate_image_descriptor(vk::DescriptorType::STORAGE_IMAGE, &main_view);
            }
        }
        if aspects.intersects(vk::ImageAspectFlags::STENCIL) {
            stencil_view = view_for_aspect(vk::ImageAspectFlags::STENCIL);
            if create_info.usage.contains(vk::ImageUsageFlags::SAMPLED) {
                // STENCIL aspect, SAMPLED access
                stencil_texture_descriptor =
                    self.allocate_image_descriptor(vk::DescriptorType::SAMPLED_IMAGE, &stencil_view);
            }
            if create_info.usage.contains(vk::ImageUsageFlags::STORAGE) {
                // STENCIL aspect, STORAGE access
                stencil_storage_descriptor =
                    self.allocate_image_descriptor(vk::DescriptorType::STORAGE_IMAGE, &stencil_view);
            }
        }
        ImageDescriptors {
            texture: texture_descriptor,
            image: storage_descriptor,
            stencil_texture: stencil_texture_descriptor,
            stencil_image: stencil_storage_descriptor,
        }
    }
}
