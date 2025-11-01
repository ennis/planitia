use crate::device::get_vk_sample_count;
use crate::{
    aspects_for_format, BufferUntyped, Descriptor, Device, Format, ResourceAllocation, ResourceDescriptorIndex,
    ResourceId, Size3D, StorageImageHandle, TextureHandle, TrackedResource,
};
use ash::vk;
use bitflags::bitflags;
use gpu_allocator::vulkan::{AllocationCreateDesc, AllocationScheme};
use gpu_allocator::MemoryLocation;
use std::{mem, ptr};

/// Dimensionality of an image.
#[derive(Copy, Clone, Debug)]
pub enum ImageType {
    Image1D,
    Image2D,
    Image3D,
}

impl ImageType {
    pub const fn to_vk_image_type(self) -> vk::ImageType {
        match self {
            Self::Image1D => vk::ImageType::TYPE_1D,
            Self::Image2D => vk::ImageType::TYPE_2D,
            Self::Image3D => vk::ImageType::TYPE_3D,
        }
    }
}

impl From<ImageType> for vk::ImageType {
    fn from(ty: ImageType) -> Self {
        ty.to_vk_image_type()
    }
}

bitflags! {
    /// Bits describing the intended usage of an image.
    #[repr(transparent)]
    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
    pub struct ImageUsage: u32 {
        const TRANSFER_SRC = 0b1;
        const TRANSFER_DST = 0b10;
        const SAMPLED = 0b100;
        const STORAGE = 0b1000;
        const COLOR_ATTACHMENT = 0b1_0000;
        const DEPTH_STENCIL_ATTACHMENT = 0b10_0000;
        const TRANSIENT_ATTACHMENT = 0b100_0000;
        const INPUT_ATTACHMENT = 0b1000_0000;
    }
}

impl Default for ImageUsage {
    fn default() -> Self {
        Self::empty()
    }
}

impl ImageUsage {
    pub const fn to_vk_image_usage_flags(self) -> vk::ImageUsageFlags {
        vk::ImageUsageFlags::from_raw(self.bits())
    }
}

impl From<ImageUsage> for vk::ImageUsageFlags {
    fn from(usage: ImageUsage) -> Self {
        usage.to_vk_image_usage_flags()
    }
}

/// Information passed to `Image::new` to describe the image to be created.
#[derive(Copy, Clone, Debug)]
pub struct ImageCreateInfo<'a> {
    pub memory_location: MemoryLocation,
    /// Dimensionality of the image.
    pub type_: ImageType,
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
    /// Optional debug label.
    pub label: &'a str = "",
}

impl<'a> Default for ImageCreateInfo<'a> {
    fn default() -> Self {
        ImageCreateInfo {
            memory_location: MemoryLocation::Unknown,
            type_: ImageType::Image2D,
            usage: Default::default(),
            format: Default::default(),
            width: 1,
            ..
        }
    }
}

/// Image data stored in CPU-visible memory.
pub struct ImageBuffer {
    /// Host-mapped buffer containing the image data.
    data: BufferUntyped,
    format: Format,
    pitch: u32,
    width: u32,
    height: u32,
    depth: u32,
}

/// Represents an image resource on the GPU.
#[derive(Debug)]
pub struct Image {
    pub(crate) id: ResourceId,
    pub(crate) allocation: ResourceAllocation,
    pub(crate) swapchain_image: bool,
    pub(crate) descriptors: ImageResourceDescriptors,
    pub(crate) handle: vk::Image,
    pub(crate) usage: ImageUsage,
    pub(crate) type_: ImageType,
    pub(crate) format: Format,
    pub(crate) size: Size3D,
}

impl Drop for Image {
    fn drop(&mut self) {
        if !self.swapchain_image {
            let mut allocation = mem::take(&mut self.allocation);
            let handle = self.handle;
            let descriptors = self.descriptors;

            Device::global().delete_tracked_resource(self.id, move || unsafe {
                //debug!("dropping image {:?} (handle: {:?})", id, handle);
                let device = Device::global();
                device.free_resource_heap_index(descriptors.texture);
                device.free_resource_heap_index(descriptors.storage);
                device.raw.destroy_image_view(descriptors.image_view, None);
                device.raw.destroy_image(handle, None);
                device.free_memory(&mut allocation);
            });
        }
    }
}

impl TrackedResource for Image {
    fn id(&self) -> ResourceId {
        self.id
    }
}

impl Image {
    /// Creates a new image resource.
    pub fn new(image_info: ImageCreateInfo) -> Image {
        Device::global().create_image(&image_info)
    }

    pub fn set_name(&self, label: &str) {
        unsafe {
            Device::global().set_object_name(self.handle, label);
        }
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

    /// Returns the handle of the default image view.
    pub fn view_handle(&self) -> vk::ImageView {
        self.descriptors.image_view
    }

    /// Returns a descriptor for sampling this image in a shader.
    pub fn texture_descriptor(&self, layout: vk::ImageLayout) -> Descriptor<'_> {
        Descriptor::SampledImage { image: self, layout }
    }

    /// Returns a descriptor for accessing this image as a storage image in a shader.
    pub fn storage_image_descriptor(&self, layout: vk::ImageLayout) -> Descriptor<'_> {
        Descriptor::StorageImage { image: self, layout }
    }

    /// Returns the bindless texture handle of this image view.
    pub fn texture_descriptor_index(&self) -> TextureHandle {
        TextureHandle {
            index: self.descriptors.texture.index(),
            _unused: 0,
        }
    }

    /// Returns the bindless storage image handle of this image view.
    pub fn storage_descriptor_index(&self) -> StorageImageHandle {
        StorageImageHandle {
            index: self.descriptors.storage.index(),
            _unused: 0,
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub(crate) struct ImageResourceDescriptors {
    /// Index of the sampled image descriptor in the global descriptor heap.
    pub(crate) texture: ResourceDescriptorIndex,
    /// Index of the storage image descriptor in the global descriptor heap.
    pub(crate) storage: ResourceDescriptorIndex,
    pub(crate) image_view: vk::ImageView,
}

/// Image creation
impl Device {
    /// Creates the default image view for the image.
    pub(crate) fn create_image_resource_descriptors(
        &self,
        handle: vk::Image,
        type_: ImageType,
        usage: vk::ImageUsageFlags,
        format: Format,
        mip_levels: u32,
        array_layers: u32,
    ) -> ImageResourceDescriptors {
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

            let sampled_index = if usage.contains(vk::ImageUsageFlags::SAMPLED) {
                self.create_global_image_descriptor(
                    image_view,
                    vk::DescriptorType::SAMPLED_IMAGE,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                )
            } else {
                ResourceDescriptorIndex::default()
            };
            let storage_index = if usage.contains(vk::ImageUsageFlags::STORAGE) {
                self.create_global_image_descriptor(
                    image_view,
                    vk::DescriptorType::STORAGE_IMAGE,
                    vk::ImageLayout::GENERAL,
                )
            } else {
                ResourceDescriptorIndex::default()
            };

            ImageResourceDescriptors {
                texture: sampled_index,
                storage: storage_index,
                image_view,
            }
        }
    }

    /// Creates a new image resource.
    pub(crate) fn create_image(&self, image_info: &ImageCreateInfo) -> Image {
        unsafe {
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
                initial_layout: vk::ImageLayout::UNDEFINED,
                ..Default::default()
            };

            let handle = self
                .raw
                .create_image(&create_info, None)
                .expect("failed to create image");

            let mem_req = self.raw.get_image_memory_requirements(handle);
            let allocation = self.allocate_memory_or_panic(&AllocationCreateDesc {
                name: "",
                requirements: mem_req,
                location: image_info.memory_location,
                linear: true,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            });

            self.raw
                .bind_image_memory(handle, allocation.memory(), allocation.offset() as u64)
                .unwrap();

            if !image_info.label.is_empty() {
                self.set_object_name(handle, image_info.label);
            }

            // create the bindless image view
            let descriptors = self.create_image_resource_descriptors(
                handle,
                image_info.type_,
                create_info.usage,
                image_info.format,
                image_info.mip_levels,
                image_info.array_layers,
            );

            Image {
                handle,
                id: self.allocate_resource_id(),
                allocation: ResourceAllocation::Allocation { allocation },
                swapchain_image: false,
                descriptors,
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
    }
}
