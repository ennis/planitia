use crate::device::get_vk_sample_count;
use crate::{
    aspects_for_format, BufferUntyped, Descriptor, Device, Format, ImageCreateInfo, ImageHandle, ImageType, ImageUsage,
    RcDevice, ResourceAllocation, ResourceHeapIndex, ResourceId, Size3D, TrackedResource,
};
use ash::vk;
use gpu_allocator::vulkan::{AllocationCreateDesc, AllocationScheme};
use std::rc::Rc;
use std::{mem, ptr};

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

/// Wrapper around a Vulkan image.
#[derive(Debug)]
pub struct Image {
    pub(crate) device: RcDevice,
    pub(crate) id: ResourceId,
    pub(crate) allocation: ResourceAllocation,
    pub(crate) swapchain_image: bool,
    pub(crate) default_view: vk::ImageView,
    pub(crate) heap_index: ResourceHeapIndex,
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
            let default_view = self.default_view;
            let heap_index = self.heap_index;
            self.device.delete_tracked_resource(self.id, move |device| unsafe {
                //debug!("dropping image {:?} (handle: {:?})", id, handle);
                device.free_resource_heap_index(heap_index);
                device.free_memory(&mut allocation);
                device.raw.destroy_image_view(default_view, None);
                device.raw.destroy_image(handle, None);
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
    pub fn set_name(&self, label: &str) {
        unsafe {
            self.device.set_object_name(self.handle, label);
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
        self.default_view
    }

    pub fn device(&self) -> &RcDevice {
        &self.device
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
    pub fn device_image_handle(&self) -> ImageHandle {
        ImageHandle {
            index: self.heap_index.index(),
            _unused: 0,
        }
    }
}

/// Image creation
impl Device {

    /// Creates the default image view for the image.
    pub(crate) fn create_bindless_image_view(
        &self,
        handle: vk::Image,
        type_: ImageType,
        format: Format,
        mip_levels: u32,
        array_layers: u32,
    ) -> (ResourceHeapIndex, vk::ImageView) {
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

            let id = self.resource_heap.lock().unwrap().insert(());
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
            initial_layout: vk::ImageLayout::UNDEFINED,
            ..Default::default()
        };
        let handle = unsafe {
            self.raw
                .create_image(&create_info, None)
                .expect("failed to create image")
        };

        let mem_req = unsafe { self.raw.get_image_memory_requirements(handle) };
        let allocation = self.allocate_memory_or_panic(&AllocationCreateDesc {
            name: "",
            requirements: mem_req,
            location: image_info.memory_location,
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        });
        unsafe {
            self.raw
                .bind_image_memory(handle, allocation.memory(), allocation.offset() as u64)
                .unwrap();
        }

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
            device: self.clone(),
            id: self.allocate_resource_id(),
            allocation: ResourceAllocation::Allocation { allocation },
            swapchain_image: false,
            default_view,
            heap_index: bindless_handle,
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
