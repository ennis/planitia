use crate::device::{RESOURCE_DESCRIPTOR_HEAP_SIZE, ResourceDescriptorIndex, SAMPLER_DESCRIPTOR_HEAP_SIZE};
use crate::{CommandBuffer, Device, Format, ImageType, SamplerDescriptorIndex, aspects_for_format};
use ash::vk;
use ash::vk::Handle;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator};
use slotmap::SlotMap;
use std::ffi::c_void;
use std::mem::MaybeUninit;
use std::ptr;
use std::sync::Mutex;
use vulkan_headers::vulkan::vulkan as vk2;
use vulkan_headers::vulkan::vulkan::{
    VK_BUFFER_USAGE_DESCRIPTOR_HEAP_BIT_EXT, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
    VK_IMAGE_LAYOUT_GENERAL, VK_STRUCTURE_TYPE_BIND_HEAP_INFO_EXT, VK_STRUCTURE_TYPE_IMAGE_DESCRIPTOR_INFO_EXT,
    VK_STRUCTURE_TYPE_RESOURCE_DESCRIPTOR_INFO_EXT, VkBindHeapInfoEXT, VkCommandBuffer, VkDevice,
    VkDeviceAddressRangeEXT, VkHostAddressRangeEXT, VkImageDescriptorInfoEXT, VkImageViewCreateInfo,
    VkPhysicalDeviceDescriptorHeapPropertiesEXT, VkResourceDescriptorDataEXT, VkResourceDescriptorInfoEXT,
    VkSamplerCreateInfo,
};

struct DescriptorHeapInfo {
    alloc: Allocation,
    buffer: vk::Buffer,
    ptr: *mut c_void,
    device_addr: vk::DeviceAddress,
    /// Offset to the beginning of descriptors in the resource heap.
    start_offset: usize,
    /// Stride between consecutive descriptors in the heap.
    stride: usize,
    /// Alignment of descriptors in the heap.
    alignment: usize,
}

unsafe impl Send for DescriptorHeapInfo {}
unsafe impl Sync for DescriptorHeapInfo {}

impl DescriptorHeapInfo {
    /// Returns the offset of the descriptor at the given global index within the heap.
    fn descriptor_offset(&self, index: usize) -> usize {
        self.start_offset + index * self.stride
    }

    /// Returns a VkHostAddressRange for the descriptor at the given global index.
    fn address_range_by_index(&self, start_index: usize, index_count: usize) -> vk2::VkHostAddressRangeEXT {
        vk2::VkHostAddressRangeEXT {
            address: unsafe { self.ptr.add(self.descriptor_offset(start_index)) },
            size: self.stride * index_count,
        }
    }
}

#[derive(Debug)]
enum DescriptorHeapType {
    Resource,
    Sampler,
}

fn allocate_descriptor_heap_memory(
    allocator: &mut Allocator,
    device: &ash::Device,
    heap_type: DescriptorHeapType,
    byte_size: usize,
    descriptor_heap_properties: &vk2::VkPhysicalDeviceDescriptorHeapPropertiesEXT,
) -> DescriptorHeapInfo {
    let mut usage_flags = vk::BufferUsageFlags::from_raw(VK_BUFFER_USAGE_DESCRIPTOR_HEAP_BIT_EXT);
    usage_flags |= vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
    let alignment = match heap_type {
        DescriptorHeapType::Resource => descriptor_heap_properties.resourceHeapAlignment,
        DescriptorHeapType::Sampler => descriptor_heap_properties.samplerHeapAlignment,
    };
    let max_size = match heap_type {
        DescriptorHeapType::Resource => descriptor_heap_properties.maxResourceHeapSize,
        DescriptorHeapType::Sampler => descriptor_heap_properties.maxSamplerHeapSize,
    } as usize;

    assert!(
        byte_size <= max_size,
        "requested descriptor heap size exceeds the maximum supported size of {} for {:?} heap",
        max_size,
        heap_type
    );

    let alloc = allocator
        .allocate(&AllocationCreateDesc {
            name: "descriptor heap".into(),
            requirements: vk::MemoryRequirements {
                size: byte_size as u64,
                alignment,
                memory_type_bits: u32::MAX,
            },
            location: gpu_allocator::MemoryLocation::CpuToGpu,
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })
        .expect("failed to allocate descriptor heap memory");

    let buffer;
    let device_addr;

    unsafe {
        buffer = device
            .create_buffer(
                &vk::BufferCreateInfo {
                    size: byte_size as u64,
                    usage: usage_flags,
                    sharing_mode: vk::SharingMode::EXCLUSIVE,
                    ..Default::default()
                },
                None,
            )
            .expect("failed to create descriptor heap buffer");
        device
            .bind_buffer_memory(buffer, alloc.memory(), alloc.offset())
            .expect("failed to bind memory for descriptor heap buffer");
        device_addr = device.get_buffer_device_address(&vk::BufferDeviceAddressInfo {
            buffer,
            ..Default::default()
        });
    }
    let ptr = alloc
        .mapped_ptr()
        .expect("failed to map descriptor heap memory")
        .as_ptr();

    let start_offset;
    let stride;
    let alignment;
    match heap_type {
        DescriptorHeapType::Resource => {
            alignment = descriptor_heap_properties
                .bufferDescriptorAlignment
                .max(descriptor_heap_properties.imageDescriptorAlignment) as usize;
            start_offset = descriptor_heap_properties
                .minResourceHeapReservedRange
                .next_multiple_of(alignment as u64) as usize;
            stride = descriptor_heap_properties
                .bufferDescriptorSize
                .max(descriptor_heap_properties.imageDescriptorSize) as usize;
        }
        DescriptorHeapType::Sampler => {
            start_offset = descriptor_heap_properties
                .minSamplerHeapReservedRange
                .next_multiple_of(descriptor_heap_properties.samplerDescriptorAlignment)
                as usize;
            stride = descriptor_heap_properties.samplerDescriptorSize as usize;
            alignment = descriptor_heap_properties.samplerDescriptorAlignment as usize;
        }
    }

    DescriptorHeapInfo {
        alloc,
        buffer,
        ptr,
        device_addr,
        start_offset,
        stride,
        alignment,
    }
}

pub(crate) struct DeviceDescriptorIndexTable {
    pub(crate) resource: SlotMap<ResourceDescriptorIndex, ()>,
    pub(crate) sampler: SlotMap<SamplerDescriptorIndex, ()>,
}

/// Device state related to resource and sampler descriptor heaps.
pub(crate) struct DescriptorHeaps {
    /// Descriptor writes must be externally synchronized, but we don't want to
    /// wrap DescriptorHeapInfo in a Mutex because that would require locking every time we want
    /// to copy the address. So instead we lock this mutex when writing to the descriptor set.
    write_lock: Mutex<()>,
    resource_heap: DescriptorHeapInfo,
    sampler_heap: DescriptorHeapInfo,

    indices: Mutex<DeviceDescriptorIndexTable>,
}

impl DescriptorHeaps {
    pub(super) unsafe fn bind_descriptor_heaps(&self, command_buffer: vk::CommandBuffer) {
        let device = Device::global();
        let ext = &device.extensions.ext_descriptor_heap;
        let cb = command_buffer.as_raw() as VkCommandBuffer;
        unsafe {
            (ext.cmd_bind_resource_heap)(
                cb,
                &VkBindHeapInfoEXT {
                    sType: VK_STRUCTURE_TYPE_BIND_HEAP_INFO_EXT,
                    pNext: ptr::null(),
                    heapRange: VkDeviceAddressRangeEXT {
                        address: self.resource_heap.device_addr,
                        size: self.resource_heap.alloc.size(),
                    },
                    reservedRangeOffset: 0,
                    reservedRangeSize: device
                        .thread_safe
                        .descriptor_heap_properties
                        .minResourceHeapReservedRange,
                },
            );
            (ext.cmd_bind_sampler_heap)(
                cb,
                &VkBindHeapInfoEXT {
                    sType: VK_STRUCTURE_TYPE_BIND_HEAP_INFO_EXT,
                    pNext: ptr::null(),
                    heapRange: VkDeviceAddressRangeEXT {
                        address: self.sampler_heap.device_addr,
                        size: self.sampler_heap.alloc.size(),
                    },
                    reservedRangeOffset: 0,
                    reservedRangeSize: device
                        .thread_safe
                        .descriptor_heap_properties
                        .minSamplerHeapReservedRange,
                },
            );
        }
    }

    pub(super) fn new(
        allocator: &mut Allocator,
        device: &ash::Device,
        descriptor_heap_properties: &VkPhysicalDeviceDescriptorHeapPropertiesEXT,
    ) -> DescriptorHeaps {
        // allocate descriptor heap memory
        let resource_heap = allocate_descriptor_heap_memory(
            allocator,
            device,
            DescriptorHeapType::Resource,
            RESOURCE_DESCRIPTOR_HEAP_SIZE,
            &descriptor_heap_properties,
        );
        let sampler_heap = allocate_descriptor_heap_memory(
            allocator,
            device,
            DescriptorHeapType::Sampler,
            SAMPLER_DESCRIPTOR_HEAP_SIZE,
            &descriptor_heap_properties,
        );

        DescriptorHeaps {
            resource_heap,
            sampler_heap,

            write_lock: Mutex::new(()),
            indices: Mutex::new(DeviceDescriptorIndexTable {
                resource: SlotMap::with_key(),
                sampler: SlotMap::with_key(),
            }),
        }
    }
}

// TODO: this could be only an offset
#[derive(Debug, Clone, Copy)]
pub(crate) struct ResourceDescriptorOffsetAndIndex {
    pub(crate) offset: usize,
    pub(crate) index: usize,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct SamplerDescriptorOffsetAndIndex {
    pub(crate) offset: usize,
    pub(crate) index: usize,
}

pub(crate) struct ImageDescriptors {
    /// Texture descriptor for sampling the image in shaders.
    pub(crate) texture: Option<ResourceDescriptorOffsetAndIndex>,
    /// Storage image descriptor for reading/writing the image in shaders.
    pub(crate) image: Option<ResourceDescriptorOffsetAndIndex>,
    /// If the image has a stencil aspect, the descriptor to sample the stencil aspect in shaders.
    pub(crate) stencil_texture: Option<ResourceDescriptorOffsetAndIndex>,
    /// If the image has a stencil aspect, the descriptor to read/write the stencil aspect in shaders.
    pub(crate) stencil_image: Option<ResourceDescriptorOffsetAndIndex>,
}

impl DescriptorHeaps {
    fn register_sampler_descriptor(
        &self,
        sampler_create_info: vk::SamplerCreateInfo,
    ) -> SamplerDescriptorOffsetAndIndex {
        let index = self.indices.lock().unwrap().sampler.insert(()).index() as usize;
        let addr_range = self.sampler_heap.address_range_by_index(index, 1);
        unsafe {
            let device = &Device::global();
            let ext = &device.extensions.ext_descriptor_heap;
            let device = device.raw().handle().as_raw() as VkDevice;

            // Write the descriptor
            // SAFETY: access to the descriptor set is externally synchronized via `self.write_lock`
            let _lock = self.write_lock.lock().unwrap();
            (ext.write_sampler_descriptors)(
                device,
                1,
                &sampler_create_info as *const _ as *const VkSamplerCreateInfo,
                &addr_range,
            );
        }

        SamplerDescriptorOffsetAndIndex {
            offset: self.sampler_heap.descriptor_offset(index),
            index,
        }
    }

    fn reserve_resource_descriptor_slot(&self) -> ResourceDescriptorOffsetAndIndex {
        let index = self.indices.lock().unwrap().resource.insert(()).index() as usize;
        ResourceDescriptorOffsetAndIndex {
            offset: self.resource_heap.descriptor_offset(index),
            index,
        }
    }

    fn register_image_descriptors(
        &self,
        handle: vk::Image,
        image_type: ImageType,
        usage: vk::ImageUsageFlags,
        array_layers: u32,
        mip_levels: u32,
        format: Format,
    ) -> ImageDescriptors {
        let image_view_type = image_type.to_vk_image_view_type(array_layers);
        let view_for_aspect = |aspect: vk::ImageAspectFlags| {
            vk::ImageViewCreateInfo {
                flags: vk::ImageViewCreateFlags::empty(),
                image: handle,
                view_type: image_view_type,
                format,
                components: vk::ComponentMapping::default(), //  defaults to IDENTITY
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: aspect,
                    base_mip_level: 0,
                    level_count: mip_levels,
                    base_array_layer: 0,
                    layer_count: array_layers,
                },
                ..Default::default()
            }
        };

        let mut descriptor_infos: [MaybeUninit<VkResourceDescriptorInfoEXT>; 4] = [MaybeUninit::uninit(); 4];
        let mut addr_ranges: [MaybeUninit<VkHostAddressRangeEXT>; 4] = [MaybeUninit::uninit(); 4];

        let aspects = aspects_for_format(format);
        let main_view: vk::ImageViewCreateInfo;
        let stencil_view: vk::ImageViewCreateInfo;
        let main_texture_descriptor: VkImageDescriptorInfoEXT;
        let main_storage_descriptor: VkImageDescriptorInfoEXT;
        let stencil_texture_descriptor: VkImageDescriptorInfoEXT;
        let stencil_storage_descriptor: VkImageDescriptorInfoEXT;
        let mut n_descriptors = 0;

        let mut main_texture_offset = None;
        let mut main_storage_offset = None;
        let mut stencil_texture_offset = None;
        let mut stencil_storage_offset = None;

        if aspects.contains(vk::ImageAspectFlags::COLOR | vk::ImageAspectFlags::DEPTH) {
            let main_aspect = if aspects.contains(vk::ImageAspectFlags::COLOR) {
                vk::ImageAspectFlags::COLOR
            } else {
                vk::ImageAspectFlags::DEPTH
            };
            main_view = view_for_aspect(main_aspect);

            if usage.contains(vk::ImageUsageFlags::SAMPLED) {
                // COLOR or DEPTH aspect, SAMPLED access
                main_texture_descriptor = VkImageDescriptorInfoEXT {
                    sType: VK_STRUCTURE_TYPE_IMAGE_DESCRIPTOR_INFO_EXT,
                    pNext: ptr::null(),
                    pView: &main_view as *const _ as *const VkImageViewCreateInfo,
                    layout: VK_IMAGE_LAYOUT_GENERAL,
                };
                descriptor_infos[n_descriptors].write(VkResourceDescriptorInfoEXT {
                    sType: VK_STRUCTURE_TYPE_RESOURCE_DESCRIPTOR_INFO_EXT,
                    pNext: ptr::null(),
                    typ: VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                    data: VkResourceDescriptorDataEXT {
                        pImage: &main_texture_descriptor,
                    },
                });
                let slot = self.reserve_resource_descriptor_slot();
                main_texture_offset = Some(slot);
                addr_ranges[n_descriptors].write(self.resource_heap.address_range_by_index(slot.index, 1));

                n_descriptors += 1;
            }
            if usage.contains(vk::ImageUsageFlags::STORAGE) {
                // COLOR or DEPTH aspect, STORAGE access
                main_storage_descriptor = VkImageDescriptorInfoEXT {
                    sType: VK_STRUCTURE_TYPE_IMAGE_DESCRIPTOR_INFO_EXT,
                    pNext: ptr::null(),
                    pView: &main_view as *const _ as *const VkImageViewCreateInfo,
                    layout: VK_IMAGE_LAYOUT_GENERAL,
                };
                descriptor_infos[n_descriptors].write(VkResourceDescriptorInfoEXT {
                    sType: VK_STRUCTURE_TYPE_RESOURCE_DESCRIPTOR_INFO_EXT,
                    pNext: ptr::null(),
                    typ: VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    data: VkResourceDescriptorDataEXT {
                        pImage: &main_storage_descriptor,
                    },
                });
                let slot = self.reserve_resource_descriptor_slot();
                main_storage_offset = Some(slot);
                addr_ranges[n_descriptors].write(self.resource_heap.address_range_by_index(slot.index, 1));
                n_descriptors += 1;
            }
        }

        if aspects.contains(vk::ImageAspectFlags::STENCIL) {
            stencil_view = view_for_aspect(vk::ImageAspectFlags::STENCIL);
            if usage.contains(vk::ImageUsageFlags::SAMPLED) {
                // STENCIL aspect, SAMPLED access
                stencil_texture_descriptor = VkImageDescriptorInfoEXT {
                    sType: VK_STRUCTURE_TYPE_IMAGE_DESCRIPTOR_INFO_EXT,
                    pNext: ptr::null(),
                    pView: &stencil_view as *const _ as *const VkImageViewCreateInfo,
                    layout: VK_IMAGE_LAYOUT_GENERAL,
                };
                descriptor_infos[n_descriptors].write(VkResourceDescriptorInfoEXT {
                    sType: VK_STRUCTURE_TYPE_RESOURCE_DESCRIPTOR_INFO_EXT,
                    pNext: ptr::null(),
                    typ: VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                    data: VkResourceDescriptorDataEXT {
                        pImage: &stencil_texture_descriptor,
                    },
                });
                let slot = self.reserve_resource_descriptor_slot();
                addr_ranges[n_descriptors].write(self.resource_heap.address_range_by_index(slot.index, 1));
                stencil_texture_offset = Some(slot);
                n_descriptors += 1;
            }
            if usage.contains(vk::ImageUsageFlags::STORAGE) {
                // STENCIL aspect, STORAGE access
                stencil_storage_descriptor = VkImageDescriptorInfoEXT {
                    sType: VK_STRUCTURE_TYPE_IMAGE_DESCRIPTOR_INFO_EXT,
                    pNext: ptr::null(),
                    pView: &stencil_view as *const _ as *const VkImageViewCreateInfo,
                    layout: VK_IMAGE_LAYOUT_GENERAL,
                };
                descriptor_infos[n_descriptors].write(VkResourceDescriptorInfoEXT {
                    sType: VK_STRUCTURE_TYPE_RESOURCE_DESCRIPTOR_INFO_EXT,
                    pNext: ptr::null(),
                    typ: VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    data: VkResourceDescriptorDataEXT {
                        pImage: &stencil_storage_descriptor,
                    },
                });
                let slot = self.reserve_resource_descriptor_slot();
                addr_ranges[n_descriptors].write(self.resource_heap.address_range_by_index(slot.index, 1));
                stencil_storage_offset = Some(slot);
                n_descriptors += 1;
            }
        }

        unsafe {
            let device = &Device::global();
            let ext = &device.extensions.ext_descriptor_heap;
            let device = device.raw().handle().as_raw() as VkDevice;

            // Write the descriptor
            // SAFETY: access to the descriptor set is synchronized via `self.write_lock`
            let _lock = self.write_lock.lock().unwrap();
            (ext.write_resource_descriptors)(
                device,
                n_descriptors as u32,
                descriptor_infos[0].assume_init_ref() as *const VkResourceDescriptorInfoEXT,
                addr_ranges[0].assume_init_ref() as *const VkHostAddressRangeEXT,
            );
        }

        ImageDescriptors {
            texture: main_texture_offset,
            image: main_storage_offset,
            stencil_texture: stencil_texture_offset,
            stencil_image: stencil_storage_offset,
        }
    }
}

impl CommandBuffer {}
