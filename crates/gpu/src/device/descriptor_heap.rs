use crate::device::{RESOURCE_DESCRIPTOR_HEAP_SIZE, SAMPLER_DESCRIPTOR_HEAP_SIZE};
use crate::Device;
use ash::vk;
use ash::vk::Handle;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator};
use std::ffi::c_void;
use std::ptr;
use std::sync::Mutex;
use vulkan_headers::vulkan::vulkan as vk2;
use vulkan_headers::vulkan::vulkan::{
    VK_BUFFER_USAGE_DESCRIPTOR_HEAP_BIT_EXT, VK_STRUCTURE_TYPE_BIND_HEAP_INFO_EXT, VkBindHeapInfoEXT, VkCommandBuffer, VkDevice,
    VkDeviceAddressRangeEXT,
    VkPhysicalDeviceDescriptorHeapPropertiesEXT, VkResourceDescriptorInfoEXT,
    VkSamplerCreateInfo,
};

/// Simple free list to allocate indices.
struct FreeList {
    /// Max number of indices.
    count: u32,
    /// High water mark.
    mark: u32,
    /// List of free indices.
    free: Vec<u32>,
}

impl FreeList {
    fn new(count: u32) -> Self {
        FreeList { count, mark: 0, free: Vec::new() }
    }

    fn alloc(&mut self) -> Option<u32> {
        if let Some(index) = self.free.pop() {
            Some(index)
        } else if self.mark < self.count {
            let index = self.mark;
            self.mark += 1;
            Some(index)
        } else {
            None
        }
    }

    fn free(&mut self, index: u32) {
        assert!(index < self.count, "index out of bounds");
        debug_assert!(
            index < self.mark,
            "index {} is greater than high water mark {}",
            index,
            self.mark
        );
        if index == self.mark - 1 {
            self.mark -= 1;
        } else {
            self.free.push(index);
        }
    }
}

///
struct DescriptorHeapInfo {
    alloc: Allocation,
    buffer: vk::Buffer,
    ptr: *mut c_void,
    device_addr: vk::DeviceAddress,
    /// Offset to the beginning of descriptors (skips the reserved range).
    start_offset: usize,
    /// Stride between consecutive descriptors.
    stride: usize,
    /// Alignment of descriptors.
    alignment: usize,
    /// Size in bytes of the heap.
    size: usize,
    /// Index of the first valid descriptor, skipping the reserved range
    /// (`start_offset / stride`).
    index_offset: u32,
}

unsafe impl Send for DescriptorHeapInfo {}
unsafe impl Sync for DescriptorHeapInfo {}

impl DescriptorHeapInfo {
    /// Returns the offset of the descriptor at the given global index within the heap.
    fn offset_by_index(&self, index: u32) -> usize {
        index as usize * self.stride
    }

    /// Returns a VkHostAddressRange for the descriptor at the given global index.
    fn address_range_by_index(&self, start_index: u32, index_count: u32) -> vk2::VkHostAddressRangeEXT {
        vk2::VkHostAddressRangeEXT {
            address: unsafe { self.ptr.add(self.offset_by_index(start_index)) },
            size: self.stride * index_count as usize,
        }
    }

    /// Returns the index of the descriptor given a host address to the descriptor.
    fn index_by_address(&self, address: *const c_void) -> u32 {
        let offset = (address as isize) - (self.ptr as isize);
        self.index_by_offset(offset)
    }

    /// Returns the index of the descriptor in the allocation table, given its offset.
    fn index_by_offset(&self, offset: isize) -> u32 {
        assert!(offset >= self.start_offset as isize, "offset is before the start of the heap");
        assert!(offset <= self.size as isize, "offset is beyond the end of the heap");
        assert!(offset % self.stride as isize == 0, "offset is not aligned to descriptor stride");
        (offset / self.stride as isize) as u32
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
            requirements: vk::MemoryRequirements { size: byte_size as u64, alignment, memory_type_bits: u32::MAX },
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
        device_addr = device.get_buffer_device_address(&vk::BufferDeviceAddressInfo { buffer, ..Default::default() });
    }
    let ptr = alloc.mapped_ptr().expect("failed to map descriptor heap memory").as_ptr();

    let start_offset;
    let stride;
    let alignment;
    match heap_type {
        DescriptorHeapType::Resource => {
            alignment = descriptor_heap_properties.imageDescriptorAlignment as usize;
            start_offset =
                descriptor_heap_properties.minResourceHeapReservedRange.next_multiple_of(alignment as u64) as usize;
            stride = descriptor_heap_properties.imageDescriptorSize as usize;
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

    let index_offset = (start_offset / stride) as u32;

    DescriptorHeapInfo { alloc, buffer, ptr, device_addr, start_offset, stride, alignment, index_offset, size: byte_size }
}

/// Device state related to resource and sampler descriptor heaps.
pub(crate) struct DescriptorHeaps {
    /// Descriptor writes must be externally synchronized, but we don't want to
    /// wrap DescriptorHeapInfo in a Mutex because that would require locking every time we want
    /// to copy the address. So instead we lock this mutex when writing to the descriptor set.
    write_lock: Mutex<()>,
    resource: DescriptorHeapInfo,
    sampler: DescriptorHeapInfo,

    /// Allocation table.
    resource_slots: Mutex<FreeList>,
    sampler_slots: Mutex<FreeList>,
}

impl DescriptorHeaps {
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

        let max_resource_descriptors = (resource_heap.size - resource_heap.start_offset) / resource_heap.stride;
        let max_sampler_descriptors = (sampler_heap.size - sampler_heap.start_offset) / sampler_heap.stride;
        DescriptorHeaps {
            write_lock: Mutex::new(()),
            resource: resource_heap,
            sampler: sampler_heap,
            resource_slots: Mutex::new(FreeList::new(max_resource_descriptors as u32)),
            sampler_slots: Mutex::new(FreeList::new(max_sampler_descriptors as u32)),
        }
    }
}

pub type ResourceDescriptorHandle = u32;
pub type SamplerDescriptorHandle = u32;

impl Device {


    /// Allocates a sampler descriptor in the corresponding descriptor heap.
    ///
    /// Returns the host pointer to the descriptor slot.
    fn allocate_sampler_descriptor_slot(
        &self,
    ) -> u32 {
        let i = self.descriptor_heaps.sampler_slots.lock().unwrap().alloc().expect("exceeded maximum number of sampler descriptors");
        self.descriptor_heaps.sampler.index_offset + i
    }

    fn allocate_resource_descriptor_slot(&self) -> u32 {
        let i = self.descriptor_heaps.resource_slots.lock().unwrap().alloc().expect("exceeded maximum number of resource descriptors");
        self.descriptor_heaps.resource.index_offset + i
    }

    /// Frees a sampler descriptor from the heap.
    pub(crate) fn free_sampler_descriptor(&self, handle: SamplerDescriptorHandle) {
        let index = handle - self.descriptor_heaps.sampler.index_offset;
        self.descriptor_heaps.sampler_slots.lock().unwrap().free(index);
    }

    pub(crate) fn free_resource_descriptor(&self, handle: ResourceDescriptorHandle) {
        let index = handle - self.descriptor_heaps.resource.index_offset;
        self.descriptor_heaps.resource_slots.lock().unwrap().free(index);
    }

    /// Allocates a sampler descriptor.
    pub(crate) fn allocate_sampler_descriptor(&self, info: &vk::SamplerCreateInfo) -> SamplerDescriptorHandle {
        let index = self.allocate_sampler_descriptor_slot();
        unsafe {
            // Write the descriptor
            // SAFETY: access to the descriptor set is externally synchronized via `self.write_lock`
            let _lock = self.descriptor_heaps.write_lock.lock().unwrap();
            (self.ext.descriptor_heap.write_sampler_descriptors)(
                self.raw.handle().as_raw() as VkDevice,
                1,
                info as *const _ as *const VkSamplerCreateInfo,
                &self.descriptor_heaps.sampler.address_range_by_index(index, 1)
            );
        }
        index
    }

    pub(crate) fn allocate_resource_descriptor(&self, info: &VkResourceDescriptorInfoEXT) -> ResourceDescriptorHandle {
        let index = self.allocate_resource_descriptor_slot();
        unsafe {
            // Write the descriptor
            // SAFETY: access to the descriptor set is externally synchronized via `self.write_lock`
            let _lock = self.descriptor_heaps.write_lock.lock().unwrap();
            (self.ext.descriptor_heap.write_resource_descriptors)(
                self.raw.handle().as_raw() as VkDevice,
                1,
                info as *const _,
                &self.descriptor_heaps.resource.address_range_by_index(index, 1)
            );
        }
        index
    }
}

impl Device {
    pub(crate) fn bind_descriptor_heaps(&self, cmdbuf: vk::CommandBuffer) {
        let cb = cmdbuf.as_raw() as VkCommandBuffer;
        unsafe {
            (self.ext.descriptor_heap.cmd_bind_resource_heap)(
                cb,
                &VkBindHeapInfoEXT {
                    sType: VK_STRUCTURE_TYPE_BIND_HEAP_INFO_EXT,
                    pNext: ptr::null(),
                    heapRange: VkDeviceAddressRangeEXT {
                        address: self.descriptor_heaps.resource.device_addr,
                        size: self.descriptor_heaps.resource.alloc.size(),
                    },
                    reservedRangeOffset: 0,
                    reservedRangeSize: self.thread_safe.descriptor_heap_properties.minResourceHeapReservedRange,
                },
            );
            (self.ext.descriptor_heap.cmd_bind_sampler_heap)(
                cb,
                &VkBindHeapInfoEXT {
                    sType: VK_STRUCTURE_TYPE_BIND_HEAP_INFO_EXT,
                    pNext: ptr::null(),
                    heapRange: VkDeviceAddressRangeEXT {
                        address: self.descriptor_heaps.sampler.device_addr,
                        size: self.descriptor_heaps.sampler.alloc.size(),
                    },
                    reservedRangeOffset: 0,
                    reservedRangeSize: self.thread_safe.descriptor_heap_properties.minSamplerHeapReservedRange,
                },
            );
        }
    }
}
