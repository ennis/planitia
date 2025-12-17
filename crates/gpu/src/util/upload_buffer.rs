use crate::{Buffer, BufferCreateInfo, BufferUntyped, BufferUsage, Ptr};
use ash::vk;
use gpu_allocator::MemoryLocation;
use std::alloc::Layout;
use std::marker::PhantomData;
use std::ptr;

const ALIGNMENT: usize = 256;
const CHUNK_SIZE: usize = 1 * 1024 * 1024; // Allocate 1 MB chunks

pub struct UploadBuffer {
    chunks: Vec<BufferUntyped>,
    offset: usize,
    usage: BufferUsage,
}

fn align(size: usize) -> usize {
    (size + ALIGNMENT - 1) / ALIGNMENT * ALIGNMENT
}

impl UploadBuffer {
    pub fn new(usage: BufferUsage) -> Self {
        Self {
            chunks: Vec::new(),
            offset: 0,
            usage,
        }
    }

    /// Ensures that there is space for an allocation of `size` bytes in the current buffer,
    /// or creates a new buffer if necessary.
    ///
    /// Returns the offset in the current buffer where the allocation can be made.
    fn allocate_raw(&mut self, layout: Layout) -> (usize, *mut u8, vk::DeviceAddress) {
        let size = layout.size();
        assert!(layout.align() <= ALIGNMENT);

        let aligned_offset = align(self.offset);
        if let Some(chunk) = self.chunks.last() {
            if aligned_offset + size <= chunk.len() {
                let offset = aligned_offset;
                self.offset = aligned_offset + size;
                let addr = unsafe { chunk.as_mut_ptr_u8().add(offset) };
                let device_address = chunk.ptr().raw + offset as u64;
                return (offset, addr, device_address);
            }
        }

        let new_chunk_size = align(CHUNK_SIZE.max(size));
        let chunk = BufferUntyped::new(BufferCreateInfo {
            len: new_chunk_size,
            usage: self.usage,
            memory_location: MemoryLocation::CpuToGpu,
            label: "upload_buffer_chunk",
        });
        let addr = chunk.as_mut_ptr_u8();
        let device_address = chunk.ptr().raw;
        self.chunks.push(chunk);
        self.offset = size;
        (0, addr, device_address)
    }

    pub fn allocate<T: Copy>(&mut self, data: &T) -> Ptr<T> {
        let (_, ptr, raw_addr) = self.allocate_raw(Layout::new::<T>());
        unsafe {
            ptr::copy_nonoverlapping(data as *const T, ptr as *mut T, size_of::<T>());
        }
        Ptr {
            raw: raw_addr,
            _phantom: PhantomData,
        }
    }

    pub fn allocate_slice<T: Copy>(&mut self, data: &[T]) -> Ptr<[T]> {
        let (_, ptr, raw_addr) = self.allocate_raw(Layout::for_value(data));
        unsafe {
            ptr::copy_nonoverlapping(data.as_ptr(), ptr as *mut T, data.len());
        }
        Ptr {
            raw: raw_addr,
            _phantom: PhantomData,
        }
    }
}
