use std::alloc::Layout;
use std::marker::PhantomData;
use std::ptr;
use ash::vk;
use gpu_allocator::MemoryLocation;
use crate::{BufferCreateInfo, BufferUntyped, BufferUsage, Ptr};

/// Alignment of upload buffer allocations.
pub(super) const UPLOAD_BUFFER_ALIGNMENT: usize = 256;

/// Upload buffer chunk size.
pub(super) const UPLOAD_BUFFER_CHUNK_SIZE: usize = 1 * 1024 * 1024; // Allocate 1 MB chunks

////////////////////////////////////////////////////////////////////////////////////////////////////

fn align(size: usize) -> usize {
    (size + UPLOAD_BUFFER_ALIGNMENT - 1) / UPLOAD_BUFFER_ALIGNMENT * UPLOAD_BUFFER_ALIGNMENT
}

/// Manages chunks of CPU-visible GPU memory for copying data from the host.
pub(super) struct UploadBuffer {
    pub(super) full: Vec<BufferUntyped>,
    current: Option<BufferUntyped>,
    offset: usize,
    usage: BufferUsage,
}

impl UploadBuffer {
    pub(super) fn new(usage: BufferUsage) -> Self {
        Self {
            full: vec![],
            offset: 0,
            usage,
            current: None,
        }
    }

    /// Ensures that there is space for an allocation of `size` bytes in the current buffer,
    /// or creates a new buffer if necessary.
    ///
    /// Returns the offset in the current buffer where the allocation can be made.
    pub(super) fn allocate_raw(&mut self, layout: Layout) -> (usize, *mut u8, vk::DeviceAddress) {
        let size = layout.size();
        assert!(layout.align() <= UPLOAD_BUFFER_ALIGNMENT);

        let aligned_offset = align(self.offset);
        if let Some(chunk) = self.current.as_ref() {
            if aligned_offset + size <= chunk.len() {
                let offset = aligned_offset;
                self.offset = aligned_offset + size;
                let addr = unsafe { chunk.as_mut_ptr_u8().add(offset) };
                let device_address = chunk.ptr().raw + offset as u64;
                return (offset, addr, device_address);
            }
        }

        let new_chunk_size = align(UPLOAD_BUFFER_CHUNK_SIZE.max(size));
        if let Some(chunk) = self.current.take() {
            self.full.push(chunk);
        }
        let chunk = BufferUntyped::new(BufferCreateInfo {
            len: new_chunk_size,
            usage: self.usage,
            memory_location: MemoryLocation::CpuToGpu,
        });
        let addr = chunk.as_mut_ptr_u8();
        let device_address = chunk.ptr().raw;
        self.current = Some(chunk);
        self.offset = size;
        (0, addr, device_address)
    }

    pub(super) fn allocate<T: Copy>(&mut self, data: &T) -> Ptr<T> {
        let (_, ptr, raw_addr) = self.allocate_raw(Layout::new::<T>());
        unsafe {
            ptr::copy_nonoverlapping(data as *const T, ptr as *mut T, 1);
        }
        Ptr {
            raw: raw_addr,
            _phantom: PhantomData,
        }
    }

    pub(super) fn allocate_slice<T: Copy>(&mut self, data: &[T]) -> Ptr<T> {
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