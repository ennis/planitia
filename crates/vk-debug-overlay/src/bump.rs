//! GPU bump allocator.
use crate::helper::{Buffer, DeviceHelper};
use ash::vk;
use std::alloc::Layout;

const ALLOC_ALIGNMENT: usize = 256;
const TEMP_BUFFER_SIZE: usize = 128 * 1024; // Allocate 128 KB chunks

/// Size threshold above which a dedicated temporary buffer is allocated instead of reserving space in the current chunk.
const DEDICATED_BUFFER_THRESHOLD_SIZE: usize = 4 * 1024; // Allocate dedicated buffers above 4 KB

/// Usage flags of temporary buffers.
const TEMP_BUFFER_USAGE: vk::BufferUsageFlags = vk::BufferUsageFlags::from_raw(
    vk::BufferUsageFlags::TRANSFER_SRC.as_raw()
        | vk::BufferUsageFlags::TRANSFER_DST.as_raw()
        | vk::BufferUsageFlags::STORAGE_BUFFER.as_raw()
        | vk::BufferUsageFlags::UNIFORM_BUFFER.as_raw(),
);

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Aligns the offset to temp buffer allocation alignment.
fn align(size: usize) -> usize {
    (size + ALLOC_ALIGNMENT - 1) / ALLOC_ALIGNMENT * ALLOC_ALIGNMENT
}

struct Chunk {
    buf: Buffer,
    age: usize,
}

impl Chunk {
    fn new(buf: Buffer) -> Chunk {
        Chunk { buf, age: 0 }
    }
}

/// GPU arena (bump) allocator.
///
/// There's no way to synchronize deletion other than `vkDeviceWaitIdle`
pub struct BumpAllocator {
    // Current temporary buffer (chunk)
    current: Option<Chunk>,
    // Current allocation offset in `current`
    offset: usize,
    // Full buffers.
    retired: Vec<Chunk>,
    free: Vec<Chunk>,
}

#[derive(Copy, Clone)]
pub struct Alloc {
    pub buffer: vk::Buffer,
    pub offset: usize,
    pub host_addr: *mut u8,
    pub dev_addr: vk::DeviceAddress,
}

unsafe impl Send for Alloc {}
unsafe impl Sync for Alloc {}

impl BumpAllocator {
    pub const fn new() -> Self {
        Self { offset: 0, current: None, retired: Vec::new(), free: Vec::new() }
    }

    /// Allocates a new temporary buffer chunk.
    pub fn alloc(&mut self, dd: &DeviceHelper, layout: Layout) -> Alloc {
        // Retire the current chunk.
        if let Some(buf) = self.current.take() {
            self.retired.push(buf);
        }

        // If the allocation size is bigger than the threshold, allocate a dedicated buffer for it.
        if layout.size() >= DEDICATED_BUFFER_THRESHOLD_SIZE {
            let mut free = None;
            // Find a buffer with the exact size in the list and reuse it.
            // For our use cases, the sizes should be highly coherent from one frame to the other.
            for i in 0..self.free.len() {
                if self.free[i].buf.size == layout.size() {
                    free = Some(self.free.remove(i));
                    break;
                }
            }

            let chunk = free.unwrap_or_else(|| {
                // Resort to a dedicated allocation.
                unsafe { Chunk::new(dd.create_buffer_helper(TEMP_BUFFER_USAGE, layout.size(), None)) }
            });

            let alloc = Alloc {
                buffer: chunk.buf.buffer,
                offset: 0,
                host_addr: chunk.buf.ptr as *mut u8,
                dev_addr: chunk.buf.device_address,
            };
            self.retired.push(chunk);
            return alloc;
        }

        // Ensure that there's enough space in the current chunk, otherwise retire the current chunk
        // and get a new one.
        let mut aligned_offset = align(self.offset);
        if self.current.is_none() || aligned_offset + layout.size() > TEMP_BUFFER_SIZE {
            if let Some(current) = self.current.take() {
                self.retired.push(current);
            }
            self.current =
                unsafe { Some(Chunk::new(dd.create_buffer_helper(TEMP_BUFFER_USAGE, TEMP_BUFFER_SIZE, None))) };
            self.offset = 0;
            debug_assert_eq!(self.offset, 0);
            aligned_offset = 0;
        }

        let current = self.current.as_ref().unwrap();
        self.offset = aligned_offset + layout.size();
        let host_addr = unsafe { (current.buf.ptr as *mut u8).add(aligned_offset) };
        Alloc {
            buffer: current.buf.buffer,
            offset: aligned_offset,
            host_addr,
            dev_addr: current.buf.device_address + aligned_offset as u64,
        }
    }

    pub fn reset(&mut self) {
        self.offset = 0;
        if let Some(current) = self.current.take() {
            self.free.push(current);
        }
        self.free.extend(self.retired.drain(..));
    }
}
