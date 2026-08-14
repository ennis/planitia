//! Temporary buffers.

use crate::{BufferCreateInfo, BufferUntyped, BufferUsage, Device, FrameIndex, Ptr, flush, present};
use ash::vk;
use gpu_allocator::MemoryLocation;
use gpu_types::Data;
use std::alloc::Layout;
use std::cell::RefCell;
use std::collections::VecDeque;
use std::marker::PhantomData;
use std::ptr;

/// Alignment of temp buffer allocations.
const ALLOC_ALIGNMENT: usize = 256;

/// Size of temporary buffers. When this is exceeded, new buffers (aka "chunks") are allocated.
const TEMP_BUFFER_SIZE: usize = 128 * 1024; // Allocate 128 KB chunks

/// Size threshold above which a dedicated temporary buffer is allocated instead of reserving space in the current chunk.
const DEDICATED_BUFFER_THRESHOLD_SIZE: usize = 4 * 1024; // Allocate dedicated buffers above 4 KB

/// Usage flags of temporary buffers.
const TEMP_BUFFER_USAGE: BufferUsage = BufferUsage::UNIFORM;

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Aligns the offset to temp buffer allocation alignment.
fn align(size: usize) -> usize {
    (size + ALLOC_ALIGNMENT - 1) / ALLOC_ALIGNMENT * ALLOC_ALIGNMENT
}

/// Manages chunks of CPU-visible GPU memory for copying data from the host.
struct ThreadLocalAllocator {
    /// Current temporary buffer (chunk).
    current: Option<BufferUntyped>,
    /// Current allocation offset in `current`.
    offset: usize,
    /// Retired temporary chunks and the frame index in which they were retired.
    /// (frame_index, buffer)
    retired: VecDeque<(FrameIndex, BufferUntyped)>,
}

pub(crate) struct TempAlloc {
    buffer: vk::Buffer,
    /// Offset of the allocation inside the buffer.
    offset: usize,
    /// Pointer to the CPU-mapped memory of the buffer.
    host_addr: *mut u8,
    /// Device address of the buffer.
    dev_addr: vk::DeviceAddress,
}

impl ThreadLocalAllocator {
    pub(super) const fn new() -> Self {
        Self { retired: VecDeque::new(), offset: 0, current: None }
    }

    /// Allocates a new temporary buffer chunk.
    fn alloc_buffer(&mut self) {
        // Retire the current chunk.
        if let Some(buf) = self.current.take() {
            let frame_index = crate::get_frame_index();
            eprintln!("alloc_buffer: retire {:p} frame_index={}", buf.handle, frame_index);
            self.retired.push_back((frame_index, buf));
        }

        let last_completed_frame = crate::get_last_completed_frame_index();
        eprintln!("alloc_buffer: last_completed_frame={}", last_completed_frame);
        let mut free_buf = None;
        // Free all chunks older than the last completed frame, save for one, which we'll reuse.
        while let Some((retired_frame, _)) = self.retired.front() {
            if *retired_frame <= last_completed_frame {
                let (r, buf) = self.retired.pop_front().unwrap();
                // This destroys the previously popped chunk.
                eprintln!("alloc_buffer: destroy {:p} retired_frame={}, last_completed_frame={}", buf.handle, r, last_completed_frame);
                free_buf = Some(buf);
            } else {
                break
            }
        }
        if let Some(free_buf) = free_buf.as_ref() {
            eprintln!("alloc_buffer: reusing {:p}", free_buf.handle);
        }

        self.offset = 0;

        // Reuse the free chunk if there's one, or allocate a new one.
        self.current = Some(free_buf.unwrap_or_else(|| {
            BufferUntyped::new(BufferCreateInfo {
                len: TEMP_BUFFER_SIZE,
                usage: TEMP_BUFFER_USAGE,
                memory_location: MemoryLocation::CpuToGpu,
            })
        }));
    }

    /// Allocates memory in a temporary buffer.
    pub(super) fn alloc_raw(&mut self, layout: Layout) -> TempAlloc {
        let size = layout.size();
        assert!(layout.align() <= ALLOC_ALIGNMENT);

        // If the allocation size is bigger than the threshold, allocate a dedicated buffer for it.
        if size >= DEDICATED_BUFFER_THRESHOLD_SIZE {
            let buf = BufferUntyped::new(BufferCreateInfo {
                len: size,
                usage: TEMP_BUFFER_USAGE,
                memory_location: MemoryLocation::CpuToGpu,
            });
            let alloc =
                TempAlloc { buffer: buf.handle(), offset: 0, host_addr: buf.as_mut_ptr_u8(), dev_addr: buf.ptr().raw };
            // Schedule for deletion immediately since we're not going to allocate anything else in it.
            // We don't recycle dedicated buffers.
            drop(buf); // buf is dropped automatically on return, but spell it out for clarity

            // `buf` is dropped, but the underlying resources are not destroyed until the current
            // frame has completed, so this is OK.
            return alloc;
        }

        // Ensure that there's enough space in the current chunk, otherwise retire the current chunk
        // and get a new one.
        let mut aligned_offset = align(self.offset);
        if self.current.is_none() || aligned_offset + size > TEMP_BUFFER_SIZE {
            self.alloc_buffer();
            debug_assert_eq!(self.offset, 0);
            aligned_offset = 0;
        }

        let current = self.current.as_ref().unwrap();
        let offset = aligned_offset;
        self.offset = aligned_offset + size;
        let host_addr = unsafe { current.as_mut_ptr_u8().add(offset) };
        let dev_addr = current.ptr().raw + offset as u64;
        TempAlloc { buffer: current.handle(), offset, host_addr, dev_addr }
    }
}

thread_local! {
    static TEMP_ALLOCATOR: RefCell<ThreadLocalAllocator> = const { RefCell::new(ThreadLocalAllocator::new()) };
}

/// Uploads data to a temporary GPU buffer.
///
/// # Validity
///
/// The returned pointer is valid until the next call to [`end_frame`](crate::end_frame),
/// **unless** `end_frame` is called concurrently with this function. In that case,
/// the validity of the pointer is unspecified. It is the caller's responsibility to synchronize
/// calls to this function with `end_frame`.
pub fn alloc_temp<T: Data>(data: &T) -> Ptr<T> {
    TEMP_ALLOCATOR.with(|alloc| {
        let TempAlloc { dev_addr, host_addr, .. } = alloc.borrow_mut().alloc_raw(Layout::new::<T>());
        unsafe {
            ptr::copy_nonoverlapping(data as *const T, host_addr as *mut T, 1);
        }
        Ptr { raw: dev_addr, _phantom: PhantomData }
    })
}

/// Uploads a slice of data to a temporary GPU buffer.
///
/// # Validity
///
/// The returned pointer is valid until the next call to [`end_frame`](crate::end_frame),
/// **unless** `end_frame` is called concurrently with this function. In that case,
/// the validity of the pointer is unspecified. It is the caller's responsibility to synchronize
/// calls to this function with `end_frame`.
pub fn alloc_temp_slice<T: Data>(data: &[T]) -> Ptr<T> {
    TEMP_ALLOCATOR.with(|alloc| {
        let TempAlloc { dev_addr, host_addr, .. } = alloc.borrow_mut().alloc_raw(Layout::for_value(data));
        unsafe {
            ptr::copy_nonoverlapping(data.as_ptr(), host_addr as *mut T, data.len());
        }
        Ptr { raw: dev_addr, _phantom: PhantomData }
    })
}
