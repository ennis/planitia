use std::cell::RefCell;
use ash::vk;
use gpu::Device;

/// Allocates command buffers in a `vk::CommandPool` and allows re-use of freed command buffers.
#[derive(Debug)]
struct ThreadLocalCommandPool {
    queue_family: u32,
    command_pool: vk::CommandPool,
    /// (frame_index, cmdbuf)
    pending: Vec<(u64, vk::CommandBuffer)>,
}

impl ThreadLocalCommandPool {
    /// Creates a new command pool for the specified queue family.
    unsafe fn new(device: &ash::Device, queue_family_index: u32) -> ThreadLocalCommandPool {
        // create a new one
        let create_info = vk::CommandPoolCreateInfo {
            flags: vk::CommandPoolCreateFlags::TRANSIENT,
            queue_family_index,
            ..Default::default()
        };
        let command_pool = device.create_command_pool(&create_info, None).expect("failed to create a command pool");
        ThreadLocalCommandPool { queue_family: queue_family_index, command_pool, pending: vec![] }
    }

    /// Allocates a command buffer from the pool.
    ///
    /// Once you have finished using the command buffer (recorded and submitted),
    /// it should be returned to the pool with [`defer_free`].
    fn alloc(&mut self, device: &ash::Device) -> vk::CommandBuffer {
        // reclaim free cmdbufs first
        unsafe {
            self.reclaim(device);
        }

        let allocate_info = vk::CommandBufferAllocateInfo {
            command_pool: self.command_pool,
            level: vk::CommandBufferLevel::PRIMARY,
            command_buffer_count: 1,
            ..Default::default()
        };
        unsafe { device.allocate_command_buffers(&allocate_info).unwrap()[0] }
    }

    /// Returns a command buffer from the pool.
    ///
    /// # Arguments
    /// * cmdbuf - command buffer to recycle
    /// * defer_until_frame_completed - recycling is delayed until this frame has completed.
    fn defer_free(&mut self, cmdbuf: vk::CommandBuffer, defer_until_frame_completed: u64) {
        self.pending.push((defer_until_frame_completed, cmdbuf));
    }

    /// Frees command buffers which are not in use anymore by the GPU (their associated frame has completed).
    unsafe fn reclaim(&mut self, device: &ash::Device) {
        let completed_frame_index = gpu::get_last_completed_frame_index();
        self.pending.retain(|(frame_index, cmdbuf)| {
            if *frame_index <= completed_frame_index {
                // [RANT] The common wisdom says that you should reset command buffers and recycle them
                // instead of freeing/allocating. Or better, you should reset whole command pools.
                // However, this would require one pool per thread and per frame-in-flight, which
                // complicates the logic in this module.
                //
                // So we don't care and free command buffers anyway. Ideally the driver should be
                // in charge of optimizing this (because it should know best), but Vulkan forces that onto
                // the application instead...
                device.free_command_buffers(self.command_pool, &[*cmdbuf]);
                false
            } else {
                true
            }
        });
    }
}

thread_local! {
    static COMMAND_POOL: RefCell<Option<ThreadLocalCommandPool>> = const { RefCell::new(None) };
}

#[inline(never)]
pub(crate) fn allocate_command_buffer() -> vk::CommandBuffer {
    COMMAND_POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        let device = Device::instance();
        if pool.is_none() {
            *pool = Some(unsafe { ThreadLocalCommandPool::new(&device.raw, device.queue_family) });
        }
        pool.as_mut().unwrap().alloc(&device.raw)
    })
}

#[inline(never)]
pub(crate) fn defer_free_command_buffer(cmdbuf: vk::CommandBuffer, defer_until_frame_completed: u64) {
    COMMAND_POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        pool.as_mut().expect("thread-local command pool not initialized").defer_free(cmdbuf, defer_until_frame_completed);
    })
}