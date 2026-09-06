use gpu::Device;
use std::cell::RefCell;
use std::mem::MaybeUninit;
use std::ptr;
use vulkan::*;


/// Allocates command buffers in a `vk::CommandPool` and allows re-use of freed command buffers.
#[derive(Debug)]
struct ThreadLocalCommandPool {
    queue_family: u32,
    command_pool: VkCommandPool,
    /// (frame_index, cmdbuf)
    pending: Vec<(u64, VkCommandBuffer)>,
}

impl ThreadLocalCommandPool {
    /// Creates a new command pool for the specified queue family.
    unsafe fn new(device: &Device, queue_family_index: u32) -> ThreadLocalCommandPool {
        // create a new one
        let create_info = VkCommandPoolCreateInfo {
            flags: VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
            queueFamilyIndex: queue_family_index,
            ..Default::default()
        };
        let command_pool = device.vk.CreateCommandPool(device.vkd, &create_info, ptr::null()).unwrap();
        ThreadLocalCommandPool { queue_family: queue_family_index, command_pool, pending: vec![] }
    }

    /// Allocates a command buffer from the pool.
    ///
    /// Once you have finished using the command buffer (recorded and submitted),
    /// it should be returned to the pool with [`defer_free`].
    fn alloc(&mut self, device: &Device) -> VkCommandBuffer {
        // reclaim free cmdbufs first
        unsafe {
            self.reclaim(&device);
            let allocate_info = VkCommandBufferAllocateInfo {
                commandPool: self.command_pool,
                level: VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                commandBufferCount: 1,
                ..Default::default()
            };
            let mut cmd = MaybeUninit::uninit();
            device.vk.AllocateCommandBuffers(device.vkd, &allocate_info, cmd.as_mut_ptr()).check();
            cmd.assume_init()
        }
    }

    /// Returns a command buffer from the pool.
    ///
    /// # Arguments
    /// * cmdbuf - command buffer to recycle
    /// * defer_until_frame_completed - recycling is delayed until this frame has completed.
    fn defer_free(&mut self, cmdbuf: VkCommandBuffer, defer_until_frame_completed: u64) {
        self.pending.push((defer_until_frame_completed, cmdbuf));
    }

    /// Frees command buffers which are not in use anymore by the GPU (their associated frame has completed).
    unsafe fn reclaim(&mut self, device: &Device) {
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
                device.vk.FreeCommandBuffers(device.vkd, self.command_pool, 1, &*cmdbuf);
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
pub(crate) fn allocate_command_buffer() -> VkCommandBuffer {
    COMMAND_POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        let device = Device::instance();
        if pool.is_none() {
            *pool = Some(unsafe { ThreadLocalCommandPool::new(device, device.queue_family) });
        }
        pool.as_mut().unwrap().alloc(device)
    })
}

#[inline(never)]
pub(crate) fn defer_free_command_buffer(cmdbuf: VkCommandBuffer, defer_until_frame_completed: u64) {
    COMMAND_POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        pool.as_mut().expect("thread-local command pool not initialized").defer_free(cmdbuf, defer_until_frame_completed);
    })
}