use crate::{Device, VulkanObject};
use std::ptr;
use gpu::vkcall;
use vulkan::*;

/// Query pools.
#[derive(Debug)]
pub struct QueryPool {
    pub(crate) pool: VkQueryPool,
    pub(crate) ty: VkQueryType,
    pub(crate) size: usize,
}

impl QueryPool {
    pub fn new(query_type: VkQueryType, pool_size: usize) -> QueryPool {
        let device = Device::instance();
        let create_info = VkQueryPoolCreateInfo { queryType: query_type, queryCount: pool_size as u32, .. };
        unsafe {
            vkcall!(let _ = device.fns.CreateQueryPool(device.vkd, &create_info, ptr::null(), @out let pool));
            device.fns.ResetQueryPool(device.vkd, pool, 0, pool_size as u32);
            QueryPool { pool, ty: query_type, size: pool_size }
        }
    }

    pub fn wait_for_results<T: Copy>(&self, first_query: u32, results: &mut [T]) {
        let device = Device::instance();
        unsafe {
            device
                .fns
                .GetQueryPoolResults(
                    device.vkd,
                    self.pool,
                    first_query,
                    results.len() as u32,
                    results.len() * size_of::<T>(),
                    results.as_mut_ptr() as *mut std::ffi::c_void,
                    size_of::<T>() as u64,
                    // FIXME: flags depend on the query type
                    VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT,
                )
                .check();
        }
    }

    pub fn reset(&self) {
        let device = Device::instance();
        unsafe {
            device.fns.ResetQueryPool(device.vkd, self.pool, 0, self.size as u32);
        }
    }
}

impl Drop for QueryPool {
    fn drop(&mut self) {
        let device = Device::instance();
        let pool = self.pool;
        device.delete_after_current_frame(move |device| unsafe {
            device.fns.DestroyQueryPool(device.vkd, pool, ptr::null());
        });
    }
}

impl VulkanObject for QueryPool {
    type Handle = VkQueryPool;
    fn handle(&self) -> Self::Handle {
        self.pool
    }
}
