use crate::{Device, VulkanObject};
use ash::vk;

/// Query pools.
#[derive(Debug)]
pub struct QueryPool {
    pub(crate) pool: vk::QueryPool,
    pub(crate) ty: vk::QueryType,
    pub(crate) size: usize,
}

impl QueryPool {
    pub fn new(query_type: vk::QueryType, pool_size: usize) -> QueryPool {
        let device = Device::instance();
        let create_info = &vk::QueryPoolCreateInfo { query_type, query_count: pool_size as u32, ..Default::default() };
        unsafe {
            let pool = device.raw.create_query_pool(&create_info, None).unwrap();
            device.raw.reset_query_pool(pool, 0, pool_size as u32);
            QueryPool { pool, ty: query_type, size: pool_size }
        }
    }

    pub fn wait_for_results<T: Copy>(&self, first_query: u32, results: &mut [T]) {
        let device = Device::instance();
        unsafe {
            device
                .raw
                .get_query_pool_results(
                    self.pool,
                    first_query,
                    &mut results[..],
                    // FIXME: flags depend on the query type
                    vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                )
                .expect("Failed to get query results");
        }
    }

    pub fn reset(&self) {
        let device = Device::instance();
        unsafe {
            device.raw.reset_query_pool(self.pool, 0, self.size as u32);
        }
    }
}

impl Drop for QueryPool {
    fn drop(&mut self) {
        let device = Device::instance();
        let pool = self.pool;
        device.delete_after_current_frame(move |device| unsafe {
            device.raw.destroy_query_pool(pool, None);
        });
    }
}

impl VulkanObject for QueryPool {
    type Handle = vk::QueryPool;
    fn handle(&self) -> Self::Handle {
        self.pool
    }
}
