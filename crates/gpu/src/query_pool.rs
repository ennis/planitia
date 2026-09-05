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
            device.raw.get_query_pool_results(
                self.pool,
                first_query,
                &mut results[..],
                // FIXME: flags depend on the query type
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
            ).expect("Failed to get query results");
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

/*
/// Returns a query slot of the given type.
///
/// The slot is valid for the current frame only.
///
/// # Safety
///
/// This must not be called concurrently with [`end_frame`](crate::end_frame).
pub fn allocate_query(ty: vk::QueryType) -> (vk::QueryPool, u32) {
    ALLOCATORS.with(|alloc| {
        let device = Device::instance();
        let mut alloc = alloc[query_type_index(ty)].borrow_mut();
        alloc.allocate(1, &device.raw)
    })
}

fn query_type_index(ty: vk::QueryType) -> usize {
    let index = match ty {
        vk::QueryType::TIMESTAMP => 0,
        _ => panic!("Unsupported query type"),
    };
    debug_assert!(index < QUERY_TYPE_COUNT);
    index
}

const QUERY_TYPE_COUNT: usize = 1;
const TIMESTAMP_QUERY_POOL_SIZE: u32 = 1024;

thread_local! {
    static ALLOCATORS: [RefCell<QueryPoolAllocator>; QUERY_TYPE_COUNT] = const { [
        RefCell::new(QueryPoolAllocator::new(vk::QueryType::TIMESTAMP, TIMESTAMP_QUERY_POOL_SIZE)),
    ] };
}

struct QueryPoolAllocator {
    pool: Option<vk::QueryPool>,
    index: u32,
    pool_size: u32,
    ty: vk::QueryType,
    retired: VecDeque<(FrameIndex, vk::QueryPool)>,
}

impl QueryPoolAllocator {
    const fn new(query_type: vk::QueryType, pool_size: u32) -> Self {
        Self { pool: None, index: 0, pool_size, ty: query_type, retired: VecDeque::new() }
    }

    #[cold]
    fn alloc_pool(&mut self, device: &ash::Device) {
        // Retire the current pool.
        let frame_index = crate::get_frame_index();
        if let Some(pool) = self.pool.take() {
            self.retired.push_back((frame_index, pool));
        }
        self.index = 0;

        // Reuse a retired pool if available:
        // free all pools older than the last completed frame, save for one, which we'll reuse.
        let last_completed_frame = crate::get_last_completed_frame_index();
        while let Some((retired_frame, _)) = self.retired.front() {
            if *retired_frame <= last_completed_frame {
                let (_r, pool) = self.retired.pop_front().unwrap();
                if let Some(pool) = self.pool.take() {
                    unsafe {
                        device.destroy_query_pool(pool, None);
                    }
                }
                self.pool = Some(pool);
            } else {
                break;
            }
        }

        // No retired pool available, create a new one
        if self.pool.is_none() {
            let create_info =
                &vk::QueryPoolCreateInfo { query_type: self.ty, query_count: self.pool_size, ..Default::default() };
            let pool = unsafe { device.create_query_pool(&create_info, None).unwrap() };
            self.pool = Some(pool);
        }

        unsafe {
            device.reset_query_pool(self.pool.unwrap(), 0, self.pool_size);
        }
    }

    fn allocate(&mut self, count: u32, device: &ash::Device) -> (vk::QueryPool, u32) {
        assert!(count <= self.pool_size, "requested query count {} exceeds pool size {}", count, self.pool_size);
        if self.pool.is_none() || self.index + count > self.pool_size {
            self.alloc_pool(device);
        }
        let index = self.index;
        self.index += count;
        (self.pool.unwrap(), index)
    }
}*/
