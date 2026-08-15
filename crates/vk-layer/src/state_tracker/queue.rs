use std::sync::LazyLock;
use ash::vk;
use ash::vk::PFN_vkGetDeviceQueue;
use dashmap::DashMap;
use crate::{layer_fn, state};

struct LayerQueue {
    device: vk::Device,
    queue_family_index: u32,
}

static QUEUE_MAP: LazyLock<DashMap<vk::Queue, LayerQueue>> = LazyLock::new(DashMap::new);

layer_fn! {
    #[proc(PFN_vkGetDeviceQueue)]
    fn layer_vkGetDeviceQueue(
        device: vk::Device,
        queue_family_index: u32,
        queue_index: u32,
        p_queue: *mut vk::Queue,
    ) {
        (state(device).fp_v1_0().get_device_queue)(device, queue_family_index, queue_index, p_queue);

        if let Some(&queue) = p_queue.as_ref() {
            if queue != vk::Queue::null() {
                QUEUE_MAP.insert(queue, LayerQueue { device, queue_family_index });
            }
        }
    }
}

pub fn get_device_for_queue(queue: vk::Queue) -> Option<vk::Device> {
    QUEUE_MAP.get(&queue).map(|entry| entry.device)
}