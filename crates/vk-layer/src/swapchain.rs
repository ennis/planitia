//! Swapchain hooks
use crate::{device_data, LayerSwapchain, QUEUE_MAP};
use ash::vk;
use ash::vk::Handle;
use std::{ptr, slice};

#[no_mangle]
pub(crate) unsafe extern "system" fn layer_vkCreateSwapchainKHR(
    device: vk::Device,
    p_create_info: *const vk::SwapchainCreateInfoKHR,
    p_allocator: *const vk::AllocationCallbacks,
    p_swapchain: *mut vk::SwapchainKHR,
) -> vk::Result {
    let d = device_data(device);
    let mut inner = d.tracked_resources.lock().unwrap();

    let mut create_info = *p_create_info;

    // The pan/zoom shader needs TRANSFER_SRC usage.
    create_info.image_usage |= vk::ImageUsageFlags::TRANSFER_SRC;

    // If there's an old swapchain to be deleted, delete the resources that we have for it
    if !create_info.old_swapchain.is_null() {
        if let Some(index) = inner.swapchains.iter().position(|sc| sc.swapchain == create_info.old_swapchain) {
            let sc = inner.swapchains.remove(index);
            for view in sc.image_views {
                d.dispatch.device.destroy_image_view(view, None);
            }
            // FIXME: we have no way to know whether the semaphore are still being waited on.
            //        Technically waitForIdle isn't sufficient as it doesn't sync with presentation.
            //        So it's basically impossible to do this correctly.
            //        See https://stackoverflow.com/questions/75437792/how-to-synchronize-vulkan-swapchain-presentation-with-sempahore-destruction
            //        As a best effort, wait for device idle first.
            d.dispatch.device.device_wait_idle().unwrap();
            for sem in sc.render_to_present {
                d.dispatch.device.destroy_semaphore(sem, None);
            }
        }
    }

    // Call next layer
    let result = (d.dispatch.khr_swapchain.create_swapchain_khr)(device, &create_info, p_allocator, p_swapchain);
    if result != vk::Result::SUCCESS {
        return result;
    }

    // Retrieve the backing images and create image views for them.
    let mut image_count = 0;
    let r =
        (d.dispatch.khr_swapchain.get_swapchain_images_khr)(device, *p_swapchain, &mut image_count, ptr::null_mut());
    assert_eq!(r, vk::Result::SUCCESS);
    let images = {
        let mut images = Vec::with_capacity(image_count as usize);
        let r = (d.dispatch.khr_swapchain.get_swapchain_images_khr)(
            device,
            *p_swapchain,
            &mut image_count,
            images.as_mut_ptr(),
        );
        assert_eq!(r, vk::Result::SUCCESS);
        images.set_len(image_count as usize);
        images
    };

    let image_views = {
        images
            .iter()
            .map(|&image| {
                d.dispatch
                    .device
                    .create_image_view(
                        &vk::ImageViewCreateInfo {
                            image,
                            view_type: vk::ImageViewType::TYPE_2D,
                            format: create_info.image_format,
                            subresource_range: vk::ImageSubresourceRange {
                                aspect_mask: vk::ImageAspectFlags::COLOR,
                                base_mip_level: 0,
                                level_count: 1,
                                base_array_layer: 0,
                                layer_count: 1,
                            },
                            ..Default::default()
                        },
                        None,
                    )
                    .expect("create_image_view failed")
            })
            .collect::<Vec<_>>()
    };

    let render_to_present_semaphores = (0..images.len())
        .map(|_| {
            d.dispatch
                .device
                .create_semaphore(&vk::SemaphoreCreateInfo { ..Default::default() }, None)
                .expect("create_semaphore failed")
        })
        .collect();

    // Register the swapchain
    inner.swapchains.push(LayerSwapchain {
        device,
        format: create_info.image_format,
        extent: create_info.image_extent,
        swapchain: *p_swapchain,
        images,
        image_views,
        render_to_present: render_to_present_semaphores,
    });
    result
}

#[no_mangle]
pub(crate) unsafe extern "system" fn layer_vkDestroySwapchainKHR(
    device: vk::Device,
    swapchain: vk::SwapchainKHR,
    p_allocator: *const vk::AllocationCallbacks,
) {
    let d = device_data(device);
    let mut inner = d.tracked_resources.lock().unwrap();

    if let Some(index) = inner.swapchains.iter().position(|sc| sc.swapchain == swapchain) {
        let sc = inner.swapchains.remove(index);
        for view in sc.image_views {
            d.dispatch.device.destroy_image_view(view, None);
        }
    }
    (d.dispatch.khr_swapchain.destroy_swapchain_khr)(device, swapchain, p_allocator);
}

#[no_mangle]
pub(crate) unsafe extern "system" fn layer_vkQueuePresentKHR(
    queue: vk::Queue,
    p_present_info: *const vk::PresentInfoKHR,
) -> vk::Result {
    let device = QUEUE_MAP.get(&queue).expect("unknown queue").device;
    let d = device_data(device);

    // extract semaphores
    let present_info = *p_present_info;
    let wait_semaphores =
        slice::from_raw_parts(present_info.p_wait_semaphores, present_info.wait_semaphore_count as usize);
    let swapchains = slice::from_raw_parts(present_info.p_swapchains, present_info.swapchain_count as usize);
    let image_indices = slice::from_raw_parts(present_info.p_image_indices, present_info.swapchain_count as usize);

    // render overlay on the first swapchain
    if present_info.swapchain_count > 0 {
        let swapchain = swapchains[0];
        let image_index = image_indices[0];
        crate::overlay::render_overlay(&*d, queue, swapchain, image_index, wait_semaphores)
    } else {
        (d.dispatch.khr_swapchain.queue_present_khr)(queue, p_present_info)
    }
}
