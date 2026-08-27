//! Swapchain interception
use crate::{Device, SwapchainInfo};
use ash::vk;
use ash::vk::Handle;
use std::{ptr, slice};
use crate::overlay::renderer::render_overlay;
use crate::surface::get_hwnd_for_surface;

impl Device {
    pub unsafe fn hook_create_swapchain_khr(
        &self,
        device: vk::Device,
        p_create_info: *const vk::SwapchainCreateInfoKHR,
        p_allocator: *const vk::AllocationCallbacks,
        p_swapchain: *mut vk::SwapchainKHR,
    ) -> vk::Result {
        let mut inner = self.tracked_resources.lock();

        let mut create_info = *p_create_info;

        // The pan/zoom shader needs TRANSFER_SRC usage.
        create_info.image_usage |= vk::ImageUsageFlags::TRANSFER_SRC;

        // If there's an old swapchain to be deleted, delete the resources that we have for it
        if !create_info.old_swapchain.is_null() {
            if let Some(index) = inner.swapchains.iter().position(|sc| sc.swapchain == create_info.old_swapchain) {
                let sc = inner.swapchains.remove(index);
                for view in sc.image_views {
                    self.destroy_image_view(view, None);
                }
                // FIXME: we have no way to know whether the semaphore are still being waited on.
                //        Technically waitForIdle isn't sufficient as it doesn't sync with presentation.
                //        So it's basically impossible to do this correctly.
                //        See https://stackoverflow.com/questions/75437792/how-to-synchronize-vulkan-swapchain-presentation-with-sempahore-destruction
                //        As a best effort, wait for device idle first.
                self.device_wait_idle().unwrap();
                for sem in sc.render_to_present {
                    self.destroy_semaphore(sem, None);
                }
            }
        }

        // Call next layer
        let result = (self.khr_swapchain.create_swapchain_khr)(device, &create_info, p_allocator, p_swapchain);
        if result != vk::Result::SUCCESS {
            return result;
        }

        // Retrieve the backing images and create image views for them.
        let mut image_count = 0;
        let r = (self.khr_swapchain.get_swapchain_images_khr)(device, *p_swapchain, &mut image_count, ptr::null_mut());

        assert_eq!(r, vk::Result::SUCCESS);
        let images = {
            let mut images = Vec::with_capacity(image_count as usize);
            let r = (self.khr_swapchain.get_swapchain_images_khr)(device, *p_swapchain, &mut image_count, images.as_mut_ptr());
            assert_eq!(r, vk::Result::SUCCESS);
            images.set_len(image_count as usize);
            images
        };

        let image_views = images
            .iter()
            .map(|&image| {
                self.create_image_view(
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
            .collect::<Vec<_>>();

        let render_to_present_semaphores = (0..images.len())
            .map(|_| {
                self.create_semaphore(&vk::SemaphoreCreateInfo { ..Default::default() }, None)
                    .expect("create_semaphore failed")
            })
            .collect();

        let surface = (*p_create_info).surface;

        // Register the swapchain
        inner.swapchains.push(SwapchainInfo {
            surface,
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

    pub unsafe fn hook_destroy_swapchain_khr(
        &self,
        device: vk::Device,
        swapchain: vk::SwapchainKHR,
        p_allocator: *const vk::AllocationCallbacks,
    ) {
        let mut inner = self.tracked_resources.lock();

        if let Some(index) = inner.swapchains.iter().position(|sc| sc.swapchain == swapchain) {
            let sc = inner.swapchains.remove(index);
            for view in sc.image_views {
                self.destroy_image_view(view, None);
            }
            for sem in sc.render_to_present {
                self.destroy_semaphore(sem, None);
            }
        }
        (self.khr_swapchain.destroy_swapchain_khr)(device, swapchain, p_allocator);
    }

    pub unsafe fn hook_queue_present_khr(&self, queue: vk::Queue, p_present_info: *const vk::PresentInfoKHR) -> vk::Result {
        // wait for our debugger probes to finish executing
        // and for the rest as well, incidentally...
        self.device_wait_idle().unwrap();


        let present_info = *p_present_info;
        let wait_semaphores =
            slice::from_raw_parts(present_info.p_wait_semaphores, present_info.wait_semaphore_count as usize);
        let swapchains = slice::from_raw_parts(present_info.p_swapchains, present_info.swapchain_count as usize);
        let image_indices = slice::from_raw_parts(present_info.p_image_indices, present_info.swapchain_count as usize);

        let result = if present_info.swapchain_count == 1 {
            // render our overlay on the first swapchain
            // TODO: support multiple swapchains in vkQueuePresent
            let swapchain = swapchains[0];
            let image_index = image_indices[0];

            // update inputs for the surface (and HWND) associated to the swapchain
            let surface = self.tracked_resources.lock().swapchains.iter().find(|sc| sc.swapchain == swapchain).map(|sc| sc.surface);
            if let Some(surface) = surface {
                self.update_inputs_for_surface(surface);
            }

            render_overlay(self, queue, swapchain, image_index, wait_semaphores)
        } else {
            (self.khr_swapchain.queue_present_khr)(queue, p_present_info)
        };

        self.end_frame();
        result
    }
}
