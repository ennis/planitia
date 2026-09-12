//! Swapchain interception
use crate::helper::vkcall;
use crate::overlay::renderer::render_overlay;
use crate::surface::get_hwnd_for_surface;
use crate::{Device, SwapchainInfo};
use std::{ptr, slice};
use vulkan::*;

impl Device {
    pub unsafe fn hook_create_swapchain_khr(
        &self,
        device: VkDevice,
        p_create_info: *const VkSwapchainCreateInfoKHR,
        p_allocator: *const VkAllocationCallbacks,
        p_swapchain: *mut VkSwapchainKHR,
    ) -> VkResult {
        let mut inner = self.tracked_objects.lock();
        let mut create_info = *p_create_info;
        // The pan/zoom shader needs TRANSFER_SRC usage.
        create_info.imageUsage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        // If there's an old swapchain to be deleted, delete the resources that we have for it
        if !create_info.oldSwapchain.is_null() {
            if let Some(index) = inner.swapchains.iter().position(|sc| sc.swapchain == create_info.oldSwapchain) {
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
        let result = self.CreateSwapchainKHR(device, &create_info, p_allocator, p_swapchain);
        if result != VK_SUCCESS {
            return result;
        }
        // Retrieve the backing images and create image views for them.
        let mut image_count = 0;
        let r = self.GetSwapchainImagesKHR(device, *p_swapchain, &mut image_count, ptr::null_mut());
        assert_eq!(r, VK_SUCCESS);
        let images = {
            let mut images = Vec::with_capacity(image_count as usize);
            let r = self.GetSwapchainImagesKHR(device, *p_swapchain, &mut image_count, images.as_mut_ptr());
            assert_eq!(r, VK_SUCCESS);
            images.set_len(image_count as usize);
            images
        };
        let image_views = images
            .iter()
            .map(|&image| {
                self.create_image_view(
                    &VkImageViewCreateInfo {
                        image,
                        viewType: VK_IMAGE_VIEW_TYPE_2D,
                        format: create_info.imageFormat,
                        subresourceRange: VkImageSubresourceRange {
                            aspectMask: VK_IMAGE_ASPECT_COLOR_BIT,
                            baseMipLevel: 0,
                            levelCount: 1,
                            baseArrayLayer: 0,
                            layerCount: 1,
                        },
                        ..
                    },
                    None,
                )
                .expect("create_image_view failed")
            })
            .collect::<Vec<_>>();
        let render_to_present_semaphores = (0..images.len())
            .map(|_| {
                vkcall!(self.CreateSemaphore(&VkSemaphoreCreateInfo { .. }, ptr::null(), @out let semaphore));
                semaphore
            })
            .collect();
        let surface = (*p_create_info).surface;
        // Register the swapchain
        inner.swapchains.push(SwapchainInfo {
            surface,
            device,
            format: create_info.imageFormat,
            extent: create_info.imageExtent,
            swapchain: *p_swapchain,
            images,
            image_views,
            render_to_present: render_to_present_semaphores,
        });
        result
    }

    pub unsafe fn hook_destroy_swapchain_khr(
        &self,
        device: VkDevice,
        swapchain: VkSwapchainKHR,
        p_allocator: *const VkAllocationCallbacks,
    ) {
        let mut inner = self.tracked_objects.lock();
        if let Some(index) = inner.swapchains.iter().position(|sc| sc.swapchain == swapchain) {
            let sc = inner.swapchains.remove(index);
            for view in sc.image_views {
                self.destroy_image_view(view, None);
            }
            for sem in sc.render_to_present {
                self.destroy_semaphore(sem, None);
            }
        }
        self.DestroySwapchainKHR(device, swapchain, p_allocator);
    }

    pub unsafe fn hook_queue_present_khr(&self, queue: VkQueue, p_present_info: *const VkPresentInfoKHR) -> VkResult {
        // wait for our debugger probes to finish executing
        // and for the rest as well, incidentally...
        self.device_wait_idle().unwrap();
        let present_info = *p_present_info;
        let wait_semaphores =
            slice::from_raw_parts(present_info.pWaitSemaphores, present_info.waitSemaphoreCount as usize);
        let swapchains = slice::from_raw_parts(present_info.pSwapchains, present_info.swapchainCount as usize);
        let image_indices = slice::from_raw_parts(present_info.pImageIndices, present_info.swapchainCount as usize);
        let result = if present_info.swapchainCount == 1 {
            // render our overlay on the first swapchain
            // TODO: support multiple swapchains in vkQueuePresent
            let swapchain = swapchains[0];
            let image_index = image_indices[0];
            // update inputs for the surface (and HWND) associated to the swapchain
            let surface =
                self.tracked_objects.lock().swapchains.iter().find(|sc| sc.swapchain == swapchain).map(|sc| sc.surface);
            if let Some(surface) = surface {
                self.update_inputs_for_surface(surface);
            }
            render_overlay(self, queue, swapchain, image_index, wait_semaphores)
        } else {
            self.QueuePresentKHR(queue, p_present_info)
        };
        self.end_frame();
        result
    }
}
