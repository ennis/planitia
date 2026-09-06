use crate::device::{get_preferred_present_mode, get_preferred_swap_extent};
use crate::image::ImageDescriptors;
use crate::{CommandBuffer, Device, Image, ImageType, ImageUsage, ResourceAllocation, Size3D, vk_khr_surface};
use ash::vk;
use gpu_allocator::MemoryLocation;
use log::info;
use std::ptr;
use std::time::Duration;
use vulkan::*;
use vulkan_headers::vulkan::vulkan::VkResourceDescriptorInfoEXT;

#[derive(Debug)]
pub struct SwapchainImage {
    pub image: Image,
    /// Sync between rendering & presentation.
    pub render_finished: VkSemaphore,
}

/// Represents a Vulkan swap chain.
pub struct SwapChain {
    pub handle: VkSwapchainKHR,
    pub surface: VkSurfaceKHR,
    pub format: VkSurfaceFormatKHR,
    pub width: u32,
    pub height: u32,
    pub images: Vec<SwapchainImage>,
}

/// Swap chains
impl Device {
    /// Creates a swap chain object.
    pub unsafe fn create_swapchain(
        &self,
        surface: VkSurfaceKHR,
        format: VkSurfaceFormatKHR,
        width: u32,
        height: u32,
    ) -> SwapChain {
        let mut swapchain = SwapChain { handle: Default::default(), surface, images: vec![], format, width, height };
        self.resize_swapchain(&mut swapchain, width, height);
        swapchain
    }

    pub(crate) fn register_swapchain_image(&self, handle: VkImage, format: VkFormat, width: u32, height: u32) -> Image {
        let attachment_view = unsafe { self.create_attachment_image_view(handle, format) };
        Image {
            handle,
            attachment_view,
            memory_location: MemoryLocation::Unknown,
            allocation: ResourceAllocation::External,
            swapchain_image: true,
            descriptors: ImageDescriptors::default(),
            usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSFER_DST,
            type_: ImageType::Image2D,
            format,
            mip_levels: 1,
            array_layers: 1,
            size: Size3D { width, height, depth: 1 },
            samples: 0,
        }
    }

    /// Acquires the next image in a swap chain.
    ///
    /// Returns the image and the semaphore that will be signaled when the image is available.
    pub unsafe fn acquire_next_swapchain_image<'a>(
        &self,
        swap_chain: &'a SwapChain,
        timeout: Duration,
    ) -> (usize, &'a Image) {
        // We can't use `get_or_create_semaphore` because according to the spec the semaphore
        // passed to `vkAcquireNextImage` must not have any pending operations.
        // `get_or_create_semaphore` only guarantees that a wait operation has been submitted
        // on the semaphore (not that the wait has completed).
        let ready = {
            let create_info = VkSemaphoreCreateInfo { ..Default::default() };
            self.vk.CreateSemaphore(self.vkd, &create_info, ptr::null()).unwrap()
        };
        let index = self
            .ext
            .swapchain
            .AcquireNextImageKHR(self.vkd, swap_chain.handle, timeout.as_nanos() as u64, ready, VkFence::null())
            .unwrap();

        // wait (GPU side) for the image to be ready
        crate::wait(ready, 0);
        let img = &swap_chain.images[index as usize].image;
        // transition image to GENERAL
        {
            let mut cmd = CommandBuffer::new();
            unsafe {
                cmd.image_barrier(&VkImageMemoryBarrier2 {
                    srcStageMask: 0,
                    srcAccessMask: 0,
                    dstStageMask: 0,
                    dstAccessMask: 0,
                    oldLayout: VK_IMAGE_LAYOUT_UNDEFINED,
                    newLayout: VK_IMAGE_LAYOUT_GENERAL,
                    srcQueueFamilyIndex: VK_QUEUE_FAMILY_IGNORED,
                    dstQueueFamilyIndex: VK_QUEUE_FAMILY_IGNORED,
                    image: img.handle,
                    subresourceRange: VkImageSubresourceRange {
                        aspectMask: VK_IMAGE_ASPECT_COLOR_BIT,
                        baseMipLevel: 0,
                        levelCount: 1,
                        baseArrayLayer: 0,
                        layerCount: 1,
                    },
                    ..
                });
            }
            crate::submit(cmd);
        }
        self.delete_after_current_frame(move |this| {
            this.vk.DestroySemaphore(this.vkd, ready, ptr::null());
        });
        (index as usize, img)
    }

    /// Resizes a swap chain.
    pub unsafe fn resize_swapchain(&self, swapchain: &mut SwapChain, width: u32, height: u32) {
        let instance = crate::Instance::get();
        let phy = self.thread_safe.physical_device;
        let capabilities =
            instance.khr_surface.GetPhysicalDeviceSurfaceCapabilitiesKHR(phy, swapchain.surface).unwrap();
        let present_modes = {
            let mut count = 0;
            instance
                .khr_surface
                .GetPhysicalDeviceSurfacePresentModesKHR(phy, swapchain.surface, &mut count, ptr::null_mut())
                .check();
            let mut modes = Vec::with_capacity(count as usize);
            instance
                .khr_surface
                .GetPhysicalDeviceSurfacePresentModesKHR(phy, swapchain.surface, &mut count, modes.as_mut_ptr())
                .check();
            modes.set_len(count as usize);
            modes
        };
        let present_mode = get_preferred_present_mode(&present_modes);
        let image_extent = get_preferred_swap_extent((width, height), &capabilities);
        let image_count =
            if capabilities.maxImageCount > 0 && capabilities.minImageCount + 1 > capabilities.maxImageCount {
                capabilities.maxImageCount
            } else {
                capabilities.minImageCount + 1
            };
        info!("gpu: creating or resizing swapchain ({width}×{height})");
        info!("     presentMode: {present_mode:?}");
        let create_info = VkSwapchainCreateInfoKHR {
            flags: 0,
            surface: swapchain.surface,
            minImageCount: image_count,
            imageFormat: swapchain.format.format,
            imageColorSpace: swapchain.format.colorSpace,
            imageExtent: image_extent,
            imageArrayLayers: 1,
            // TODO: this should be a parameter
            imageUsage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST,
            imageSharingMode: vk::SharingMode::EXCLUSIVE,
            queueFamilyIndexCount: 0,
            pQueueFamilyIndices: ptr::null(),
            preTransform: vk::SurfaceTransformFlagsKHR::IDENTITY,
            // TODO: this should be a parameter
            compositeAlpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
            presentMode: present_mode,
            clipped: VK_TRUE,
            oldSwapchain: swapchain.handle,
            ..
        };
        let new_handle = self.ext.swapchain.CreateSwapchainKHR(self.vkd, &create_info, ptr::null()).unwrap();
        // destroy the old swapchain if it exists
        if swapchain.handle != VkSwapchainKHR::null() {
            // FIXME the images may be in use, we should wait for the device to be idle
            self.ext.swapchain.DestroySwapchainKHR(self.vkd, swapchain.handle, ptr::null());
        }
        swapchain.handle = new_handle;
        swapchain.width = width;
        swapchain.height = height;
        // reset images & semaphores
        for SwapchainImage { render_finished, .. } in swapchain.images.drain(..) {
            self.recycle_binary_semaphore(render_finished);
        }
        swapchain.images = Vec::with_capacity(image_count as usize);
        let images = {
            let mut count = 0;
            self.ext.swapchain.GetSwapchainImagesKHR(self.vkd, swapchain.handle, &mut count, ptr::null_mut()).check();
            let mut images = Vec::with_capacity(count as usize);
            self.ext
                .swapchain
                .GetSwapchainImagesKHR(self.vkd, swapchain.handle, &mut count, images.as_mut_ptr())
                .check();
            images.set_len(count as usize);
            images
        };
        for image in images {
            let render_finished = self.get_or_create_semaphore();
            swapchain.images.push(SwapchainImage {
                image: self.register_swapchain_image(image, swapchain.format.format, width, height),
                render_finished,
            });
        }
    }
}
