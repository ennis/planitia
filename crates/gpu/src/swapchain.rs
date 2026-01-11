use crate::device::{get_preferred_present_mode, get_preferred_swap_extent};
use crate::{vk_khr_surface, CommandBuffer, Device, Image, ImageType, ImageUsage, ResourceAllocation, Size3D};
use ash::vk;
use std::ptr;
use std::time::Duration;
use gpu_allocator::MemoryLocation;

#[derive(Debug)]
struct SwapchainImageInner {
    image: Image,
    render_finished: vk::Semaphore,
}

/// Represents a swap chain.
#[derive(Debug)]
pub struct SwapChain {
    pub handle: vk::SwapchainKHR,
    pub surface: vk::SurfaceKHR,
    pub format: vk::SurfaceFormatKHR,
    pub width: u32,
    pub height: u32,
    images: Vec<SwapchainImageInner>,
}

/// Contains information about an image in a swapchain.
#[derive(Debug)]
pub struct SwapchainImage<'a> {
    /// Handle of the swapchain that owns this image.
    pub swapchain: vk::SwapchainKHR,
    /// Index of the image in the swap chain.
    pub index: u32,
    pub image: &'a Image,
    pub(crate) render_finished: vk::Semaphore,
}

/// Swap chains
impl Device {
    /// Creates a swap chain object.
    pub unsafe fn create_swapchain(
        &self,
        surface: vk::SurfaceKHR,
        format: vk::SurfaceFormatKHR,
        width: u32,
        height: u32,
    ) -> SwapChain {
        let mut swapchain = SwapChain {
            handle: Default::default(),
            surface,
            images: vec![],
            format,
            width,
            height,
        };
        self.resize_swapchain(&mut swapchain, width, height);
        swapchain
    }

    pub(crate) fn register_swapchain_image(
        &self,
        handle: vk::Image,
        format: vk::Format,
        width: u32,
        height: u32,
    ) -> Image {
        let descriptors = self.create_image_resource_descriptors(
            handle,
            ImageType::Image2D,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST,
            format,
            1,
            1,
        );
        Image {
            id: self.allocate_resource_id(),
            memory_location: MemoryLocation::Unknown,
            allocation: ResourceAllocation::External,
            handle,
            swapchain_image: true,
            descriptors,
            usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSFER_DST,
            type_: ImageType::Image2D,
            format,
            size: Size3D {
                width,
                height,
                depth: 1,
            },
        }
    }

    /// Acquires the next image in a swap chain.
    ///
    /// Returns the image and the semaphore that will be signaled when the image is available.
    pub unsafe fn acquire_next_swapchain_image<'a>(
        &self,
        swap_chain: &'a SwapChain,
        timeout: Duration,
    ) -> Result<SwapchainImage<'a>, vk::Result> {
        // We can't use `get_or_create_semaphore` because according to the spec the semaphore
        // passed to `vkAcquireNextImage` must not have any pending operations, whereas
        // `get_or_create_semaphore` only guarantees that a wait operation has been submitted
        // on the semaphore (not that the wait has completed).
        let ready = {
            let create_info = vk::SemaphoreCreateInfo { ..Default::default() };
            self.raw.create_semaphore(&create_info, None).unwrap()
        };

        let (index, _suboptimal) = match self.extensions.khr_swapchain.acquire_next_image(
            swap_chain.handle,
            timeout.as_nanos() as u64,
            ready,
            vk::Fence::null(),
        ) {
            Ok(result) => result,
            Err(err) => {
                // delete the semaphore before returning
                self.raw.destroy_semaphore(ready, None);
                return Err(err);
            }
        };

        let img = SwapchainImage {
            swapchain: swap_chain.handle,
            image: &swap_chain.images[index as usize].image,
            index,
            render_finished: swap_chain.images[index as usize].render_finished,
        };

        // wait (GPU side) for the image to be ready
        crate::sync_wait(ready, 0);

        // transition image to GENERAL
        {
            let mut cmd = CommandBuffer::new();
            unsafe {
                cmd.image_barrier(&vk::ImageMemoryBarrier2 {
                    src_stage_mask: vk::PipelineStageFlags2::NONE,
                    src_access_mask: vk::AccessFlags2::NONE,
                    dst_stage_mask: vk::PipelineStageFlags2::NONE,
                    dst_access_mask: vk::AccessFlags2::NONE,
                    old_layout: vk::ImageLayout::UNDEFINED,
                    new_layout: vk::ImageLayout::GENERAL,
                    src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    image: img.image.handle,
                    subresource_range: vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    ..Default::default()
                });
            }
            crate::submit(cmd)?;
        }

        //self.delete_after_current_submission(move |this| {
        //    this.raw.destroy_semaphore(ready, None);
        //});

        Ok(img)
    }

    /// Resizes a swap chain.
    pub unsafe fn resize_swapchain(&self, swapchain: &mut SwapChain, width: u32, height: u32) {
        let phy = self.thread_safe.physical_device;
        let capabilities = vk_khr_surface()
            .get_physical_device_surface_capabilities(phy, swapchain.surface)
            .unwrap();
        /*let formats = self
        .vk_khr_surface
        .get_physical_device_surface_formats(phy, swapchain.surface)
        .unwrap();*/
        let present_modes = vk_khr_surface()
            .get_physical_device_surface_present_modes(phy, swapchain.surface)
            .unwrap();

        let present_mode = get_preferred_present_mode(&present_modes);
        let image_extent = get_preferred_swap_extent((width, height), &capabilities);
        let image_count =
            if capabilities.max_image_count > 0 && capabilities.min_image_count + 1 > capabilities.max_image_count {
                capabilities.max_image_count
            } else {
                capabilities.min_image_count + 1
            };

        let create_info = vk::SwapchainCreateInfoKHR {
            flags: Default::default(),
            surface: swapchain.surface,
            min_image_count: image_count,
            image_format: swapchain.format.format,
            image_color_space: swapchain.format.color_space,
            image_extent,
            image_array_layers: 1,
            // TODO: this should be a parameter
            image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST,
            image_sharing_mode: vk::SharingMode::EXCLUSIVE,
            queue_family_index_count: 0,
            p_queue_family_indices: ptr::null(),
            pre_transform: vk::SurfaceTransformFlagsKHR::IDENTITY,
            // TODO: this should be a parameter
            composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
            present_mode,
            clipped: vk::TRUE,
            old_swapchain: swapchain.handle,
            ..Default::default()
        };

        let new_handle = self
            .extensions
            .khr_swapchain
            .create_swapchain(&create_info, None)
            .unwrap();
        if swapchain.handle != vk::SwapchainKHR::null() {
            // FIXME the images may be in use, we should wait for the device to be idle
            self.extensions.khr_swapchain.destroy_swapchain(swapchain.handle, None);
        }

        swapchain.handle = new_handle;
        swapchain.width = width;
        swapchain.height = height;

        // reset images & semaphores
        for SwapchainImageInner { render_finished, .. } in swapchain.images.drain(..) {
            self.recycle_binary_semaphore(render_finished);
        }
        swapchain.images = Vec::with_capacity(image_count as usize);

        let images = self
            .extensions
            .khr_swapchain
            .get_swapchain_images(swapchain.handle)
            .unwrap();
        for image in images {
            let render_finished = self.get_or_create_semaphore();
            swapchain.images.push(SwapchainImageInner {
                image: self.register_swapchain_image(image, swapchain.format.format, width, height),
                render_finished,
            });
        }
    }
}
