use crate::{Buffer, DeviceHelper, Image};
use ash::vk;
use std::ptr;

pub(crate) unsafe fn transition_image_layout(
    d: &ash::Device,
    cmdbuf: vk::CommandBuffer,
    image: vk::Image,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
) {
    // We are heavy-handed on the pipeline stages & access flags, as this is not worth the trouble.
    let barrier = vk::ImageMemoryBarrier {
        old_layout,
        new_layout,
        src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
        dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
        image,
        subresource_range: vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: vk::REMAINING_MIP_LEVELS,
            base_array_layer: 0,
            layer_count: vk::REMAINING_ARRAY_LAYERS,
        },
        src_access_mask: vk::AccessFlags::MEMORY_WRITE,
        dst_access_mask: vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE,
        ..Default::default()
    };

    d.cmd_pipeline_barrier(
        cmdbuf,
        vk::PipelineStageFlags::ALL_COMMANDS,
        vk::PipelineStageFlags::ALL_COMMANDS,
        vk::DependencyFlags::empty(),
        &[],
        &[],
        &[barrier],
    );
}

impl DeviceHelper {
    pub(crate) fn find_memory_type(&self, type_filter: u32, required_flags: vk::MemoryPropertyFlags) -> u32 {
        (0..self.mem_props.memory_type_count)
            .find(|&i| {
                (type_filter & (1 << i)) != 0
                    && self.mem_props.memory_types[i as usize].property_flags.contains(required_flags)
            })
            .expect("no compatible memory type found")
    }

    pub(crate) unsafe fn create_image_helper(
        &self,
        create_info: &vk::ImageCreateInfo,
        required_flags: vk::MemoryPropertyFlags,
    ) -> Image {
        let d = &self.dispatch.device;
        let image = d.create_image(create_info, None).unwrap();
        let img_req = d.get_image_memory_requirements(image);
        let img_mem_type = self.find_memory_type(img_req.memory_type_bits, required_flags);
        let image_memory = d
            .allocate_memory(
                &vk::MemoryAllocateInfo {
                    allocation_size: img_req.size,
                    memory_type_index: img_mem_type,
                    ..Default::default()
                },
                None,
            )
            .unwrap();
        d.bind_image_memory(image, image_memory, 0).unwrap();
        let image_view = d
            .create_image_view(
                &vk::ImageViewCreateInfo {
                    image,
                    view_type: vk::ImageViewType::TYPE_2D,
                    format: create_info.format,
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
            .unwrap();
        Image { image, image_view, memory: image_memory }
    }

    pub(crate) unsafe fn destroy_image_helper(&self, image: Image) {
        self.destroy_image_view(image.image_view, None);
        self.destroy_image(image.image, None);
        self.free_memory(image.memory, None);
    }

    pub(crate) unsafe fn create_buffer_helper(
        &self,
        create_info: &vk::BufferCreateInfo,
        required_flags: vk::MemoryPropertyFlags,
        initial_data: Option<&[u8]>,
    ) -> Buffer {
        let buffer = self.create_buffer(create_info, None).unwrap();
        let buf_req = self.get_buffer_memory_requirements(buffer);
        let buf_mem_type = self.find_memory_type(buf_req.memory_type_bits, required_flags);
        let buffer_memory = self
            .allocate_memory(
                &vk::MemoryAllocateInfo {
                    allocation_size: buf_req.size,
                    memory_type_index: buf_mem_type,
                    ..Default::default()
                },
                None,
            )
            .unwrap();
        self.bind_buffer_memory(buffer, buffer_memory, 0).unwrap();
        let ptr = self.map_memory(buffer_memory, 0, buf_req.size, vk::MemoryMapFlags::empty()).unwrap();
        if let Some(initial_data) = initial_data {
            ptr::copy_nonoverlapping(initial_data.as_ptr(), ptr.cast::<u8>(), initial_data.len());
        }
        Buffer { buffer, memory: buffer_memory, ptr }
    }

    pub(crate) unsafe fn destroy_buffer_helper(&self, buffer: Buffer) {
        self.destroy_buffer(buffer.buffer, None);
        self.free_memory(buffer.memory, None);
    }

    pub(crate) unsafe fn set_device_loader_data(&self, handle: impl vk::Handle) {
        let _ = (self.set_device_loader_data)(self.device.handle(), handle.as_raw() as *mut _);
    }

    pub(crate) unsafe fn submit_oneshot(&self, record_fn: impl FnOnce(&Self, vk::CommandBuffer)) {
        let cmdbuf = self
            .allocate_command_buffers(&vk::CommandBufferAllocateInfo {
                command_pool: self.command_pool,
                level: vk::CommandBufferLevel::PRIMARY,
                command_buffer_count: 1,
                ..Default::default()
            })
            .unwrap()[0];
        self.set_device_loader_data(cmdbuf);
        self.begin_command_buffer(
            cmdbuf,
            &vk::CommandBufferBeginInfo { flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT, ..Default::default() },
        )
        .unwrap();
        record_fn(self, cmdbuf);
        self.end_command_buffer(cmdbuf).unwrap();
        self.queue_submit(
            self.queue,
            &[vk::SubmitInfo { command_buffer_count: 1, p_command_buffers: &cmdbuf, ..Default::default() }],
            vk::Fence::null(),
        )
        .unwrap();
    }
}
