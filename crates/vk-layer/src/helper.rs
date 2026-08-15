//! Helper utilities.
use crate::dispatch::DeviceDispatch;
use ash::vk;
use std::ffi::{c_void, CStr};
use std::ops::Deref;
use std::ptr;

#[derive(Copy, Clone, Default)]
pub struct Image {
    pub image: vk::Image,
    pub image_view: vk::ImageView,
    pub memory: vk::DeviceMemory,
}

#[derive(Copy, Clone, Default)]
pub struct Buffer {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub ptr: *mut c_void,
}

unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

#[derive(Copy, Clone, Default)]
pub struct Pipeline {
    pub pipeline: vk::Pipeline,
    pub pipeline_layout: vk::PipelineLayout,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
}

pub struct GraphicsPipelineHelperCreateInfo<'a> {
    pub spirv: &'a [u8],
    pub vertex_entry: &'a CStr,
    pub fragment_entry: &'a CStr,
    pub vertex_attributes: &'a [vk::VertexInputAttributeDescription],
    pub vertex_stride: usize,
    pub bindings: &'a [vk::DescriptorSetLayoutBinding<'a>],
    pub push_constants_size: usize,
    pub color_attachment_format: vk::Format,
}

pub enum Descriptor {
    Texture { binding: u32, image_view: vk::ImageView, image_layout: vk::ImageLayout },
    Sampler { binding: u32, sampler: vk::Sampler },
}

pub trait PrivateData: Copy + 'static {
    type Handle: vk::Handle;
}

/// Device & command pool wrapper with useful utilities.
pub struct DeviceHelper {
    pub dispatch: DeviceDispatch,
    pub mem_props: vk::PhysicalDeviceMemoryProperties,
    pub command_pool: vk::CommandPool,
    pub queue: vk::Queue,
    pub private_data_slot: vk::PrivateDataSlot,
}

impl Deref for DeviceHelper {
    type Target = DeviceDispatch;

    fn deref(&self) -> &Self::Target {
        &self.dispatch
    }
}

impl DeviceHelper {
    pub unsafe fn new(
        dispatch: DeviceDispatch,
        mem_props: vk::PhysicalDeviceMemoryProperties,
        queue_family_index: u32,
    ) -> DeviceHelper {
        let command_pool = dispatch
            .create_command_pool(
                &vk::CommandPoolCreateInfo {
                    flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
                    queue_family_index,
                    ..Default::default()
                },
                None,
            )
            .expect("create_command_pool failed");
        let queue = dispatch.get_device_queue(queue_family_index, 0);
        dispatch.set_device_loader_data(queue);
        let private_data_slot = dispatch
            .create_private_data_slot(&vk::PrivateDataSlotCreateInfo::default(), None)
            .expect("create_private_data_slot failed");
        DeviceHelper { dispatch, mem_props, command_pool, queue, private_data_slot }
    }

    pub unsafe fn set_private_data<T: PrivateData>(&self, handle: T::Handle, data: T) {
        let data_ptr = Box::into_raw(Box::new(data)) as *mut c_void as u64;
        self.dispatch.set_private_data(handle, self.private_data_slot, data_ptr).unwrap();
    }

    pub unsafe fn get_private_data<T: PrivateData>(&self, handle: T::Handle) -> Option<&T> {
        let data_ptr = self.dispatch.get_private_data(handle, self.private_data_slot);
        if data_ptr == 0 {
            None
        } else {
            Some(&*(data_ptr as *const T))
        }
    }

    pub unsafe fn take_private_data<T: PrivateData>(&self, handle: T::Handle) -> Option<Box<T>> {
        let data_ptr = self.dispatch.get_private_data(handle, self.private_data_slot);
        if data_ptr == 0 {
            None
        } else {
            let data = Box::from_raw(data_ptr as *mut T);
            Some(data)
        }
    }

    pub fn find_memory_type(&self, type_filter: u32, required_flags: vk::MemoryPropertyFlags) -> u32 {
        (0..self.mem_props.memory_type_count)
            .find(|&i| {
                (type_filter & (1 << i)) != 0
                    && self.mem_props.memory_types[i as usize].property_flags.contains(required_flags)
            })
            .expect("no compatible memory type found")
    }

    pub unsafe fn allocate_command_buffers_helper(&self, count: usize) -> Vec<vk::CommandBuffer> {
        let buffers = self
            .allocate_command_buffers(&vk::CommandBufferAllocateInfo {
                command_pool: self.command_pool,
                level: vk::CommandBufferLevel::PRIMARY,
                command_buffer_count: count as u32,
                ..Default::default()
            })
            .unwrap();
        for b in buffers.iter() {
            self.set_device_loader_data(*b);
        }
        buffers
    }

    pub unsafe fn wait_for_fence_and_reset(&self, fence: vk::Fence) {
        self.wait_for_fences(&[fence], true, u64::MAX).unwrap();
        self.reset_fences(&[fence]).unwrap();
    }

    pub unsafe fn reset_and_begin_command_buffer(&self, cmdbuf: vk::CommandBuffer) {
        self.reset_command_buffer(cmdbuf, vk::CommandBufferResetFlags::empty()).unwrap();
        self.begin_command_buffer(
            cmdbuf,
            &vk::CommandBufferBeginInfo { flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT, ..Default::default() },
        )
        .unwrap();
    }

    pub unsafe fn cmd_push_descriptors_helper(
        &self,
        cmdbuf: vk::CommandBuffer,
        pipeline_layout: vk::PipelineLayout,
        descriptors: &[Descriptor],
    ) {
        union DescriptorInfo {
            image: vk::DescriptorImageInfo,
            buffer: vk::DescriptorBufferInfo,
        }

        let mut descriptor_infos = Vec::with_capacity(descriptors.len());
        let mut write_descriptors = Vec::with_capacity(descriptors.len());

        for descriptor in descriptors {
            match descriptor {
                Descriptor::Texture { binding, image_view, image_layout } => {
                    descriptor_infos.push(DescriptorInfo {
                        image: vk::DescriptorImageInfo {
                            sampler: vk::Sampler::null(),
                            image_view: *image_view,
                            image_layout: *image_layout,
                        },
                    });
                    write_descriptors.push(vk::WriteDescriptorSet {
                        dst_binding: *binding,
                        descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                        descriptor_count: 1,
                        p_image_info: &descriptor_infos.last().unwrap().image,
                        ..Default::default()
                    });
                }
                Descriptor::Sampler { binding, sampler } => {
                    descriptor_infos.push(DescriptorInfo {
                        image: vk::DescriptorImageInfo {
                            sampler: *sampler,
                            image_view: vk::ImageView::null(),
                            image_layout: vk::ImageLayout::UNDEFINED,
                        },
                    });
                    write_descriptors.push(vk::WriteDescriptorSet {
                        dst_binding: *binding,
                        descriptor_type: vk::DescriptorType::SAMPLER,
                        descriptor_count: 1,
                        p_image_info: &descriptor_infos.last().unwrap().image,
                        ..Default::default()
                    });
                }
            }
        }

        (self.khr_push_descriptors.cmd_push_descriptor_set_khr)(
            cmdbuf,
            // We don't have compute shaders for the moment
            vk::PipelineBindPoint::GRAPHICS,
            pipeline_layout,
            // and we won't need more than one descriptor set
            0,
            write_descriptors.len() as u32,
            write_descriptors.as_ptr(),
        );
    }

    pub(crate) unsafe fn cmd_set_viewport_helper(&self, cmdbuf: vk::CommandBuffer, x: i32, y: i32, w: i32, h: i32) {
        self.cmd_set_viewport(
            cmdbuf,
            0,
            &[vk::Viewport {
                x: x as f32,
                y: y as f32,
                width: w as f32,
                height: h as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            }],
        );
    }

    pub(crate) unsafe fn cmd_set_scissor_helper(&self, cmdbuf: vk::CommandBuffer, x: i32, y: i32, w: i32, h: i32) {
        self.cmd_set_scissor(
            cmdbuf,
            0,
            &[vk::Rect2D { offset: vk::Offset2D { x, y }, extent: vk::Extent2D { width: w as u32, height: h as u32 } }],
        );
    }

    pub(crate) unsafe fn layout_barrier(
        &self,
        cmdbuf: vk::CommandBuffer,
        transitions: &[(vk::Image, vk::ImageLayout, vk::ImageLayout)],
    ) {
        // We are heavy-handed on the pipeline stages & access flags,
        // as this is not remotely worth the trouble.
        let barriers = transitions
            .iter()
            .map(|&(image, old_layout, new_layout)| vk::ImageMemoryBarrier {
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
            })
            .collect::<Vec<_>>();

        self.cmd_pipeline_barrier(
            cmdbuf,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &barriers,
        );
    }

    pub(crate) unsafe fn create_graphics_pipeline_helper(
        &self,
        create_info: &GraphicsPipelineHelperCreateInfo,
    ) -> Pipeline {
        assert_eq!(create_info.spirv.len() % 4, 0, "SPIR-V size must be a multiple of 4");
        let spv_words: Vec<u32> =
            create_info.spirv.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();

        let shader_module = self
            .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&spv_words), None)
            .expect("failed to create shader module");

        let push_constant_range = vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            offset: 0,
            size: create_info.push_constants_size as u32,
        };

        let descriptor_set_layout = self
            .create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo {
                    flags: vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR,
                    binding_count: create_info.bindings.len() as u32,
                    p_bindings: create_info.bindings.as_ptr(),
                    ..Default::default()
                },
                None,
            )
            .unwrap();

        let pipeline_layout = self
            .create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo {
                    set_layout_count: 1,
                    p_set_layouts: &descriptor_set_layout,
                    push_constant_range_count: 1,
                    p_push_constant_ranges: &push_constant_range,
                    ..Default::default()
                },
                None,
            )
            .unwrap();
        let shader_stages = [
            vk::PipelineShaderStageCreateInfo {
                stage: vk::ShaderStageFlags::VERTEX,
                module: shader_module,
                p_name: create_info.vertex_entry.as_ptr(),
                ..Default::default()
            },
            vk::PipelineShaderStageCreateInfo {
                stage: vk::ShaderStageFlags::FRAGMENT,
                module: shader_module,
                p_name: create_info.fragment_entry.as_ptr(),
                ..Default::default()
            },
        ];

        let vertex_binding = vk::VertexInputBindingDescription {
            binding: 0,
            stride: create_info.vertex_stride as u32,
            input_rate: vk::VertexInputRate::VERTEX,
        };
        let vertex_input_state = vk::PipelineVertexInputStateCreateInfo {
            vertex_binding_description_count: 1,
            p_vertex_binding_descriptions: &vertex_binding,
            vertex_attribute_description_count: create_info.vertex_attributes.len() as u32,
            p_vertex_attribute_descriptions: create_info.vertex_attributes.as_ptr(),
            ..Default::default()
        };
        let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo {
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            ..Default::default()
        };
        let viewport_state =
            vk::PipelineViewportStateCreateInfo { viewport_count: 1, scissor_count: 1, ..Default::default() };

        let rasterization_state = vk::PipelineRasterizationStateCreateInfo {
            polygon_mode: vk::PolygonMode::FILL,
            cull_mode: vk::CullModeFlags::NONE,
            front_face: vk::FrontFace::COUNTER_CLOCKWISE,
            line_width: 1.0,
            ..Default::default()
        };

        let multisample_state = vk::PipelineMultisampleStateCreateInfo {
            rasterization_samples: vk::SampleCountFlags::TYPE_1,
            ..Default::default()
        };

        // Standard "source-over" alpha compositing for the overlay.
        let blend_attachment = vk::PipelineColorBlendAttachmentState {
            blend_enable: vk::TRUE,
            src_color_blend_factor: vk::BlendFactor::SRC_ALPHA,
            dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
            color_blend_op: vk::BlendOp::ADD,
            src_alpha_blend_factor: vk::BlendFactor::ONE,
            dst_alpha_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
            alpha_blend_op: vk::BlendOp::ADD,
            color_write_mask: vk::ColorComponentFlags::RGBA,
        };
        let color_blend_state = vk::PipelineColorBlendStateCreateInfo {
            attachment_count: 1,
            p_attachments: &blend_attachment,
            ..Default::default()
        };
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state = vk::PipelineDynamicStateCreateInfo {
            dynamic_state_count: dynamic_states.len() as u32,
            p_dynamic_states: dynamic_states.as_ptr(),
            ..Default::default()
        };
        let mut rendering_info = vk::PipelineRenderingCreateInfo {
            color_attachment_count: 1,
            p_color_attachment_formats: &create_info.color_attachment_format,
            ..Default::default()
        };

        let pipeline_create_info = vk::GraphicsPipelineCreateInfo {
            p_next: &mut rendering_info as *const _ as *const c_void,
            p_stages: shader_stages.as_ptr(),
            stage_count: shader_stages.len() as u32,
            p_vertex_input_state: &vertex_input_state,
            p_input_assembly_state: &input_assembly_state,
            p_viewport_state: &viewport_state,
            p_rasterization_state: &rasterization_state,
            p_multisample_state: &multisample_state,
            p_color_blend_state: &color_blend_state,
            p_dynamic_state: &dynamic_state,
            layout: pipeline_layout,
            render_pass: vk::RenderPass::null(),
            ..Default::default()
        };

        let pipeline = self
            .create_graphics_pipelines(vk::PipelineCache::null(), std::slice::from_ref(&pipeline_create_info), None)
            .expect("failed to create graphics pipeline")[0];

        // The shader module is no longer needed once the pipeline is built.
        self.destroy_shader_module(shader_module, None);

        Pipeline { pipeline, pipeline_layout, descriptor_set_layout }
    }

    pub(crate) unsafe fn create_color_image_helper(
        &self,
        format: vk::Format,
        width: u32,
        height: u32,
        usage: vk::ImageUsageFlags,
    ) -> Image {
        let d = &self.dispatch.device;

        let create_info = vk::ImageCreateInfo {
            image_type: vk::ImageType::TYPE_2D,
            format,
            extent: vk::Extent3D { width, height, depth: 1 },
            mip_levels: 1,
            array_layers: 1,
            samples: vk::SampleCountFlags::TYPE_1,
            tiling: vk::ImageTiling::OPTIMAL,
            usage,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            ..Default::default()
        };

        let image = d.create_image(&create_info, None).unwrap();
        let img_req = d.get_image_memory_requirements(image);
        let img_mem_type = self.find_memory_type(img_req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL);
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

    pub(crate) unsafe fn create_buffer_from_data<T: Copy + 'static>(
        &self,
        usage: vk::BufferUsageFlags,
        data: &[T],
    ) -> Buffer {
        let data_bytes = std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * std::mem::size_of::<T>());
        self.create_buffer_helper(usage, data_bytes.len(), Some(data_bytes))
    }

    pub(crate) unsafe fn create_buffer_helper(
        &self,
        usage: vk::BufferUsageFlags,
        byte_size: usize,
        initial_data: Option<&[u8]>,
    ) -> Buffer {
        let create_info = vk::BufferCreateInfo {
            size: byte_size as u64,
            // We may not need TRANSFER_DST if there's no initial data,
            // but adding it most likely doesn't have any perf impact whatsoever
            usage: usage | vk::BufferUsageFlags::TRANSFER_DST,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };
        let required_flags = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let buffer = self.create_buffer(&create_info, None).unwrap();
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
