//! Helper utilities.
use crate::dispatch::DeviceDispatch;
use std::ffi::{CStr, c_void};
use std::ops::Deref;
use std::ptr::NonNull;
use std::{mem, ptr};
use vulkan::*;

// Implementation detail of shader_module
#[doc(hidden)]
macro_rules! include_bytes_as_u32 {
    // https://docs.rs/resb/latest/src/resb/binary.rs.html#25-44
    ($path:literal) => {
        const {
            #[repr(align(4))]
            pub struct AlignedAs<Bytes: ?Sized> {
                pub bytes: Bytes,
            }

            const B: &[u8] = &AlignedAs { bytes: *include_bytes!($path) }.bytes;
            // SAFETY: B is statically borrowed, 4-aligned, and the length is within
            // the static slice (truncated to a multiple of four).
            unsafe { core::slice::from_raw_parts(B.as_ptr() as *const u32, B.len() / size_of::<u32>()) }
        }
    };
}
pub(crate) use include_bytes_as_u32;

/// Convenience macro to call vulkan functions that return arrays via a count/output pointer pair.
macro_rules! vkarraycall {
    ($p:ident$(.$ps:ident)* ($($args:expr,)* @count let $count:ident, @out let $array:ident)) => {
        vkarraycall!(let _ = $p$(.$ps)* ($($args,)* @count let $count, @out let $array));
    };
    (let $result:pat = $p:ident$(.$ps:ident)* ($($args:expr,)* @count let $count:ident, @out let $array:ident)) => {
        let mut $count = 0;
        let mut $array = vec![];
        let __result = $p$(.$ps)*($($args,)* &mut $count, ptr::null_mut());
        if __result.0 < 0 {
            $crate::panic_vulkan_api_call_failed(__result);
        }
        $array.reserve($count as usize);
        let __result = $p$(.$ps)*($($args,)* &mut $count, $array.as_mut_ptr());
        if __result.0 < 0 {
            $crate::panic_vulkan_api_call_failed(__result);
        }
        let $result = __result;
        unsafe { $array.set_len($count as usize); }
    };
}

/// Same as [`vkarraycall`] but without result checks.
macro_rules! vkarraycallnc {
    ($p:ident$(.$ps:ident)* ($($args:expr,)* @count let $count:ident, @out let $array:ident)) => {
        let mut $count = 0;
        let mut $array = vec![];
        $p$(.$ps)*($($args,)* &mut $count, ptr::null_mut());
        $array.reserve($count as usize);
        $p$(.$ps)*($($args,)* &mut $count, $array.as_mut_ptr());
        unsafe { $array.set_len($count as usize); }
    };
}

/// Convenience macro to call vulkan functions that return results via output pointer parameters.
macro_rules! vkcallnc {
    ($p:ident$(.$ps:ident)* ($($args:expr,)* $(@out let $out:ident),*)) => {
        $(let mut $out = ::core::mem::MaybeUninit::uninit();)*
        let _ = $p$(.$ps)*($($args,)* $($out.as_mut_ptr()),*);
        $(let $out = unsafe { $out.assume_init() };)*
    };
}

/// Same as [`vkcallnc`] but panics on an unsuccessful result, and puts the VkResult in a variable.
macro_rules! vkcall {
    ($p:ident$(.$ps:ident)* ($($args:expr,)* $(@out let $out:ident),*)) => {
        $(let mut $out = ::core::mem::MaybeUninit::uninit();)*
        let __result = $p$(.$ps)*($($args,)*, $($out.as_mut_ptr()),*);
        if __result.0 < 0 {
            $crate::panic_vulkan_api_call_failed(__result);
        }
        $(let $out = unsafe { $out.assume_init() };)*
    };
    (let $result:pat = $p:ident$(.$ps:ident)* ($($args:expr,)* $(@out let $out:ident),*)) => {
        $(let mut $out = ::core::mem::MaybeUninit::uninit();)*
        let __result = $p$(.$ps)*($($args,)* $($out.as_mut_ptr()),*);
        if __result.0 < 0 {
            $crate::panic_vulkan_api_call_failed(__result);
        }
        let $result = __result;
        $(let $out = unsafe { $out.assume_init() };)*
    };
}
pub(crate) use vkarraycall;
pub(crate) use vkarraycallnc;
pub(crate) use vkcall;
pub(crate) use vkcallnc;

#[derive(Copy, Clone, Default)]
pub struct Image {
    pub image: VkImage,
    pub image_view: VkImageView,
    pub memory: VkDeviceMemory,
}

#[derive(Default)]
pub struct Buffer {
    pub buffer: VkBuffer,
    pub memory: VkDeviceMemory,
    pub ptr: *mut c_void,
    pub size: usize,
    pub device_address: VkDeviceAddress,
}

unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

#[derive(Copy, Clone, Default)]
pub struct Pipeline {
    pub pipeline: VkPipeline,
    pub pipeline_layout: VkPipelineLayout,
    pub descriptor_set_layout: VkDescriptorSetLayout,
}

pub struct GraphicsPipelineHelperCreateInfo<'a> {
    pub spirv: &'a [u32],
    pub vertex_entry: &'a CStr,
    pub fragment_entry: &'a CStr,
    pub vertex_attributes: &'a [VkVertexInputAttributeDescription],
    pub vertex_stride: usize,
    pub bindings: &'a [VkDescriptorSetLayoutBinding],
    pub push_constants_size: usize,
    pub color_attachment_format: VkFormat,
}

pub enum Descriptor {
    Texture { binding: u32, image_view: VkImageView, image_layout: VkImageLayout },
    Sampler { binding: u32, sampler: VkSampler },
}

pub trait HasPrivateData: VulkanHandle + Copy {
    type PrivateData;
}

/// Device & command pool wrapper with useful utilities.
pub struct DeviceHelper {
    pub dispatch: DeviceDispatch,
    pub mem_props: VkPhysicalDeviceMemoryProperties,
    pub command_pool: VkCommandPool,
    pub queue: VkQueue,
    pub private_data_slot: VkPrivateDataSlot,
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
        mem_props: VkPhysicalDeviceMemoryProperties,
        queue_family_index: u32,
    ) -> DeviceHelper {
        let device = dispatch.device;
        let command_pool_create_info = VkCommandPoolCreateInfo {
            flags: VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            queueFamilyIndex: queue_family_index,
            ..
        };
        vkcall!(dispatch.CreateCommandPool(device, &command_pool_create_info, ptr::null(), @out let command_pool));
        vkcallnc!(dispatch.GetDeviceQueue(device, queue_family_index, 0, @out let queue));
        dispatch.set_device_loader_data(queue);
        vkcall!(dispatch.CreatePrivateDataSlot(device, &VkPrivateDataSlotCreateInfo { .. }, ptr::null(), @out let private_data_slot));
        DeviceHelper { dispatch, mem_props, command_pool, queue, private_data_slot }
    }

    pub unsafe fn set_private_data<H: HasPrivateData>(&self, handle: H, data: H::PrivateData) -> *mut H::PrivateData {
        let data_ptr = Box::into_raw(Box::new(data)) as *mut c_void as u64;
        self.dispatch.SetPrivateData(self.device, H::TYPE, handle.as_raw(), self.private_data_slot, data_ptr).check();
        data_ptr as *mut H::PrivateData
    }

    pub unsafe fn get_private_data<H: HasPrivateData>(&self, handle: H) -> Option<NonNull<H::PrivateData>> {
        vkcallnc!(self.dispatch.GetPrivateData(self.device, H::TYPE, handle.as_raw(), self.private_data_slot, @out let data_ptr));
        if data_ptr == 0 { None } else { Some(NonNull::new_unchecked(data_ptr as *mut H::PrivateData)) }
    }

    pub unsafe fn get_private_data_ref<'a, H: HasPrivateData>(&self, handle: H) -> Option<&'a H::PrivateData> {
        self.get_private_data(handle).map(|p| p.as_ref())
    }

    pub unsafe fn get_private_data_mut<'a, H: HasPrivateData>(&self, handle: H) -> Option<&'a mut H::PrivateData> {
        self.get_private_data(handle).map(|mut p| p.as_mut())
    }

    pub unsafe fn take_private_data<H: HasPrivateData>(&self, handle: H) -> Option<Box<H::PrivateData>> {
        vkcallnc!(self.dispatch.GetPrivateData(self.device, H::TYPE, handle.as_raw(), self.private_data_slot, @out let data_ptr));
        if data_ptr == 0 {
            None
        } else {
            let data = Box::from_raw(data_ptr as *mut H::PrivateData);
            Some(data)
        }
    }

    pub fn find_memory_type(&self, type_filter: u32, required_flags: VkMemoryPropertyFlags) -> u32 {
        (0..self.mem_props.memoryTypeCount)
            .find(|&i| {
                (type_filter & (1 << i)) != 0
                    && (self.mem_props.memoryTypes[i as usize].propertyFlags & required_flags == required_flags)
            })
            .expect("no compatible memory type found")
    }

    pub unsafe fn allocate_command_buffers_helper(&self, count: usize) -> Vec<VkCommandBuffer> {
        let allocate_info = VkCommandBufferAllocateInfo {
            commandPool: self.command_pool,
            level: VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount: count as u32,
            ..Default::default()
        };
        let mut buffers = Vec::with_capacity(count);
        self.AllocateCommandBuffers(self.device, &allocate_info, buffers.as_mut_ptr()).check();
        buffers.set_len(count);
        for b in buffers.iter() {
            self.set_device_loader_data(*b);
        }
        buffers
    }

    pub unsafe fn wait_for_fence_and_reset(&self, fence: VkFence) {
        self.WaitForFences(self.device, 1, &fence, VK_TRUE, u64::MAX).check();
        self.ResetFences(self.device, 1, &fence).check();
    }

    pub unsafe fn reset_and_begin_command_buffer(&self, cmdbuf: VkCommandBuffer) {
        self.ResetCommandBuffer(cmdbuf, 0).check();
        self.BeginCommandBuffer(
            cmdbuf,
            &VkCommandBufferBeginInfo { flags: VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, ..Default::default() },
        )
        .check();
    }

    pub unsafe fn cmd_push_descriptors_helper(
        &self,
        cmdbuf: VkCommandBuffer,
        pipeline_layout: VkPipelineLayout,
        descriptors: &[Descriptor],
    ) {
        union DescriptorInfo {
            image: VkDescriptorImageInfo,
            buffer: VkDescriptorBufferInfo,
        }
        let mut descriptor_infos = Vec::with_capacity(descriptors.len());
        let mut write_descriptors = Vec::with_capacity(descriptors.len());
        for descriptor in descriptors {
            match descriptor {
                Descriptor::Texture { binding, image_view, image_layout } => {
                    descriptor_infos.push(DescriptorInfo {
                        image: VkDescriptorImageInfo {
                            sampler: VkSampler::null(),
                            imageView: *image_view,
                            imageLayout: *image_layout,
                        },
                    });
                    write_descriptors.push(VkWriteDescriptorSet {
                        dstBinding: *binding,
                        descriptorType: VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                        descriptorCount: 1,
                        pImageInfo: &descriptor_infos.last().unwrap().image,
                        ..Default::default()
                    });
                }
                Descriptor::Sampler { binding, sampler } => {
                    descriptor_infos.push(DescriptorInfo {
                        image: VkDescriptorImageInfo {
                            sampler: *sampler,
                            imageView: VkImageView::null(),
                            imageLayout: VK_IMAGE_LAYOUT_UNDEFINED,
                        },
                    });
                    write_descriptors.push(VkWriteDescriptorSet {
                        dstBinding: *binding,
                        descriptorType: VK_DESCRIPTOR_TYPE_SAMPLER,
                        descriptorCount: 1,
                        pImageInfo: &descriptor_infos.last().unwrap().image,
                        ..Default::default()
                    });
                }
            }
        }
        self.CmdPushDescriptorSetKHR(
            cmdbuf,
            // We don't have compute shaders for the moment
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            pipeline_layout,
            // and we won't need more than one descriptor set
            0,
            write_descriptors.len() as u32,
            write_descriptors.as_ptr(),
        );
    }

    pub unsafe fn cmd_set_viewport_helper(&self, cmdbuf: VkCommandBuffer, x: i32, y: i32, w: i32, h: i32) {
        self.CmdSetViewport(
            cmdbuf,
            0,
            1,
            &VkViewport { x: x as f32, y: y as f32, width: w as f32, height: h as f32, minDepth: 0.0, maxDepth: 1.0 },
        );
    }

    pub unsafe fn cmd_set_scissor_helper(&self, cmdbuf: VkCommandBuffer, x: i32, y: i32, w: i32, h: i32) {
        self.CmdSetScissor(
            cmdbuf,
            0,
            1,
            &VkRect2D { offset: VkOffset2D { x, y }, extent: VkExtent2D { width: w as u32, height: h as u32 } },
        );
    }

    pub unsafe fn layout_barrier(
        &self,
        cmdbuf: VkCommandBuffer,
        transitions: &[(VkImage, VkImageLayout, VkImageLayout)],
    ) {
        // We are heavy-handed on the pipeline stages & access flags,
        // as this is not remotely worth the trouble.
        let barriers = transitions
            .iter()
            .map(|&(image, old_layout, new_layout)| VkImageMemoryBarrier {
                oldLayout: old_layout,
                newLayout: new_layout,
                srcQueueFamilyIndex: VK_QUEUE_FAMILY_IGNORED,
                dstQueueFamilyIndex: VK_QUEUE_FAMILY_IGNORED,
                image,
                subresourceRange: VkImageSubresourceRange {
                    aspectMask: VK_IMAGE_ASPECT_COLOR_BIT,
                    baseMipLevel: 0,
                    levelCount: VK_REMAINING_MIP_LEVELS,
                    baseArrayLayer: 0,
                    layerCount: VK_REMAINING_ARRAY_LAYERS,
                },
                srcAccessMask: VK_ACCESS_MEMORY_WRITE_BIT,
                dstAccessMask: VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
                ..
            })
            .collect::<Vec<_>>();
        self.CmdPipelineBarrier(
            cmdbuf,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            0,
            0,
            ptr::null(),
            0,
            ptr::null(),
            barriers.len() as u32,
            barriers.as_ptr(),
        );
    }

    pub unsafe fn create_graphics_pipeline_helper(&self, create_info: &GraphicsPipelineHelperCreateInfo) -> Pipeline {
        let shader_module = self
            .create_shader_module(
                &VkShaderModuleCreateInfo {
                    codeSize: create_info.spirv.len() * 4,
                    pCode: create_info.spirv.as_ptr(),
                    ..
                },
                None,
            )
            .expect("failed to create shader module");
        let push_constant_range = VkPushConstantRange {
            stageFlags: VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            offset: 0,
            size: create_info.push_constants_size as u32,
        };
        let descriptor_set_layout = self
            .create_descriptor_set_layout(
                &VkDescriptorSetLayoutCreateInfo {
                    flags: VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
                    bindingCount: create_info.bindings.len() as u32,
                    pBindings: create_info.bindings.as_ptr(),
                    ..
                },
                None,
            )
            .unwrap();
        let pipeline_layout = self
            .create_pipeline_layout(
                &VkPipelineLayoutCreateInfo {
                    setLayoutCount: 1,
                    pSetLayouts: &descriptor_set_layout,
                    pushConstantRangeCount: 1,
                    pPushConstantRanges: &push_constant_range,
                    ..
                },
                None,
            )
            .unwrap();
        let shader_stages = [
            VkPipelineShaderStageCreateInfo {
                stage: VK_SHADER_STAGE_VERTEX_BIT,
                module: shader_module,
                pName: create_info.vertex_entry.as_ptr(),
                ..Default::default()
            },
            VkPipelineShaderStageCreateInfo {
                stage: VK_SHADER_STAGE_FRAGMENT_BIT,
                module: shader_module,
                pName: create_info.fragment_entry.as_ptr(),
                ..Default::default()
            },
        ];
        let vertex_binding = VkVertexInputBindingDescription {
            binding: 0,
            stride: create_info.vertex_stride as u32,
            inputRate: VK_VERTEX_INPUT_RATE_VERTEX,
        };
        let vertex_input_state = VkPipelineVertexInputStateCreateInfo {
            vertexBindingDescriptionCount: 1,
            pVertexBindingDescriptions: &vertex_binding,
            vertexAttributeDescriptionCount: create_info.vertex_attributes.len() as u32,
            pVertexAttributeDescriptions: create_info.vertex_attributes.as_ptr(),
            ..
        };
        let input_assembly_state =
            VkPipelineInputAssemblyStateCreateInfo { topology: VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, .. };
        let viewport_state = VkPipelineViewportStateCreateInfo { viewportCount: 1, scissorCount: 1, .. };
        let rasterization_state = VkPipelineRasterizationStateCreateInfo {
            polygonMode: VK_POLYGON_MODE_FILL,
            cullMode: VK_CULL_MODE_NONE,
            frontFace: VK_FRONT_FACE_COUNTER_CLOCKWISE,
            lineWidth: 1.0,
            ..Default::default()
        };
        let multisample_state =
            VkPipelineMultisampleStateCreateInfo { rasterizationSamples: VK_SAMPLE_COUNT_1_BIT, ..Default::default() };
        // Standard "source-over" alpha compositing for the overlay.
        let blend_attachment = VkPipelineColorBlendAttachmentState {
            blendEnable: VK_TRUE,
            srcColorBlendFactor: VK_BLEND_FACTOR_SRC_ALPHA,
            dstColorBlendFactor: VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
            colorBlendOp: VK_BLEND_OP_ADD,
            srcAlphaBlendFactor: VK_BLEND_FACTOR_ONE,
            dstAlphaBlendFactor: VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
            alphaBlendOp: VK_BLEND_OP_ADD,
            colorWriteMask: 0xF,
        };
        let color_blend_state = VkPipelineColorBlendStateCreateInfo {
            attachmentCount: 1,
            pAttachments: &blend_attachment,
            ..Default::default()
        };
        let dynamic_states = [VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR];
        let dynamic_state = VkPipelineDynamicStateCreateInfo {
            dynamicStateCount: dynamic_states.len() as u32,
            pDynamicStates: dynamic_states.as_ptr(),
            ..Default::default()
        };
        let mut rendering_info = VkPipelineRenderingCreateInfo {
            colorAttachmentCount: 1,
            pColorAttachmentFormats: &create_info.color_attachment_format,
            ..Default::default()
        };
        let pipeline_create_info = VkGraphicsPipelineCreateInfo {
            pNext: &mut rendering_info as *const _ as *const c_void,
            pStages: shader_stages.as_ptr(),
            stageCount: shader_stages.len() as u32,
            pVertexInputState: &vertex_input_state,
            pInputAssemblyState: &input_assembly_state,
            pViewportState: &viewport_state,
            pRasterizationState: &rasterization_state,
            pMultisampleState: &multisample_state,
            pColorBlendState: &color_blend_state,
            pDynamicState: &dynamic_state,
            layout: pipeline_layout,
            renderPass: VkRenderPass::null(),
            ..
        };
        vkcall!(self.CreateGraphicsPipelines(self.device, VkPipelineCache::null(), 1, &pipeline_create_info, ptr::null(), @out let pipeline));
        // The shader module is no longer needed once the pipeline is built.
        self.destroy_shader_module(shader_module, None);
        Pipeline { pipeline, pipeline_layout, descriptor_set_layout }
    }

    pub(crate) unsafe fn create_compute_pipeline_helper(
        &self,
        spirv: &[u32],
        entry_point: &CStr,
        bindings: &[VkDescriptorSetLayoutBinding],
        push_constants_size: usize,
    ) -> Pipeline {
        let shader_module = self
            .create_shader_module(
                &VkShaderModuleCreateInfo { flags: 0, codeSize: spirv.len() * 4, pCode: spirv.as_ptr(), .. },
                None,
            )
            .expect("failed to create shader module");
        let push_constant_range = VkPushConstantRange {
            stageFlags: VK_SHADER_STAGE_COMPUTE_BIT,
            offset: 0,
            size: push_constants_size as u32,
        };
        let descriptor_set_layout = self
            .create_descriptor_set_layout(
                &VkDescriptorSetLayoutCreateInfo {
                    flags: VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
                    bindingCount: bindings.len() as u32,
                    pBindings: bindings.as_ptr(),
                    ..
                },
                None,
            )
            .unwrap();
        let pipeline_layout = self
            .create_pipeline_layout(
                &VkPipelineLayoutCreateInfo {
                    setLayoutCount: 1,
                    pSetLayouts: &descriptor_set_layout,
                    pushConstantRangeCount: 1,
                    pPushConstantRanges: &push_constant_range,
                    ..
                },
                None,
            )
            .unwrap();
        let compute_pipeline_create_info = VkComputePipelineCreateInfo {
            stage: VkPipelineShaderStageCreateInfo {
                stage: VK_SHADER_STAGE_COMPUTE_BIT,
                module: shader_module,
                pName: entry_point.as_ptr(),
                ..
            },
            layout: pipeline_layout,
            ..
        };
        vkcall!(self.CreateComputePipelines(
            self.device,
            VkPipelineCache::null(),
            1,
            &compute_pipeline_create_info,
            ptr::null_mut(),
            @out let pipeline
        ));
        self.DestroyShaderModule(self.device, shader_module, ptr::null());
        Pipeline { pipeline_layout, descriptor_set_layout, pipeline }
    }

    pub unsafe fn create_color_image_helper(
        &self,
        format: VkFormat,
        width: u32,
        height: u32,
        usage: VkImageUsageFlags,
    ) -> Image {
        let vk = &self.dispatch;
        let device = self.device;
        let create_info = VkImageCreateInfo {
            imageType: VK_IMAGE_TYPE_2D,
            format,
            extent: VkExtent3D { width, height, depth: 1 },
            mipLevels: 1,
            arrayLayers: 1,
            samples: VK_SAMPLE_COUNT_1_BIT,
            tiling: VK_IMAGE_TILING_OPTIMAL,
            usage,
            sharingMode: VK_SHARING_MODE_EXCLUSIVE,
            initialLayout: VK_IMAGE_LAYOUT_UNDEFINED,
            ..
        };

        vkcall!(vk.CreateImage(device, &create_info, ptr::null(), @out let image));
        vkcallnc!(vk.GetImageMemoryRequirements(device, image, @out let img_req));
        let img_mem_type = self.find_memory_type(img_req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        vkcall!(vk.AllocateMemory(
            device,
            &VkMemoryAllocateInfo {
                allocationSize: img_req.size,
                memoryTypeIndex: img_mem_type,
                ..Default::default()
            },
            ptr::null(),
            @out let image_memory
        ));
        vk.BindImageMemory(device, image, image_memory, 0).check();
        vkcall!(vk.CreateImageView(
            device,
            &VkImageViewCreateInfo {
                image,
                viewType: VK_IMAGE_VIEW_TYPE_2D,
                format: create_info.format,
                subresourceRange: VkImageSubresourceRange {
                    aspectMask: VK_IMAGE_ASPECT_COLOR_BIT,
                    baseMipLevel: 0,
                    levelCount: 1,
                    baseArrayLayer: 0,
                    layerCount: 1,
                },
                ..Default::default()
            },
            ptr::null(),
            @out let image_view
        ));
        Image { image, image_view, memory: image_memory }
    }

    pub unsafe fn destroy_image_helper(&self, image: Image) {
        self.DestroyImageView(self.device, image.image_view, ptr::null());
        self.DestroyImage(self.device, image.image, ptr::null());
        self.FreeMemory(self.device, image.memory, ptr::null());
    }

    pub unsafe fn create_buffer_from_data<T: Copy + 'static>(&self, usage: VkBufferUsageFlags, data: &[T]) -> Buffer {
        let data_bytes = std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * std::mem::size_of::<T>());
        self.create_buffer_helper(usage, data_bytes.len(), Some(data_bytes))
    }

    pub unsafe fn create_buffer_helper(
        &self,
        usage: VkBufferUsageFlags,
        byte_size: usize,
        initial_data: Option<&[u8]>,
    ) -> Buffer {
        let create_info = VkBufferCreateInfo {
            size: byte_size as u64,
            // We may not need TRANSFER_DST if there's no initial data,
            // but adding it most likely doesn't have any perf impact whatsoever
            usage: usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            sharingMode: VK_SHARING_MODE_EXCLUSIVE,
            ..
        };
        let required_flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        vkcall!(self.CreateBuffer(self.device, &create_info, ptr::null(), @out let buffer));
        //let buffer = self.create_buffer(&create_info, None).unwrap();
        vkcallnc!(self.GetBufferMemoryRequirements(self.device, buffer, @out let buf_req));
        //let buf_req = self.get_buffer_memory_requirements(buffer);
        let buf_mem_type = self.find_memory_type(buf_req.memoryTypeBits, required_flags);
        let allocate_flags = VkMemoryAllocateFlagsInfo { flags: VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT, .. };
        vkcall!(self.AllocateMemory(
            self.device,
            &VkMemoryAllocateInfo {
                pNext: &allocate_flags as *const _ as *const c_void,
                allocationSize: buf_req.size,
                memoryTypeIndex: buf_mem_type,
                ..
            },
            ptr::null(),
            @out let buffer_memory
        ));
        self.BindBufferMemory(self.device, buffer, buffer_memory, 0).check();
        vkcall!(self.MapMemory(self.device, buffer_memory, 0, buf_req.size, 0, @out let ptr));
        if let Some(initial_data) = initial_data {
            ptr::copy_nonoverlapping(initial_data.as_ptr(), ptr.cast::<u8>(), initial_data.len());
        }
        let device_address = self.GetBufferDeviceAddress(self.device, &VkBufferDeviceAddressInfo { buffer, .. });
        Buffer { buffer, memory: buffer_memory, ptr, size: byte_size, device_address }
    }

    pub unsafe fn destroy_buffer_helper(&self, buffer: Buffer) {
        self.DestroyBuffer(self.device, buffer.buffer, ptr::null());
        self.FreeMemory(self.device, buffer.memory, ptr::null());
    }

    pub unsafe fn submit_oneshot(&self, record_fn: impl FnOnce(&Self, VkCommandBuffer)) {
        let mut cmdbuf = VkCommandBuffer::null();
        self.AllocateCommandBuffers(
            self.device,
            &VkCommandBufferAllocateInfo {
                commandPool: self.command_pool,
                level: VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                commandBufferCount: 1,
                ..
            },
            &mut cmdbuf,
        )
        .check();
        self.set_device_loader_data(cmdbuf);
        self.BeginCommandBuffer(
            cmdbuf,
            &VkCommandBufferBeginInfo { flags: VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, ..Default::default() },
        )
        .check();
        record_fn(self, cmdbuf);
        self.EndCommandBuffer(cmdbuf).check();
        self.QueueSubmit(
            self.queue,
            1,
            &VkSubmitInfo { commandBufferCount: 1, pCommandBuffers: &cmdbuf, ..Default::default() },
            VkFence::null(),
        )
        .check();
    }

    pub(crate) unsafe fn queue_submit_helper(
        &self,
        queue: VkQueue,
        cmd_buf: VkCommandBuffer,
        wait_semaphores: &[VkSemaphore],
        signal_semaphores: &[VkSemaphore],
        signal_fence: VkFence,
    ) -> VkResult {
        // It's a sad thing that we have to allocate memory dynamically for something that is probably
        // ignored by the driver, but here we are.
        let wait_dst_stage_mask = vec![VK_PIPELINE_STAGE_ALL_COMMANDS_BIT; wait_semaphores.len()];
        let submit_info = VkSubmitInfo {
            waitSemaphoreCount: wait_semaphores.len() as u32,
            pWaitSemaphores: wait_semaphores.as_ptr(),
            pWaitDstStageMask: wait_dst_stage_mask.as_ptr(),
            commandBufferCount: 1,
            pCommandBuffers: &cmd_buf,
            signalSemaphoreCount: signal_semaphores.len() as u32,
            pSignalSemaphores: signal_semaphores.as_ptr(),
            ..
        };
        self.QueueSubmit(queue, 1, &submit_info, signal_fence)
    }

    pub(crate) unsafe fn queue_present_helper(
        &self,
        queue: VkQueue,
        swapchain: VkSwapchainKHR,
        image_index: u32,
        wait_semaphore: VkSemaphore,
    ) -> VkResult {
        let present_info = VkPresentInfoKHR {
            waitSemaphoreCount: 1,
            pWaitSemaphores: &wait_semaphore,
            swapchainCount: 1,
            pSwapchains: &swapchain,
            pImageIndices: &image_index,
            ..
        };
        self.khr_swapchain.QueuePresentKHR(queue, &present_info).check()
    }

    pub(crate) unsafe fn push_constants_helper<T: Copy + 'static>(
        &self,
        cmd_buf: VkCommandBuffer,
        pipeline_layout: VkPipelineLayout,
        stages: VkShaderStageFlags,
        data: &T,
    ) {
        self.CmdPushConstants(
            cmd_buf,
            pipeline_layout,
            stages,
            0,
            size_of::<T>() as u32,
            data as *const _ as *const c_void,
        );
    }

    pub(crate) unsafe fn create_color_image_from_data(
        &self,
        format: VkFormat,
        width: u32,
        height: u32,
        usage: VkImageUsageFlags,
        data: &[u8],
    ) -> Image {
        let image = self.create_color_image_helper(format, width, height, usage);

        // Staging buffer: host-visible, coherent.
        let staging_buf = self.create_buffer_from_data(VK_BUFFER_USAGE_TRANSFER_SRC_BIT, data);

        self.submit_oneshot(|device, upload_cmdbuf| {
            self.layout_barrier(
                upload_cmdbuf,
                &[(image.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)],
            );

            device.cmd_copy_buffer_to_image(
                upload_cmdbuf,
                staging_buf.buffer,
                image.image,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                &[VkBufferImageCopy {
                    bufferOffset: 0,
                    bufferRowLength: 0,
                    bufferImageHeight: 0,
                    imageSubresource: VkImageSubresourceLayers {
                        aspectMask: VK_IMAGE_ASPECT_COLOR_BIT,
                        mipLevel: 0,
                        baseArrayLayer: 0,
                        layerCount: 1,
                    },
                    imageOffset: VkOffset3D { x: 0, y: 0, z: 0 },
                    imageExtent: VkExtent3D { width, height, depth: 1 },
                }],
            );

            self.layout_barrier(
                upload_cmdbuf,
                &[(image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)],
            );
        });

        // Cleanup transient resources.
        self.DeviceWaitIdle(self.device).check();
        self.destroy_buffer_helper(staging_buf);
        image
    }
}

fn spirv_u8_to_u32(spv: &[u8]) -> Vec<u32> {
    // It would be better if the input slice was al
    assert_eq!(spv.len() % 4, 0, "SPIR-V size must be a multiple of 4");
    spv.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect()
}
