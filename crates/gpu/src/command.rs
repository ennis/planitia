use ash::prelude::VkResult;
use fxhash::FxHashMap;
pub use render::{RenderEncoder, RenderPassInfo};
use slotmap::secondary::Entry;
use std::ffi::{c_char, c_void, CString};
use std::mem::{ManuallyDrop, MaybeUninit};
use std::sync::atomic::Ordering::Relaxed;
use std::{mem, ptr, slice};
use log::trace;
use crate::device::ActiveSubmission;
use crate::{aspects_for_format, vk, vk_ext_debug_utils, CommandPool, ComputePipeline, Descriptor, Device, Image, MemoryAccess, ResourceId, SwapchainImage, TrackedResource};

mod blit;
mod render;

////////////////////////////////////////////////////////////////////////////////////////////////////

union DescriptorBufferOrImage {
    image: vk::DescriptorImageInfo,
    buffer: vk::DescriptorBufferInfo,
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// Describes a pipeline barrier.
///
/// TODO: This should be refactored so that it doesn't need to allocate a vec on each layout transition.
pub struct Barrier<'a> {
    access: MemoryAccess,
    transitions: Vec<(&'a Image, MemoryAccess)>,
}

impl<'a> Barrier<'a> {
    pub fn new() -> Self {
        Barrier {
            access: MemoryAccess::empty(),
            transitions: vec![],
        }
    }

    pub fn color_attachment_write(mut self, image: &'a Image) -> Self {
        self.transitions.push((image, MemoryAccess::COLOR_ATTACHMENT_WRITE));
        self.access |= MemoryAccess::COLOR_ATTACHMENT_WRITE;
        self
    }

    pub fn depth_stencil_attachment_write(mut self, image: &'a Image) -> Self {
        self.transitions
            .push((image, MemoryAccess::DEPTH_STENCIL_ATTACHMENT_WRITE));
        self.access |= MemoryAccess::DEPTH_STENCIL_ATTACHMENT_WRITE;
        self
    }

    pub fn shader_storage_read(mut self) -> Self {
        self.access |= MemoryAccess::SHADER_STORAGE_READ | MemoryAccess::ALL_STAGES;
        self
    }

    pub fn shader_storage_write(mut self) -> Self {
        self.access |= MemoryAccess::SHADER_STORAGE_WRITE | MemoryAccess::ALL_STAGES;
        self
    }

    pub fn shader_read_image(mut self, image: &'a Image) -> Self {
        self.transitions
            .push((image, MemoryAccess::SHADER_STORAGE_READ | MemoryAccess::ALL_STAGES));
        self.access |= MemoryAccess::SHADER_STORAGE_READ | MemoryAccess::ALL_STAGES;
        self
    }

    pub fn shader_write_image(mut self, image: &'a Image) -> Self {
        self.transitions
            .push((image, MemoryAccess::SHADER_STORAGE_WRITE | MemoryAccess::ALL_STAGES));
        self.access |= MemoryAccess::SHADER_STORAGE_WRITE | MemoryAccess::ALL_STAGES;
        self
    }

    pub fn present(mut self, image: &'a Image) -> Self {
        self.transitions.push((image, MemoryAccess::PRESENT));
        self.access |= MemoryAccess::PRESENT;
        self
    }

    pub fn sample_read_image(mut self, image: &'a Image) -> Self {
        self.transitions
            .push((image, MemoryAccess::SAMPLED_READ | MemoryAccess::ALL_STAGES));
        self.access |= MemoryAccess::SAMPLED_READ | MemoryAccess::ALL_STAGES;
        self
    }

    pub fn transfer_read(mut self) -> Self {
        self.access |= MemoryAccess::TRANSFER_READ;
        self
    }

    pub fn transfer_write(mut self) -> Self {
        self.access |= MemoryAccess::TRANSFER_WRITE;
        self
    }

    pub fn transfer_read_image(mut self, image: &'a Image) -> Self {
        self.transitions.push((image, MemoryAccess::TRANSFER_READ));
        self.access |= MemoryAccess::TRANSFER_READ;
        self
    }

    pub fn transfer_write_image(mut self, image: &'a Image) -> Self {
        self.transitions.push((image, MemoryAccess::TRANSFER_WRITE));
        self.access |= MemoryAccess::TRANSFER_WRITE;
        self
    }
}

/// TODO rename this, it's not really a stream as it needs to be dropped to submit work to the queue
pub struct CommandStream {
    command_pool: ManuallyDrop<CommandPool>,
    submission_index: u64,
    /// Command buffers waiting to be submitted.
    command_buffers_to_submit: Vec<vk::CommandBuffer>,
    /// Current command buffer.
    command_buffer: Option<vk::CommandBuffer>,

    // Buffer writes that need to be made available
    tracked_writes: MemoryAccess,
    tracked_images: FxHashMap<ResourceId, CommandBufferImageState>,
    //pub(crate) tracked_image_views: FxHashMap<ImageViewId, ImageView>,
    seen_initial_barrier: bool,
    //initial_writes: MemoryAccess,
    initial_access: MemoryAccess,
    submitted: bool,
    /// Last bound compute pipeline layout.
    pipeline_layout: vk::PipelineLayout,
}

pub(crate) struct CommandBufferImageState {
    pub image: vk::Image,
    pub format: vk::Format,
    pub id: ResourceId,
    pub first_access: MemoryAccess,
    pub last_access: MemoryAccess,
}

/// A wrapper around a signaled binary semaphore.
///
/// This should be used in a wait operation, otherwise the semaphore will be leaked.
#[derive(Debug)]
pub struct SignaledSemaphore(pub(crate) vk::Semaphore);

impl SignaledSemaphore {
    pub fn wait(self) -> SemaphoreWait {
        self.wait_dst_stage(vk::PipelineStageFlags::ALL_COMMANDS)
    }

    pub fn wait_dst_stage(self, dst_stage: vk::PipelineStageFlags) -> SemaphoreWait {
        SemaphoreWait {
            kind: SemaphoreWaitKind::Binary {
                semaphore: self.0,
                transfer_ownership: true,
            },
            dst_stage,
        }
    }
}

/// A wrapper around an unsignaled binary semaphore.
#[derive(Debug)]
pub struct UnsignaledSemaphore(pub(crate) vk::Semaphore);

/// Describes the type of semaphore in a semaphore wait operation.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum SemaphoreWaitKind {
    /// Binary semaphore wait.
    Binary {
        /// The semaphore to wait on.
        semaphore: vk::Semaphore,
        /// Whether to transfer ownership of the semaphore to the queue.
        transfer_ownership: bool,
    },
    /// Timeline semaphore wait.
    Timeline { semaphore: vk::Semaphore, value: u64 },
    /// D3D12 fence wait.
    D3D12Fence {
        semaphore: vk::Semaphore,
        fence: vk::Fence,
        value: u64,
    },
}

/// Describes a semaphore wait operation.
#[derive(Clone, Debug)]
pub struct SemaphoreWait {
    /// The kind of wait operation.
    pub kind: SemaphoreWaitKind,
    /// Destination stage
    pub dst_stage: vk::PipelineStageFlags,
}

/// Describes the kind of semaphore signal operation.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum SemaphoreSignal {
    /// Binary semaphore signal.
    Binary {
        /// The semaphore to signal.
        semaphore: vk::Semaphore,
    },
    /// Timeline semaphore signal.
    Timeline {
        /// The semaphore to signal.
        semaphore: vk::Semaphore,
        /// The value to signal.
        value: u64,
    },
    /// D3D12 fence signal.
    D3D12Fence {
        /// The semaphore to signal.
        semaphore: vk::Semaphore,
        /// The fence to signal.
        fence: vk::Fence,
        /// The value to signal.
        value: u64,
    },
}

impl CommandStream {

    /// Creates a command stream used to submit commands to the GPU.
    ///
    /// Once finished, the command stream should be submitted to the GPU using
    /// `CommandStream::flush`.
    /// They should be submitted in the same order as they were created.
    pub fn new() -> CommandStream {
        let device = Device::global();
        let submission_index = device.next_submission_index.fetch_add(1, Relaxed);
        let command_pool = device.get_or_create_command_pool(device.queue_family);
        trace!("GPU: begin CommandStream {}", submission_index);

        CommandStream {
            command_pool: ManuallyDrop::new(command_pool),
            submission_index,
            command_buffers_to_submit: vec![],
            command_buffer: None,
            tracked_images: Default::default(),
            seen_initial_barrier: false,
            initial_access: MemoryAccess::empty(),
            tracked_writes: MemoryAccess::empty(),
            submitted: false,
            pipeline_layout: Default::default(),
        }
    }

    unsafe fn bind_bindless_descriptor_sets(
        &mut self,
        command_buffer: vk::CommandBuffer,
        bind_point: vk::PipelineBindPoint,
        pipeline_layout: vk::PipelineLayout,
    ) {
        let device = Device::global();
        device.raw.cmd_bind_descriptor_sets(
            command_buffer,
            bind_point,
            pipeline_layout,
            0,
            &[device.descriptor_table.set],
            &[],
        );
    }

    unsafe fn do_cmd_push_descriptor_set(
        &mut self,
        command_buffer: vk::CommandBuffer,
        bind_point: vk::PipelineBindPoint,
        pipeline_layout: vk::PipelineLayout,
        set: u32,
        bindings: &[(u32, Descriptor)],
    ) {
        let mut descriptors = Vec::with_capacity(bindings.len());
        let mut descriptor_writes = Vec::with_capacity(bindings.len());

        for (binding, descriptor) in bindings {
            match *descriptor {
                Descriptor::SampledImage {
                    image: image_view,
                    layout,
                } => {
                    self.reference_resource(image_view);
                    descriptors.push(DescriptorBufferOrImage {
                        image: vk::DescriptorImageInfo {
                            sampler: Default::default(),
                            image_view: image_view.view_handle(),
                            image_layout: layout,
                        },
                    });
                    descriptor_writes.push(vk::WriteDescriptorSet {
                        dst_binding: *binding,
                        dst_array_element: 0,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                        p_image_info: &descriptors.last().unwrap().image,
                        ..Default::default()
                    });
                }
                Descriptor::StorageImage {
                    image: image_view,
                    layout,
                } => {
                    self.reference_resource(image_view);
                    descriptors.push(DescriptorBufferOrImage {
                        image: vk::DescriptorImageInfo {
                            sampler: Default::default(),
                            image_view: image_view.view_handle(),
                            image_layout: layout,
                        },
                    });
                    descriptor_writes.push(vk::WriteDescriptorSet {
                        dst_binding: *binding,
                        dst_array_element: 0,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                        p_image_info: &descriptors.last().unwrap().image,
                        ..Default::default()
                    });
                }
                Descriptor::UniformBuffer { buffer, offset, size } => {
                    self.reference_resource(buffer);
                    descriptors.push(DescriptorBufferOrImage {
                        buffer: vk::DescriptorBufferInfo {
                            buffer: buffer.handle(),
                            offset,
                            range: size,
                        },
                    });
                    descriptor_writes.push(vk::WriteDescriptorSet {
                        dst_binding: *binding,
                        dst_array_element: 0,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                        p_buffer_info: &descriptors.last().unwrap().buffer,
                        ..Default::default()
                    });
                }
                Descriptor::StorageBuffer { buffer, offset, size } => {
                    self.reference_resource(buffer);
                    descriptors.push(DescriptorBufferOrImage {
                        buffer: vk::DescriptorBufferInfo {
                            buffer: buffer.handle(),
                            offset,
                            range: size,
                        },
                    });
                    descriptor_writes.push(vk::WriteDescriptorSet {
                        dst_binding: *binding,
                        dst_array_element: 0,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                        p_buffer_info: &descriptors.last().unwrap().buffer,
                        ..Default::default()
                    });
                }
                Descriptor::Sampler { ref sampler } => {
                    descriptors.push(DescriptorBufferOrImage {
                        image: vk::DescriptorImageInfo {
                            sampler: sampler.handle(),
                            image_view: Default::default(),
                            image_layout: Default::default(),
                        },
                    });
                    descriptor_writes.push(vk::WriteDescriptorSet {
                        dst_binding: *binding,
                        dst_array_element: 0,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::SAMPLER,
                        p_image_info: &descriptors.last().unwrap().image,
                        ..Default::default()
                    });
                }
            }
        }

        unsafe {
            Device::global().khr_push_descriptor().cmd_push_descriptor_set(
                command_buffer,
                bind_point,
                pipeline_layout,
                set,
                &descriptor_writes,
            );
        }
    }

    /// Binds push constants.
    fn do_cmd_push_constants(
        &mut self,
        command_buffer: vk::CommandBuffer,
        bind_point: vk::PipelineBindPoint,
        pipeline_layout: vk::PipelineLayout,
        data: &[MaybeUninit<u8>],
    ) {
        let size = size_of_val(data);

        // Minimum push constant size guaranteed by Vulkan is 128 bytes.
        assert!(size <= 128, "push constant size must be <= 128 bytes");
        assert!(size % 4 == 0, "push constant size must be a multiple of 4 bytes");

        // None of the relevant drivers on desktop care about the actual stages,
        // only if it's graphics, compute, or ray tracing.
        let stages = match bind_point {
            vk::PipelineBindPoint::GRAPHICS => {
                vk::ShaderStageFlags::ALL_GRAPHICS | vk::ShaderStageFlags::MESH_EXT | vk::ShaderStageFlags::TASK_EXT
            }
            vk::PipelineBindPoint::COMPUTE => vk::ShaderStageFlags::COMPUTE,
            _ => panic!("unsupported bind point"),
        };

        // Use the raw function pointer because the wrapper takes a `&[u8]` slice which we can't
        // get from `&[MaybeUninit<u8>]` safely (even if we won't read uninitialized data).
        unsafe {
            (Device::global().raw.fp_v1_0().cmd_push_constants)(
                command_buffer,
                pipeline_layout,
                stages,
                0,
                size as u32,
                data as *const _ as *const c_void,
            );
        }
    }

    /*/// Tells the command stream that an operation has made writes that are not available to
    /// subsequent operations.
    pub fn invalidate(&mut self, scope: MemoryAccess) {
        self.tracked_writes |= scope;
    }*/

    /// Emits a pipeline barrier (if necessary) that ensures that all previous writes are
    /// visible to subsequent operations for the given memory access type.
    ///
    /// Note that it's not possible to make only one specific type of write available. All pending
    /// writes are made available unconditionally.
    ///
    // TODO split in two parameters: one for global memory barrier, one for image layout transitions
    pub fn barrier(&mut self, barrier: Barrier) {
        let mut global_memory_barrier = None;
        let mut image_barriers = vec![];

        if !self.seen_initial_barrier {
            self.initial_access = barrier.access;
            self.seen_initial_barrier = true;
        } else {
            let (src_stage_mask, src_access_mask) = self.tracked_writes.to_vk_scope_flags();
            let (dst_stage_mask, dst_access_mask) = barrier.access.to_vk_scope_flags();
            global_memory_barrier = Some(vk::MemoryBarrier2 {
                src_access_mask,
                dst_access_mask,
                src_stage_mask,
                dst_stage_mask,
                ..Default::default()
            });
        }

        for (image, access) in barrier.transitions {
            if let Some(entry) = self.tracked_images.get_mut(&image.id()) {
                if entry.last_access != access {
                    let (src_stage_mask, src_access_mask) = entry.last_access.to_vk_scope_flags();
                    let (dst_stage_mask, dst_access_mask) = access.to_vk_scope_flags();
                    image_barriers.push(vk::ImageMemoryBarrier2 {
                        src_stage_mask,
                        src_access_mask,
                        dst_stage_mask,
                        dst_access_mask,
                        image: image.handle(),
                        old_layout: entry.last_access.to_vk_image_layout(image.format()),
                        new_layout: access.to_vk_image_layout(image.format()),
                        subresource_range: vk::ImageSubresourceRange {
                            aspect_mask: aspects_for_format(image.format()),
                            base_mip_level: 0,
                            level_count: vk::REMAINING_MIP_LEVELS,
                            base_array_layer: 0,
                            layer_count: vk::REMAINING_ARRAY_LAYERS,
                        },
                        ..Default::default()
                    });
                }
                entry.last_access = access;
            } else {
                self.tracked_images.insert(
                    image.id(),
                    CommandBufferImageState {
                        image: image.handle(),
                        format: image.format(),
                        id: image.id(),
                        first_access: access,
                        last_access: access,
                    },
                );
            }
        }

        if global_memory_barrier.is_some() || !image_barriers.is_empty() {
            // a global memory barrier is needed or there are image layout transitions
            let command_buffer = self.get_or_create_command_buffer();
            unsafe {
                Device::global().raw.cmd_pipeline_barrier2(
                    command_buffer,
                    &vk::DependencyInfo {
                        dependency_flags: Default::default(),
                        memory_barrier_count: global_memory_barrier.iter().len() as u32,
                        p_memory_barriers: global_memory_barrier
                            .as_ref()
                            .map(|b| b as *const vk::MemoryBarrier2)
                            .unwrap_or(ptr::null()),
                        buffer_memory_barrier_count: 0,
                        p_buffer_memory_barriers: ptr::null(),
                        image_memory_barrier_count: image_barriers.len() as u32,
                        p_image_memory_barriers: image_barriers.as_ptr(),
                        ..Default::default()
                    },
                );
            }
        }

        self.tracked_writes = barrier.access.write_flags();
    }

    pub unsafe fn bind_descriptor_set(&mut self, index: u32, set: vk::DescriptorSet) {
        let cb = self.get_or_create_command_buffer();
        Device::global().raw.cmd_bind_descriptor_sets(
            cb,
            vk::PipelineBindPoint::COMPUTE,
            self.pipeline_layout,
            index,
            &[set],
            &[],
        )
    }

    /// Sets push descriptors.
    pub fn push_descriptors(&mut self, set: u32, bindings: &[(u32, Descriptor)]) {
        assert!(
            self.pipeline_layout != vk::PipelineLayout::null(),
            "must have a pipeline bound before binding arguments"
        );

        let cb = self.get_or_create_command_buffer();
        unsafe {
            self.do_cmd_push_descriptor_set(
                cb,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                set,
                bindings,
            );
        }
    }

    // SAFETY: TBD
    pub fn bind_compute_pipeline(&mut self, pipeline: &ComputePipeline) {
        let device = Device::global();
        let cb = self.get_or_create_command_buffer();
        unsafe {
            device
                .raw
                .cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, pipeline.pipeline);
            if pipeline.bindless {
                self.bind_bindless_descriptor_sets(
                    cb,
                    vk::PipelineBindPoint::COMPUTE,
                    pipeline.pipeline_layout,
                );
            }
        }
        self.pipeline_layout = pipeline.pipeline_layout;

        // TODO: we need to hold a reference to the pipeline until the command buffers are submitted
    }

    /// Binds push constants.
    ///
    /// Push constants stay valid until the bound pipeline is changed.
    ///
    /// FIXME: this assumes that `bind_compute_pipeline` has been called.
    fn push_constants<P>(&mut self, data: &P)
    where
        P: Copy,
    {
        let cb = self.get_or_create_command_buffer();
        unsafe {
            self.do_cmd_push_constants(
                cb,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                slice::from_raw_parts(data as *const P as *const MaybeUninit<u8>, mem::size_of_val(data)),
            );
        }
    }

    pub fn dispatch<RootParams: Copy>(&mut self,  group_count_x: u32, group_count_y: u32, group_count_z: u32, root_params: &RootParams) {
        let cb = self.get_or_create_command_buffer();
        unsafe {
            self.do_cmd_push_constants(
                cb,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                slice::from_raw_parts(root_params as *const RootParams as *const MaybeUninit<u8>, size_of_val(root_params)),
            );
            Device::global()
                .raw
                .cmd_dispatch(cb, group_count_x, group_count_y, group_count_z);
        }
    }


    pub fn push_debug_group(&mut self, label: &str) {
        let command_buffer = self.get_or_create_command_buffer();
        unsafe {
            let label = CString::new(label).unwrap();
            vk_ext_debug_utils().cmd_begin_debug_utils_label(
                command_buffer,
                &vk::DebugUtilsLabelEXT {
                    p_label_name: label.as_ptr(),
                    color: [0.0, 0.0, 0.0, 0.0],
                    ..Default::default()
                },
            );
        }
    }

    pub fn pop_debug_group(&mut self) {
        // TODO check that push/pop calls are balanced
        let command_buffer = self.get_or_create_command_buffer();
        unsafe {
            vk_ext_debug_utils().cmd_end_debug_utils_label(command_buffer);
        }
    }

    pub fn debug_group(&mut self, label: &str, f: impl FnOnce(&mut Self)) {
        self.push_debug_group(label);
        f(self);
        self.pop_debug_group();
    }

    /// Specifies that the resource will be used in the current submission.
    pub fn reference_resource<R: TrackedResource>(&mut self, resource: &R) {
        // TODO this should be done during submission because it's possible to drop the command stream
        //      without submitting it
        //      (well, at least it would if we didn't explicitly panic on unsubmitted command streams)
        Device::global().set_last_submission_index(resource.id(), self.submission_index)
    }

    pub(crate) fn create_command_buffer_raw(&mut self) -> vk::CommandBuffer {
        let raw_device = Device::global().raw();
        let cb = self.command_pool.alloc(raw_device);

        unsafe {
            raw_device
                .begin_command_buffer(
                    cb,
                    &vk::CommandBufferBeginInfo {
                        flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                        ..Default::default()
                    },
                )
                .unwrap();
            //Device::global().set_object_name(cb, &format!("submission_{}", self.submission_index));
        }
        cb
    }

    /// Returns the current command buffer, creating a new one if necessary.
    ///
    /// The returned command buffer is ready to record commands.
    pub(crate) fn get_or_create_command_buffer(&mut self) -> vk::CommandBuffer {
        if let Some(cb) = self.command_buffer {
            cb
        } else {
            let cb = self.create_command_buffer_raw();
            self.command_buffer = Some(cb);
            cb
        }
    }

    /// Closes the current command buffer.
    ///
    /// This does nothing if there is no current command buffer.
    pub(crate) fn close_command_buffer(&mut self) {
        if let Some(cb) = self.command_buffer.take() {
            unsafe {
                Device::global().raw().end_command_buffer(cb).unwrap();
            }
            self.command_buffers_to_submit.push(cb);
        }
    }

    /// FIXME: this should acquire ownership of semaphores in `waits`
    ///        but then we won't be able to pass a slice
    pub fn flush(
        mut self,
        waits: &[SemaphoreWait],
        signals: &[SemaphoreSignal],
        present_image: Option<&SwapchainImage>,
    ) -> VkResult<()> {
        let device = Device::global();

        if let Some(present_image) = present_image {
            self.barrier(Barrier::new().present(&present_image.image));
        }

        //----------------------
        // /!\ Lock the device for command submission.
        // This effectively synchronizes submissions on the device.
        //----------------------
        let mut submission_state = device.submission_state.lock().unwrap();

        // Verify that the command streams are submitted in the order in which they were created.
        // Timeline semaphore values depend on this.
        assert!(!self.submitted);
        assert_eq!(
            device.expected_submission_index.load(Relaxed),
            self.submission_index,
            "CommandStream submitted out of order"
        );
        // Increment now so that this doesn't block other submissions if this one fails somehow.
        device
            .expected_submission_index
            .store(self.submission_index + 1, Relaxed);

        // finish recording the current command buffer if not already done
        self.close_command_buffer();

        // The complete list of command buffers to submit, including fixup command buffers between the ones passed to this function.
        let mut command_buffers = mem::take(&mut self.command_buffers_to_submit);

        //----------------------
        // Update tracked resources:
        //
        // Update the tracked state of each resource used in the command buffer,
        // and insert pipeline barriers if necessary.
        {
            let (src_stage_mask, src_access_mask) = submission_state.writes.to_vk_scope_flags();
            let (dst_stage_mask, dst_access_mask) = self.initial_access.to_vk_scope_flags();
            // TODO: verify that a barrier is necessary
            let global_memory_barrier = Some(vk::MemoryBarrier2 {
                src_stage_mask,
                src_access_mask,
                dst_stage_mask,
                dst_access_mask,
                ..Default::default()
            });

            let mut image_barriers = Vec::new();
            for (_, state) in self.tracked_images.drain() {
                let prev_access = match submission_state.access_per_resource.entry(state.id) {
                    Some(entry) => {
                        match entry {
                            Entry::Occupied(res) => mem::replace(res.into_mut(), state.last_access),
                            Entry::Vacant(res) => {
                                res.insert(state.last_access);
                                // if the image was not previously tracked, the contents are undefined
                                MemoryAccess::UNINITIALIZED
                            }
                        }
                    }
                    // if the image was not previously tracked, the contents are undefined
                    None => MemoryAccess::UNINITIALIZED,
                };
                if prev_access != state.first_access {
                    let format = state.format;
                    image_barriers.push(vk::ImageMemoryBarrier2 {
                        src_stage_mask,
                        src_access_mask,
                        dst_stage_mask,
                        dst_access_mask,
                        old_layout: prev_access.to_vk_image_layout(format),
                        new_layout: state.first_access.to_vk_image_layout(format),
                        image: state.image,
                        subresource_range: vk::ImageSubresourceRange {
                            aspect_mask: aspects_for_format(format),
                            base_mip_level: 0,
                            level_count: vk::REMAINING_MIP_LEVELS,
                            base_array_layer: 0,
                            layer_count: vk::REMAINING_ARRAY_LAYERS,
                        },
                        ..Default::default()
                    });
                }
            }

            // update tracked writes across submissions
            submission_state.writes = self.tracked_writes;

            // If we need a pipeline barrier before submitting the command buffers, we insert a "fixup" command buffer
            // containing the pipeline barrier, before the others.
            if global_memory_barrier.is_some() || !image_barriers.is_empty() {
                let fixup_cb = self.command_pool.alloc(&device.raw);
                unsafe {
                    device
                        .raw
                        .begin_command_buffer(
                            fixup_cb,
                            &vk::CommandBufferBeginInfo {
                                flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                                ..Default::default()
                            },
                        )
                        .unwrap();
                    vk_ext_debug_utils().cmd_begin_debug_utils_label(
                        fixup_cb,
                        &vk::DebugUtilsLabelEXT {
                            p_label_name: b"barrier fixup\0".as_ptr() as *const c_char,
                            color: [0.0, 0.0, 0.0, 0.0],
                            ..Default::default()
                        },
                    );
                    device.raw.cmd_pipeline_barrier2(
                        fixup_cb,
                        &vk::DependencyInfo {
                            dependency_flags: Default::default(),
                            memory_barrier_count: global_memory_barrier.iter().len() as u32,
                            p_memory_barriers: global_memory_barrier
                                .as_ref()
                                .map(|b| b as *const vk::MemoryBarrier2)
                                .unwrap_or(ptr::null()),
                            buffer_memory_barrier_count: 0,
                            p_buffer_memory_barriers: ptr::null(),
                            image_memory_barrier_count: image_barriers.len() as u32,
                            p_image_memory_barriers: image_barriers.as_ptr(),
                            ..Default::default()
                        },
                    );
                    vk_ext_debug_utils().cmd_end_debug_utils_label(fixup_cb);
                    device.raw.end_command_buffer(fixup_cb).unwrap();
                }
                command_buffers.insert(0, fixup_cb);
            }
        }

        // Build submission
        let mut signal_semaphores = Vec::new();
        let mut signal_semaphore_values = Vec::new();
        let mut wait_semaphores = Vec::new();
        let mut wait_semaphore_values = Vec::new();
        let mut wait_semaphore_dst_stages = Vec::new();
        let mut d3d12_fence_submit = false;

        // update the timeline semaphore with the submission index
        signal_semaphores.push(device.thread_safe.timeline);
        signal_semaphore_values.push(self.submission_index);

        // If presenting, signal the "render finished" semaphore associated with the swapchain image
        if let Some(present_image) = present_image {
            signal_semaphores.push(present_image.render_finished);
            signal_semaphore_values.push(0); // dummy
        }

        // setup semaphore signal operations
        for signal in signals.iter() {
            match signal {
                SemaphoreSignal::Binary { semaphore } => {
                    signal_semaphores.push(*semaphore);
                    signal_semaphore_values.push(0);
                }
                SemaphoreSignal::Timeline { semaphore, value } => {
                    signal_semaphores.push(*semaphore);
                    signal_semaphore_values.push(*value);
                }
                SemaphoreSignal::D3D12Fence {
                    semaphore,
                    fence: _,
                    value,
                } => {
                    signal_semaphores.push(*semaphore);
                    signal_semaphore_values.push(*value);
                    d3d12_fence_submit = true;
                }
            }
        }

        // setup semaphore wait operations
        for (_i, w) in waits.iter().enumerate() {
            wait_semaphore_dst_stages.push(w.dst_stage);
            match w.kind {
                SemaphoreWaitKind::Binary {
                    semaphore,
                    transfer_ownership,
                } => {
                    wait_semaphores.push(semaphore);
                    wait_semaphore_values.push(0);
                    if transfer_ownership {
                        // we own the semaphore and need to delete it
                        device.call_later(self.submission_index, move || unsafe {
                            Device::global().raw.destroy_semaphore(semaphore, None);
                        });
                    }
                }
                SemaphoreWaitKind::Timeline { semaphore, value } => {
                    wait_semaphores.push(semaphore);
                    wait_semaphore_values.push(value);
                }
                SemaphoreWaitKind::D3D12Fence { semaphore, value, .. } => {
                    wait_semaphores.push(semaphore);
                    wait_semaphore_values.push(value);
                    d3d12_fence_submit = true;
                }
            }
        }

        // setup D3D12 fence submissions
        let d3d12_fence_submit_info_ptr;
        let d3d12_fence_submit_info;

        if d3d12_fence_submit {
            d3d12_fence_submit_info = vk::D3D12FenceSubmitInfoKHR {
                wait_semaphore_values_count: wait_semaphore_values.len() as u32,
                p_wait_semaphore_values: wait_semaphore_values.as_ptr(),
                signal_semaphore_values_count: signal_semaphore_values.len() as u32,
                p_signal_semaphore_values: signal_semaphore_values.as_ptr(),
                ..Default::default()
            };
            d3d12_fence_submit_info_ptr = &d3d12_fence_submit_info as *const _ as *const c_void;
        } else {
            d3d12_fence_submit_info_ptr = ptr::null();
        }

        let timeline_submit_info = vk::TimelineSemaphoreSubmitInfo {
            p_next: d3d12_fence_submit_info_ptr,
            wait_semaphore_value_count: wait_semaphore_values.len() as u32,
            p_wait_semaphore_values: wait_semaphore_values.as_ptr(),
            signal_semaphore_value_count: signal_semaphore_values.len() as u32,
            p_signal_semaphore_values: signal_semaphore_values.as_ptr(),
            ..Default::default()
        };

        let submit_info = vk::SubmitInfo {
            p_next: &timeline_submit_info as *const _ as *const c_void,
            wait_semaphore_count: wait_semaphores.len() as u32,
            p_wait_semaphores: wait_semaphores.as_ptr(),
            p_wait_dst_stage_mask: wait_semaphore_dst_stages.as_ptr(),
            command_buffer_count: command_buffers.len() as u32,
            p_command_buffers: command_buffers.as_ptr(),
            signal_semaphore_count: signal_semaphores.len() as u32,
            p_signal_semaphores: signal_semaphores.as_ptr(),
            ..Default::default()
        };

        let mut result;
        unsafe {
            // SAFETY: apart from Vulkan handles being valid, Vulkan specifies that access to the
            //         queue object should be externally synchronized, which is realized here by the
            //         lock on submission_state.
            trace!("GPU: QueueSubmit");
            result = device
                .raw
                .queue_submit(submission_state.queue, &[submit_info], vk::Fence::null());

            if result.is_ok() {
                if let Some(present_image) = present_image {
                    result = Device::global()
                        .khr_swapchain()
                        .queue_present(
                            submission_state.queue,
                            &vk::PresentInfoKHR {
                                wait_semaphore_count: 1,
                                p_wait_semaphores: &present_image.render_finished,
                                swapchain_count: 1,
                                p_swapchains: &present_image.swapchain,
                                p_image_indices: &present_image.index,
                                p_results: ptr::null_mut(),
                                ..Default::default()
                            },
                        )
                        .map(|_| ());
                }
            }

            submission_state.active_submissions.push_back(ActiveSubmission {
                index: self.submission_index,
                // SAFETY: submitted = false so the command pool is valid
                command_pools: vec![ManuallyDrop::take(&mut self.command_pool)],
            });
        };

        self.submitted = true;

        result
    }
}

impl Drop for CommandStream {
    fn drop(&mut self) {
        if !self.submitted {
            panic!("CommandStream was not submitted before being dropped");
        }
    }
}
