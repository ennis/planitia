use crate::device::ActiveSubmission;
use crate::{
    aspects_for_format, vk, CommandPool, ComputePipeline, Descriptor, Device, Image, MemoryAccess, Ptr, ResourceId,
    SwapchainImage, TrackedResource,
};
use ash::prelude::VkResult;
use ash::vk::DeviceAddress;
use bitflags::bitflags;
use fxhash::FxHashMap;
use log::{error, trace};
pub use render::{DrawIndexedIndirectCommand, DrawIndirectCommand, RenderEncoder, RenderPassInfo};
use slotmap::secondary::Entry;
use std::ffi::{c_char, c_void, CString};
use std::mem::{ManuallyDrop, MaybeUninit};
use std::sync::atomic::Ordering::Relaxed;
use std::{mem, ptr, slice};

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

    pub fn indirect(mut self) -> Self {
        self.access |= MemoryAccess::INDIRECT_READ | MemoryAccess::ALL_STAGES;
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
    //tracked_writes: MemoryAccess,
    //tracked_images: FxHashMap<ResourceId, CommandBufferImageState>,

    //pub(crate) tracked_image_views: FxHashMap<ImageViewId, ImageView>,
    submitted: bool,
    /// Last bound compute pipeline layout.
    pipeline_layout: vk::PipelineLayout,

    //seen_initial_barrier: bool,
    //initial_writes: MemoryAccess,
    //initial_barrier: BarrierFlags,
    barrier_source: BarrierFlags,
}

pub(crate) struct CommandBufferImageState {
    pub image: vk::Image,
    pub format: vk::Format,
    pub id: ResourceId,
    pub first_access: MemoryAccess,
    pub last_access: MemoryAccess,
}

/*
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
            kind: SyncWait::Binary {
                semaphore: self.0,
                transfer_ownership: true,
            },
            dst_stage,
        }
    }
}*/

/*
/// Describes the type of semaphore in a semaphore wait operation.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum SyncWait {
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
}*/

/*/// Describes the kind of semaphore signal operation.
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
}*/

bitflags! {
    #[derive(Copy,Clone,Debug,PartialEq,Eq,Hash)]
    pub struct BarrierFlags: u64 {

        /// Transfer stage (execution only dependency)
        const TRANSFER = 1 << 0;

        #[doc(hidden)]
        const TRANSFER_MEMORY_READ =  1 << 14;
        #[doc(hidden)]
        const TRANSFER_MEMORY_WRITE = 1 << 15;

        const TRANSFER_READ =  Self::TRANSFER.bits() | Self::TRANSFER_MEMORY_READ.bits();
        const TRANSFER_WRITE = Self::TRANSFER.bits() | Self::TRANSFER_MEMORY_WRITE.bits();

        const VERTEX_SHADER = 1 << 1;
        const COMPUTE_SHADER = 1 << 2;
        const FRAGMENT_SHADER = 1 << 3;
        const MESH_SHADER = 1 << 4;
        const TASK_SHADER = 1 << 5;
        const ALL_SHADER_STAGES = Self::VERTEX_SHADER.bits() | Self::COMPUTE_SHADER.bits() | Self::FRAGMENT_SHADER.bits() | Self::MESH_SHADER.bits() | Self::TASK_SHADER.bits();

        /// Source: shader storage write
        /// Destination: shader storage read
        ///
        /// Equivalent to:
        /// - source:      SHADER_STORAGE_WRITE
        /// - destination: SHADER_STORAGE_READ
        const STORAGE = 1 << 6;

        /// Destination: shader sampled read
        const SAMPLED_READ = 1 << 7;

        /// Destination: indirect command read
        const INDIRECT_READ = 1 << 8;

        /// Destination: uniform read
        const UNIFORM_READ = 1 << 9;

        /// Color attachment read/output stage
        /// Destination: Color attachment read
        /// OR Source: Color attachment write + late fragment tests stage
        ///
        /// Equivalent to:
        /// - source:      COLOR_ATTACHMENT_OUTPUT + COLOR_ATTACHMENT_WRITE
        /// - destination: COLOR_ATTACHMENT_OUTPUT + COLOR_ATTACHMENT_READ
        const COLOR_ATTACHMENT = 1 << 10;

        /// Destination: Depth-stencil read + early fragment tests stage
        /// OR Source: Depth-stencil write + late fragment tests stage
        ///
        /// Equivalent to:
        /// - source:      LATE_FRAGMENT_TESTS  + DEPTH_STENCIL_ATTACHMENT_WRITE
        /// - destination: EARLY_FRAGMENT_TESTS + DEPTH_STENCIL_ATTACHMENT_READ
        const DEPTH_STENCIL = 1 << 11;

        /// Vertex input stage & vertex attribute read.
        const VERTEX_READ = 1 << 12;

        /// Index input stage & index read.
        const INDEX_READ = 1 << 13;
    }
}

impl BarrierFlags {
    fn shader_stage_flags(self) -> vk::PipelineStageFlags2 {
        let mut flags = vk::PipelineStageFlags2::empty();
        if self.contains(Self::VERTEX_SHADER) {
            flags |= vk::PipelineStageFlags2::VERTEX_SHADER;
        }
        if self.contains(Self::FRAGMENT_SHADER) {
            flags |= vk::PipelineStageFlags2::FRAGMENT_SHADER;
        }
        if self.contains(Self::COMPUTE_SHADER) {
            flags |= vk::PipelineStageFlags2::COMPUTE_SHADER;
        }
        if self.contains(Self::MESH_SHADER) {
            flags |= vk::PipelineStageFlags2::MESH_SHADER_EXT;
        }
        if self.contains(Self::TASK_SHADER) {
            flags |= vk::PipelineStageFlags2::TASK_SHADER_EXT;
        }
        flags
    }

    fn to_vk_barrier_src_flags(&self) -> (vk::PipelineStageFlags2, vk::AccessFlags2) {
        let mut stages = vk::PipelineStageFlags2::empty();
        let mut access = vk::AccessFlags2::empty();

        stages |= self.shader_stage_flags();

        if self.contains(Self::DEPTH_STENCIL) {
            stages |= vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS;
            access |= vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE;
        }
        if self.contains(Self::TRANSFER) {
            stages |= vk::PipelineStageFlags2::TRANSFER;
        }
        if self.contains(Self::TRANSFER_MEMORY_WRITE) {
            access |= vk::AccessFlags2::TRANSFER_WRITE;
        }
        if self.contains(Self::STORAGE) {
            access |= vk::AccessFlags2::SHADER_STORAGE_WRITE;
        }
        if self.contains(Self::COLOR_ATTACHMENT) {
            stages |= vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT;
            access |= vk::AccessFlags2::COLOR_ATTACHMENT_WRITE;
        }

        (stages, access)
    }

    fn to_vk_barrier_dst_flags(&self) -> (vk::PipelineStageFlags2, vk::AccessFlags2) {
        let mut stages = vk::PipelineStageFlags2::empty();
        let mut access = vk::AccessFlags2::empty();

        stages |= self.shader_stage_flags();

        if self.contains(Self::DEPTH_STENCIL) {
            stages |= vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS;
            access |= vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ;
        }
        if self.contains(Self::TRANSFER) {
            stages |= vk::PipelineStageFlags2::TRANSFER;
            if self.contains(Self::TRANSFER_MEMORY_READ) {
                // no need for a memory dependency if we're only writing
                access |= vk::AccessFlags2::TRANSFER_READ;
            }
        }
        if self.contains(Self::STORAGE) {
            access |= vk::AccessFlags2::SHADER_STORAGE_READ;
        }
        if self.contains(Self::COLOR_ATTACHMENT) {
            stages |= vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT;
            access |= vk::AccessFlags2::COLOR_ATTACHMENT_READ;
        }
        if self.contains(Self::SAMPLED_READ) {
            access |= vk::AccessFlags2::SHADER_SAMPLED_READ;
        }
        if self.contains(Self::INDIRECT_READ) {
            stages |= vk::PipelineStageFlags2::DRAW_INDIRECT;
            access |= vk::AccessFlags2::INDIRECT_COMMAND_READ;
        }
        if self.contains(Self::VERTEX_READ) {
            stages |= vk::PipelineStageFlags2::VERTEX_ATTRIBUTE_INPUT;
            access |= vk::AccessFlags2::VERTEX_ATTRIBUTE_READ;
        }
        if self.contains(Self::INDEX_READ) {
            stages |= vk::PipelineStageFlags2::INDEX_INPUT;
            access |= vk::AccessFlags2::INDEX_READ;
        }

        (stages, access)
    }
}

bitflags! {
    #[derive(Copy,Clone,Debug,PartialEq,Eq,Hash)]
    pub struct MemoryFlags: u32 {
        const STORAGE_WRITE = 0;
        const DEPTH_STENCIL_ATTACHMENT_WRITE = 0;
    }
}

/// Describes root parameters for a command.
#[derive(Clone, Copy)]
pub enum RootParams<'a, T: Copy + 'static> {
    /// Root parameters are provided as a GPU pointer to a structure.
    Ptr(Ptr<T>),
    /// Root parameters are provided as immediate data.
    Immediate(&'a T),
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
            //tracked_images: Default::default(),
            //seen_initial_barrier: false,
            //tracked_writes: MemoryAccess::empty(),
            submitted: false,
            pipeline_layout: Default::default(),
            barrier_source: BarrierFlags::empty(),
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
            Device::global().extensions.khr_push_descriptor.cmd_push_descriptor_set(
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

    pub(crate) unsafe fn transition_image_layout(
        &mut self,
        image: vk::Image,
        src_stage_mask: vk::PipelineStageFlags2,
        src_access_mask: vk::AccessFlags2,
        dst_stage_mask: vk::PipelineStageFlags2,
        dst_access_mask: vk::AccessFlags2,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
    ) {
        unsafe {
            let cb = self.get_or_create_command_buffer();
            let barrier = vk::ImageMemoryBarrier2 {
                src_stage_mask,
                src_access_mask,
                dst_stage_mask,
                dst_access_mask,
                old_layout,
                new_layout,
                image,
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: vk::REMAINING_MIP_LEVELS,
                    base_array_layer: 0,
                    layer_count: vk::REMAINING_ARRAY_LAYERS,
                },
                ..Default::default()
            };
            Device::global().raw.cmd_pipeline_barrier2(
                cb,
                &vk::DependencyInfo {
                    dependency_flags: Default::default(),
                    image_memory_barrier_count: 1,
                    p_image_memory_barriers: &barrier,
                    ..Default::default()
                },
            );
        }
    }

    fn set_root_params<T: Copy + 'static>(
        &mut self,
        command_buffer: vk::CommandBuffer,
        bind_point: vk::PipelineBindPoint,
        pipeline_layout: vk::PipelineLayout,
        params: RootParams<T>,
    ) {
        // None of the relevant drivers on desktop care about the actual stages,
        // only if it's graphics, compute, or ray tracing.
        let stages = match bind_point {
            vk::PipelineBindPoint::GRAPHICS => {
                vk::ShaderStageFlags::ALL_GRAPHICS | vk::ShaderStageFlags::MESH_EXT | vk::ShaderStageFlags::TASK_EXT
            }
            vk::PipelineBindPoint::COMPUTE => vk::ShaderStageFlags::COMPUTE,
            _ => panic!("unsupported bind point"),
        };

        let ptr = match params {
            RootParams::Ptr(p) => p.raw,
            RootParams::Immediate(data) => Device::global().upload(slice::from_ref(data)).raw,
        };

        unsafe {
            Device::global().raw.cmd_push_constants(
                command_buffer,
                pipeline_layout,
                stages,
                0,
                slice::from_raw_parts(&ptr as *const DeviceAddress as *const u8, size_of::<DeviceAddress>()),
            );
        }
    }

    /*/// Tells the command stream that an operation has made writes that are not available to
    /// subsequent operations.
    pub fn invalidate(&mut self, scope: MemoryAccess) {
        self.tracked_writes |= scope;
    }*/

    /// Tells future barriers to wait for the specified producer stages.
    ///
    /// Typically, if you want future barriers to wait for all prior work to complete, you would call this
    /// with the stages that are writing data in the previous commands.
    ///
    /// The "memory" parameter tells which types of non-coherent memory accesses should be made
    /// available at the next barrier. This should be set if prior commands write to non-coherent
    /// caches (e.g. ???) and you want to ensure that subsequent commands see the results of those writes.
    pub fn barrier_source(&mut self, flags: BarrierFlags) {
        self.barrier_source = flags;
    }

    /// Emits a pipeline barrier (if necessary) that ensures that all previous writes are
    /// visible to subsequent operations for the given memory access type.
    ///
    /// Note that it's not possible to make only one specific type of write available. All pending
    /// writes are made available unconditionally.
    ///
    // TODO split in two parameters: one for global memory barrier, one for image layout transitions
    pub fn barrier(&mut self, dest: BarrierFlags) {
        //let mut image_barriers = vec![];

        let (src_stage_mask, src_access_mask) = self.barrier_source.to_vk_barrier_src_flags();
        let (dst_stage_mask, dst_access_mask) = dest.to_vk_barrier_dst_flags();
        let global_memory_barrier = vk::MemoryBarrier2 {
            src_access_mask,
            dst_access_mask,
            src_stage_mask,
            dst_stage_mask,
            ..Default::default()
        };

        let command_buffer = self.get_or_create_command_buffer();
        unsafe {
            Device::global().raw.cmd_pipeline_barrier2(
                command_buffer,
                &vk::DependencyInfo {
                    dependency_flags: Default::default(),
                    memory_barrier_count: 1,
                    p_memory_barriers: &global_memory_barrier,
                    ..Default::default()
                },
            );
        }

        /*for (image, access) in barrier.transitions {
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
        }*/
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
            self.do_cmd_push_descriptor_set(cb, vk::PipelineBindPoint::COMPUTE, self.pipeline_layout, set, bindings);
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
                self.bind_bindless_descriptor_sets(cb, vk::PipelineBindPoint::COMPUTE, pipeline.pipeline_layout);
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
                slice::from_raw_parts(data as *const P as *const MaybeUninit<u8>, size_of_val(data)),
            );
        }
    }

    pub fn dispatch<T: Copy + 'static>(
        &mut self,
        group_count_x: u32,
        group_count_y: u32,
        group_count_z: u32,
        root_params: RootParams<T>,
    ) {
        let cb = self.get_or_create_command_buffer();
        unsafe {
            self.set_root_params(cb, vk::PipelineBindPoint::COMPUTE, self.pipeline_layout, root_params);
            Device::global()
                .raw
                .cmd_dispatch(cb, group_count_x, group_count_y, group_count_z);
        }
    }

    pub fn push_debug_group(&mut self, label: &str) {
        let command_buffer = self.get_or_create_command_buffer();
        unsafe {
            let label = CString::new(label).unwrap();
            Device::global().extensions.ext_debug_utils.cmd_begin_debug_utils_label(
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
            Device::global()
                .extensions
                .ext_debug_utils
                .cmd_end_debug_utils_label(command_buffer);
        }
    }

    pub fn debug_group(&mut self, label: &str, f: impl FnOnce(&mut Self)) {
        self.push_debug_group(label);
        f(self);
        self.pop_debug_group();
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
}

impl Drop for CommandStream {
    fn drop(&mut self) {
        if !self.submitted {
            panic!("CommandStream was not submitted before being dropped");
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#[derive(Copy, Clone)]
pub struct SyncWait {
    pub sync: vk::Semaphore,
    pub value: u64,
}

#[derive(Copy, Clone)]
pub struct SyncSignal {
    pub sync: vk::Semaphore,
    pub value: u64,
}

///
fn sync(waits: &[SyncWait], signals: &[SyncSignal]) {
    let device = Device::global();

    // /!\ Lock the device for command submission.
    let submission_state = device.submission_state.lock().unwrap();

    let wait_count = waits.len();
    let mut wait_semaphores = Vec::with_capacity(wait_count);
    let mut wait_semaphore_values = Vec::with_capacity(wait_count);
    let mut wait_semaphore_dst_stages = Vec::with_capacity(wait_count);

    let signal_count = signals.len();
    let mut signal_semaphores = Vec::with_capacity(signal_count);
    let mut signal_semaphore_values = Vec::with_capacity(signal_count);

    for signal in signals.iter() {
        signal_semaphores.push(signal.sync);
        signal_semaphore_values.push(signal.value);
    }

    for (_i, w) in waits.iter().enumerate() {
        wait_semaphore_dst_stages.push(vk::PipelineStageFlags::ALL_COMMANDS);
        wait_semaphores.push(w.sync);
        wait_semaphore_values.push(w.value);
    }

    let timeline_submit_info = vk::TimelineSemaphoreSubmitInfo {
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
        command_buffer_count: 0,
        p_command_buffers: ptr::null(),
        signal_semaphore_count: signal_semaphores.len() as u32,
        p_signal_semaphores: signal_semaphores.as_ptr(),
        ..Default::default()
    };

    unsafe {
        trace!("GPU: QueueSubmit (synchronization)");
        match device
            .raw
            .queue_submit(submission_state.queue, &[submit_info], vk::Fence::null())
        {
            Ok(()) => {}
            Err(e) => {
                error!("QueueSubmit (synchronization) failed: {:?}", e);
            }
        }
    }
}

pub fn sync_wait(semaphore: vk::Semaphore, value: u64) {
    sync(&[SyncWait { sync: semaphore, value }], &[]);
}

pub fn sync_signal(semaphore: vk::Semaphore, value: u64) {
    sync(&[], &[SyncSignal { sync: semaphore, value }]);
}

pub fn submit(mut cmd: CommandStream) -> VkResult<()> {
    let device = Device::global();

    //----------------------
    // /!\ Lock the device for command submission.
    // This effectively synchronizes submissions on the device.
    //----------------------
    let mut submission_state = device.submission_state.lock().unwrap();

    // Verify that the command streams are submitted in the order in which they were created.
    // Timeline semaphore values depend on this.
    assert!(!cmd.submitted);
    assert_eq!(
        device.expected_submission_index.load(Relaxed),
        cmd.submission_index,
        "CommandStream submitted out of order"
    );
    // Increment now so that this doesn't block other submissions if this one fails somehow.
    device
        .expected_submission_index
        .store(cmd.submission_index + 1, Relaxed);

    // flush pending writes
    cmd.barrier(BarrierFlags::empty());

    // finish recording the current command buffer if not already done
    cmd.close_command_buffer();

    let mut command_buffers = mem::take(&mut cmd.command_buffers_to_submit);

    //----------------------
    // Update tracked resources:
    //
    // Update the tracked state of each resource used in the command buffer,
    // and insert pipeline barriers if necessary.
    //{
    //    let (src_stage_mask, src_access_mask) = submission_state.writes.to_vk_scope_flags();
    //    let (dst_stage_mask, dst_access_mask) = cmd.initial_access.to_vk_scope_flags();
    //    // TODO: verify that a barrier is necessary
    //    let global_memory_barrier = Some(vk::MemoryBarrier2 {
    //        src_stage_mask,
    //        src_access_mask,
    //        dst_stage_mask,
    //        dst_access_mask,
    //        ..Default::default()
    //    });
//
    //    let mut image_barriers = Vec::new();
    //    for (_, state) in cmd.tracked_images.drain() {
    //        let prev_access = match submission_state.access_per_resource.entry(state.id) {
    //            Some(entry) => {
    //                match entry {
    //                    Entry::Occupied(res) => mem::replace(res.into_mut(), state.last_access),
    //                    Entry::Vacant(res) => {
    //                        res.insert(state.last_access);
    //                        // if the image was not previously tracked, the contents are undefined
    //                        MemoryAccess::UNINITIALIZED
    //                    }
    //                }
    //            }
    //            // if the image was not previously tracked, the contents are undefined
    //            None => MemoryAccess::UNINITIALIZED,
    //        };
    //        if prev_access != state.first_access {
    //            let format = state.format;
    //            image_barriers.push(vk::ImageMemoryBarrier2 {
    //                src_stage_mask,
    //                src_access_mask,
    //                dst_stage_mask,
    //                dst_access_mask,
    //                old_layout: prev_access.to_vk_image_layout(format),
    //                new_layout: state.first_access.to_vk_image_layout(format),
    //                image: state.image,
    //                subresource_range: vk::ImageSubresourceRange {
    //                    aspect_mask: aspects_for_format(format),
    //                    base_mip_level: 0,
    //                    level_count: vk::REMAINING_MIP_LEVELS,
    //                    base_array_layer: 0,
    //                    layer_count: vk::REMAINING_ARRAY_LAYERS,
    //                },
    //                ..Default::default()
    //            });
    //        }
    //    }
//
    //    // update tracked writes across submissions
    //    submission_state.writes = cmd.tracked_writes;
//
    //    // If we need a pipeline barrier before submitting the command buffers, we insert a "fixup" command buffer
    //    // containing the pipeline barrier, before the others.
    //    if global_memory_barrier.is_some() || !image_barriers.is_empty() {
    //        let fixup_cb = cmd.command_pool.alloc(&device.raw);
    //        unsafe {
    //            device
    //                .raw
    //                .begin_command_buffer(
    //                    fixup_cb,
    //                    &vk::CommandBufferBeginInfo {
    //                        flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
    //                        ..Default::default()
    //                    },
    //                )
    //                .unwrap();
    //            device.extensions.ext_debug_utils.cmd_begin_debug_utils_label(
    //                fixup_cb,
    //                &vk::DebugUtilsLabelEXT {
    //                    p_label_name: b"barrier fixup\0".as_ptr() as *const c_char,
    //                    color: [0.0, 0.0, 0.0, 0.0],
    //                    ..Default::default()
    //                },
    //            );
    //            device.raw.cmd_pipeline_barrier2(
    //                fixup_cb,
    //                &vk::DependencyInfo {
    //                    dependency_flags: Default::default(),
    //                    memory_barrier_count: global_memory_barrier.iter().len() as u32,
    //                    p_memory_barriers: global_memory_barrier
    //                        .as_ref()
    //                        .map(|b| b as *const vk::MemoryBarrier2)
    //                        .unwrap_or(ptr::null()),
    //                    buffer_memory_barrier_count: 0,
    //                    p_buffer_memory_barriers: ptr::null(),
    //                    image_memory_barrier_count: image_barriers.len() as u32,
    //                    p_image_memory_barriers: image_barriers.as_ptr(),
    //                    ..Default::default()
    //                },
    //            );
    //            device.extensions.ext_debug_utils.cmd_end_debug_utils_label(fixup_cb);
    //            device.raw.end_command_buffer(fixup_cb).unwrap();
    //        }
    //        command_buffers.insert(0, fixup_cb);
    //    }
    //}

    //----------------------
    // submit
    let signal_semaphores = vec![device.thread_safe.timeline];
    let signal_semaphore_values = vec![cmd.submission_index];
    let timeline_submit_info = vk::TimelineSemaphoreSubmitInfo {
        signal_semaphore_value_count: signal_semaphore_values.len() as u32,
        p_signal_semaphore_values: signal_semaphore_values.as_ptr(),
        ..Default::default()
    };
    let submit_info = vk::SubmitInfo {
        p_next: &timeline_submit_info as *const _ as *const c_void,
        command_buffer_count: command_buffers.len() as u32,
        p_command_buffers: command_buffers.as_ptr(),
        signal_semaphore_count: signal_semaphores.len() as u32,
        p_signal_semaphores: signal_semaphores.as_ptr(),
        ..Default::default()
    };

    let result;
    unsafe {
        // SAFETY: apart from Vulkan handles being valid, Vulkan specifies that access to the
        //         queue object should be externally synchronized, which is realized here by the
        //         lock on submission_state.
        trace!("GPU: QueueSubmit");
        result = device
            .raw
            .queue_submit(submission_state.queue, &[submit_info], vk::Fence::null());

        submission_state.active_submissions.push_back(ActiveSubmission {
            index: cmd.submission_index,
            // SAFETY: submitted = false so the command pool is valid
            command_pools: vec![ManuallyDrop::take(&mut cmd.command_pool)],
        });
    };

    cmd.submitted = true;
    result
}

pub fn present(image: &SwapchainImage) -> VkResult<()> {
    // transition image to PRESENT_SRC
    let mut cmd = CommandStream::new();
    unsafe {
        cmd.transition_image_layout(
            image.image.handle,
            vk::PipelineStageFlags2::ALL_COMMANDS,
            vk::AccessFlags2::MEMORY_WRITE,
            vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
            vk::AccessFlags2::NONE,
            vk::ImageLayout::GENERAL,
            vk::ImageLayout::PRESENT_SRC_KHR,
        );
    }
    submit(cmd)?;

    // NOTE: submission state is unlocked here, so it's possible that another thread submits
    //       commands to the image that was transitioned to PRESENT layout before we present it.
    //       This is up to the caller to avoid doing that.

    // set up semaphore to wait for rendering to finish
    let device = Device::global();
    let render_finished = device.get_or_create_semaphore();
    sync_signal(render_finished, 0);

    unsafe {
        let submission_state = device.submission_state.lock().unwrap();
        let result = Device::global()
            .extensions
            .khr_swapchain
            .queue_present(
                submission_state.queue,
                &vk::PresentInfoKHR {
                    wait_semaphore_count: 1,
                    p_wait_semaphores: &render_finished,
                    swapchain_count: 1,
                    p_swapchains: &image.swapchain,
                    p_image_indices: &image.index,
                    p_results: ptr::null_mut(),
                    ..Default::default()
                },
            )
            .map(|_| ());
        device.recycle_binary_semaphore(render_finished);
        result
    }
}
