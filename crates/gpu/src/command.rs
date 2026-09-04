use crate::device::ActiveSubmission;
use crate::{
    Buffer, BufferRangeUntyped, BufferUntyped, ColorAttachment, ComputePipeline, DepthStencilAttachment, Device, Image,
    ImageCopyBuffer, ImageCopyView, ImageCreateInfo, MAX_TIMESTAMP_QUERY_COUNT, Ptr, ShaderReflection, SwapChain,
    VulkanObject, command_pool, vk,
};
use arrayvec::ArrayVec;
use ash::prelude::VkResult;
use ash::vk::{DeviceAddress, Handle};
use bitflags::bitflags;
use gpu_types::{
    ClearColorValue, Data, ImageAspect, ImageDataLayout, ImageSubresourceLayers, ImageUsage, Offset3D, Rect3D, Size3D,
};
use log::{error, trace};
pub use render::{DrawIndexedIndirectCommand, DrawIndirectCommand, RenderEncoder};
use slotmap::new_key_type;
use std::cell::{BorrowMutError, RefCell, RefMut};
use std::ffi::{CString, c_void};
use std::marker::PhantomData;
use std::mem::ManuallyDrop;
use std::sync::atomic::Ordering::Relaxed;
use std::{mem, ptr, slice};
use vulkan_headers::vulkan::vulkan::{
    VK_STRUCTURE_TYPE_PUSH_DATA_INFO_EXT, VkHostAddressRangeConstEXT, VkPushDataInfoEXT,
};

mod blit;
mod render;

union DescriptorBufferOrImage {
    image: vk::DescriptorImageInfo,
    buffer: vk::DescriptorBufferInfo,
}

bitflags! {
    /// Describes the memory types to invalidate during a barrier operation.
    // Under the hood, these are just VkAccessFlags2 bits
    pub struct BarrierFlags: u64 {
        const STORAGE = vk::AccessFlags2::SHADER_STORAGE_READ.as_raw();
        const TEXTURE = vk::AccessFlags2::SHADER_SAMPLED_READ.as_raw();
        const INDIRECT = vk::AccessFlags2::INDIRECT_COMMAND_READ.as_raw();
        const UNIFORM = vk::AccessFlags2::UNIFORM_READ.as_raw();
    }
}

// non-associated consts look better in code

/// Invalidate any cache related to shader storage memory, in preparation for storage reads or writes.
pub const BARRIER_STORAGE: BarrierFlags = BarrierFlags::STORAGE;

/// Invalidate any cache related to texture memory, in preparation for texture reads.
pub const BARRIER_TEXTURE: BarrierFlags = BarrierFlags::TEXTURE;

/// Invalidate any cache related to indirect command data, in preparation for indirect draws or dispatches.
pub const BARRIER_INDIRECT: BarrierFlags = BarrierFlags::INDIRECT;

/// Invalidate any cache related to uniform buffer memory, in preparation for uniform buffer reads.
pub const BARRIER_UNIFORM: BarrierFlags = BarrierFlags::UNIFORM;

/// Describes root parameters for a command.
#[derive(Clone, Copy)]
pub enum PushDataSource<'a, T: Copy + 'static> {
    /// Pass a GPU pointer to root parameters in push data. The shader should expect the pointer at offset 0 in push data.
    Indirect(Ptr<T>),
    /// Pass a GPU pointer to root parameters in push data. The provided data is first uploaded to an internal GPU buffer. The shader should expect the pointer at offset 0 in push data.
    IndirectUpload(&'a T),
    /// Write push data directly, without an extra indirection. The shader should expect the root parameters directly in push data.
    Direct(&'a T),
}

impl<T: Copy + 'static> From<Ptr<T>> for PushDataSource<'_, T> {
    fn from(p: Ptr<T>) -> Self {
        PushDataSource::Indirect(p)
    }
}

impl<'a, T: Copy + 'static> From<&'a T> for PushDataSource<'a, T> {
    fn from(data: &'a T) -> Self {
        PushDataSource::IndirectUpload(data)
    }
}

#[derive(Clone, Copy)]
pub struct ImmediatePushData<'a, T: Copy + 'static>(pub &'a T);

impl<'a, T: Copy + 'static> From<ImmediatePushData<'a, T>> for PushDataSource<'a, T> {
    fn from(data: ImmediatePushData<'a, T>) -> Self {
        PushDataSource::Direct(data.0)
    }
}

/// Buffer into which GPU commands are recorded.
///
/// These should be submitted to the GPU using [`submit`].
pub struct CommandBuffer {
    // FIXME: the query pool should be created on-demand
    timestamp_query_pool: vk::QueryPool,
    timestamp_query_count: u32,
    timestamp_callbacks: Vec<Box<dyn FnOnce(u64) + Send>>,
    /// Current command buffer.
    cmdbuf: vk::CommandBuffer,
    submitted: bool,
    /// The index of the frame in which this command buffer was created (and should be recorded).
    frame_index_created: u64,
    // Make this type `!Send`, `!Sync`:
    // * `!Send` because it allocates from the thread-local command pool, and the command buffers must be released to the same command pool
    // * `!Sync` because the command methods allocate from the same thread-local command pool, which is not thread-safe
    _unsync_unsend: PhantomData<*const ()>,
}

impl CommandBuffer {
    /// Creates a command stream used to submit commands to the GPU.
    ///
    /// Once finished, the command stream should be submitted to the GPU using
    /// `CommandStream::flush`.
    /// They should be submitted in the same order as they were created.
    #[inline(never)]
    pub fn new() -> CommandBuffer {
        let device = Device::instance();
        let timestamp_query_pool = device.get_or_create_timestamp_query_pool();
        let frame_index_created = device.frame_index.load(Relaxed);
        trace!("GPU: create CommandBuffer, frame_index_created={}", frame_index_created);
        let cmdbuf = command_pool::allocate_command_buffer();
        unsafe {
            let info = vk::CommandBufferBeginInfo {
                flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                ..Default::default()
            };
            device.raw.begin_command_buffer(cmdbuf, &info).unwrap();
            // setup default dynamic state so validation layers don't complain
            device.raw.cmd_set_depth_bias_enable(cmdbuf, false);
            device.bind_descriptor_heaps(cmdbuf);
        }
        CommandBuffer {
            timestamp_query_pool,
            timestamp_query_count: 0,
            timestamp_callbacks: vec![],
            cmdbuf,
            submitted: false,
            frame_index_created,
            _unsync_unsend: PhantomData,
        }
    }

    /*
    /// Returns the current command buffer, creating a new one if necessary.
    ///
    /// The returned command buffer is ready to record commands.
    pub(crate) fn get_or_create_command_buffer(&mut self) -> vk::CommandBuffer {
        if let Some(cb) = self.cmdbuf {
            cb
        } else {
            let cb = self.create_command_buffer_raw();

            // setup default dynamic state so validation layers don't complain
            unsafe {
                let device = Device::instance();
            }

            self.cmdbuf = Some(cb);
            cb
        }
    }*/

    /*
    /// Closes the current command buffer.
    ///
    /// This does nothing if there is no current command buffer.
    pub(crate) fn close_command_buffer(&mut self) {
        Device::instance().raw().end_command_buffer(cb).unwrap();
        if let Some(cb) = self.cmdbuf.take() {
            unsafe {
            }
            self.command_buffers_to_submit.push(cb);
        }
    }*/

    /// Internal function to emit an image memory barrier.
    ///
    /// Mostly used for image layout transitions.
    pub(crate) unsafe fn image_barrier(&mut self, barrier: &vk::ImageMemoryBarrier2) {
        unsafe {
            Device::instance().raw.cmd_pipeline_barrier2(
                self.cmdbuf,
                &vk::DependencyInfo {
                    dependency_flags: Default::default(),
                    image_memory_barrier_count: 1,
                    p_image_memory_barriers: barrier,
                    ..Default::default()
                },
            );
        }
    }

    fn set_push_data<T: Data>(&mut self, cb: vk::CommandBuffer, params: PushDataSource<T>) {
        // None of the relevant drivers on desktop care about the actual stages,
        // only if it's graphics, compute, or ray tracing.
        let device = Device::instance();

        unsafe {
            let tmp;
            let address: *const c_void;
            let size: usize;
            match params {
                PushDataSource::Indirect(p) => {
                    // XXX: move p into a stable variable (tmp), don't make the address point
                    //      directly into `p` as it is a temporary.
                    //      Previously this was `address = &p.raw as *const _` and this crashed
                    //      in release because `params` became invalid after the match.
                    //      We could also match `params` by reference, but I find this cleaner.
                    tmp = p.raw;
                    address = &tmp as *const _ as *const c_void;
                    size = size_of::<DeviceAddress>();
                }
                PushDataSource::IndirectUpload(data) => {
                    tmp = gpu::alloc_temp_slice(slice::from_ref(data)).raw;
                    address = &tmp as *const _ as *const c_void;
                    size = size_of::<DeviceAddress>();
                }
                PushDataSource::Direct(data) => {
                    address = data as *const _ as *const c_void;
                    size = size_of::<T>();
                }
            };
            let push_data_info = VkPushDataInfoEXT {
                sType: VK_STRUCTURE_TYPE_PUSH_DATA_INFO_EXT,
                pNext: ptr::null(),
                offset: 0,
                data: VkHostAddressRangeConstEXT { address, size },
            };
            (device.ext.descriptor_heap.cmd_push_data)(cb.as_raw() as *mut _, &push_data_info);
        }
    }

    /// Emits a pipeline barrier.
    ///
    /// The barrier introduces an unconditional execution dependency between all previous
    /// and subsequent commands (equivalent to an ALL_COMMANDS -> ALL_COMMANDS execution dependency
    /// in Vulkan).
    ///
    /// The `flags` specify the memory access types that should be made available to subsequent commands.
    /// You can think of it as a list of caches that should be invalidated as a result
    /// of previous commands.
    ///
    /// The barrier makes all previous writes visible unconditionally (equivalent to
    /// src_access_mask = MEMORY_WRITE).
    pub fn barrier(&mut self, flags: BarrierFlags) {
        // This simplified barrier API just includes all previous stages and memory write types
        // in the source scope.
        // NVIDIA: stage execution dependencies seem to be ignored anyway.
        // AMD: this may affect performance, but I'm not sure.
        // TODO: add COLOR_ATTACHMENT & DEPTH_ATTACHMENT to InvalidateFlags
        let global_memory_barrier = vk::MemoryBarrier2 {
            src_access_mask: vk::AccessFlags2::MEMORY_WRITE,
            dst_access_mask: vk::AccessFlags2::from_raw(flags.bits())
                | vk::AccessFlags2::COLOR_ATTACHMENT_READ
                | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ,
            src_stage_mask: vk::PipelineStageFlags2::ALL_COMMANDS,
            dst_stage_mask: vk::PipelineStageFlags2::ALL_COMMANDS,
            ..Default::default()
        };
        unsafe {
            Device::instance().raw.cmd_pipeline_barrier2(
                self.cmdbuf,
                &vk::DependencyInfo {
                    dependency_flags: Default::default(),
                    memory_barrier_count: 1,
                    p_memory_barriers: &global_memory_barrier,
                    ..Default::default()
                },
            );
        }
    }

    /// Writes values to a buffer.
    ///
    /// # Safety
    ///
    /// TODO not sure why this was made unsafe?
    pub unsafe fn update_buffer(&mut self, buffer: &BufferUntyped, offset: usize, data: &[u8]) {
        unsafe {
            Device::instance().raw.cmd_update_buffer(self.cmdbuf, buffer.handle(), offset as u64, data);
        }
    }

    // SAFETY: TBD
    pub fn bind_compute_pipeline(&mut self, pipeline: &ComputePipeline) {
        unsafe {
            Device::instance().raw.cmd_bind_pipeline(self.cmdbuf, vk::PipelineBindPoint::COMPUTE, pipeline.pipeline);
        }
    }

    /// Dispatches compute work items.
    ///
    /// # Arguments
    ///
    /// * `group_count_x` - Number of workgroups to dispatch in the X dimension.
    /// * `group_count_y` - Number of workgroups to dispatch in the Y dimension.
    /// * `group_count_z` - Number of workgroups to dispatch in the Z dimension.
    /// * `root_params` - Root parameters to bind for the dispatch.
    ///
    pub fn dispatch<'params, T: Copy + 'static>(
        &mut self,
        group_count_x: u32,
        group_count_y: u32,
        group_count_z: u32,
        root_params: impl Into<PushDataSource<'params, T>>,
    ) {
        unsafe {
            self.set_push_data(self.cmdbuf, root_params.into());
            Device::instance().raw.cmd_dispatch(self.cmdbuf, group_count_x, group_count_y, group_count_z);
        }
    }

    /// TODO documentation
    pub fn push_debug_group(&mut self, label: &str) {
        unsafe {
            let label = CString::new(label).unwrap();
            Device::instance().ext.debug_utils.cmd_begin_debug_utils_label(
                self.cmdbuf,
                &vk::DebugUtilsLabelEXT {
                    p_label_name: label.as_ptr(),
                    color: [0.0, 0.0, 0.0, 0.0],
                    ..Default::default()
                },
            );
        }
    }

    /// TODO documentation
    pub fn pop_debug_group(&mut self) {
        // TODO check that push/pop calls are balanced
        unsafe {
            Device::instance().ext.debug_utils.cmd_end_debug_utils_label(self.cmdbuf);
        }
    }

    /// Surrounds a set of commands with a debug group.
    pub fn debug_group(&mut self, label: &str, f: impl FnOnce(&mut Self)) {
        self.push_debug_group(label);
        f(self);
        self.pop_debug_group();
    }

    /// TODO documentation
    pub fn write_timestamp(&mut self, callback: impl FnOnce(u64) + Send + 'static) {
        let cb = self.cmdbuf;
        let index = self.timestamp_query_count;
        assert!(index < MAX_TIMESTAMP_QUERY_COUNT, "maximum number of timestamp queries reached");
        self.timestamp_query_count += 1;
        self.timestamp_callbacks.push(Box::new(callback));

        unsafe {
            Device::instance().raw.cmd_write_timestamp2(
                cb,
                vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
                self.timestamp_query_pool,
                index,
            );
        }
    }

    /// TODO documentation
    pub fn upload_image_data(&mut self, image: ImageCopyView, size: Size3D, data: &[u8]) {
        let staging_buffer = Buffer::from_slice(data);

        self.copy_buffer_to_image(
            ImageCopyBuffer {
                buffer: staging_buffer.as_bytes(),
                layout: ImageDataLayout { offset: 0, texel_row_length: Some(size.width), row_count: Some(size.height) },
            },
            image,
            vk::Extent3D { width: size.width, height: size.height, depth: size.depth },
        );
    }

    /// TODO documentation
    pub fn create_image_with_data(&mut self, create_info: &ImageCreateInfo, aspect: ImageAspect, data: &[u8]) -> Image {
        let mut create_info_with_transfer_dst = create_info.clone();
        create_info_with_transfer_dst.usage |= ImageUsage::TRANSFER_DST;
        let image = Device::instance().create_image(&create_info_with_transfer_dst);
        self.upload_image_data(
            ImageCopyView { image: &image, mip_level: 0, origin: Offset3D::ZERO, aspect },
            Size3D { width: create_info.width, height: create_info.height, depth: create_info.depth },
            data,
        );
        image
    }

    /// TODO documentation
    pub fn blit_full_image_top_mip_level(&mut self, src: &Image, dst: &Image) {
        let width = src.width() as i32;
        let height = src.height() as i32;
        self.blit_image(
            &src,
            ImageSubresourceLayers { layer_count: 1, .. },
            Rect3D { min: Offset3D { x: 0, y: 0, z: 0 }, max: Offset3D { x: width, y: height, z: 1 } },
            &dst,
            ImageSubresourceLayers { layer_count: 1, .. },
            Rect3D { min: Offset3D { x: 0, y: 0, z: 0 }, max: Offset3D { x: width, y: height, z: 1 } },
            vk::Filter::NEAREST,
        );
    }
}

#[cold]
fn panic_command_buffer_not_submitted() {
    panic!("CommandBuffer was not submitted before being dropped");
}

impl Drop for CommandBuffer {
    fn drop(&mut self) {
        if !self.submitted {
            panic_command_buffer_not_submitted();
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Specifies a timeline semaphore wait operation.
#[derive(Copy, Clone)]
pub struct SyncWait {
    pub semaphore: vk::Semaphore,
    pub value: u64,
}

/// Specifies a timeline semaphore signal operation.
#[derive(Copy, Clone)]
pub struct SyncSignal {
    pub semaphore: vk::Semaphore,
    pub value: u64,
}

/// Maximum number of timeline semaphore wait or timeline semaphore signal operations that can be submitted in a single call to `sync`.
pub const MAX_SYNC_COUNT: usize = 4;

/// Executes semaphore wait and signal operations on the device.
///
/// # Arguments
/// * `waits` - semaphores to wait for
/// * `signals` - semaphores to signal, once the waits have completed
///
/// # Notes
///
/// The number of waits passed to this function
/// must not exceed `MAX_SYNC_COUNT`. Same with signals.
fn sync(waits: &[SyncWait], signals: &[SyncSignal]) {
    let device = Device::instance();

    // /!\ Lock the device for command submission.
    let submission_state = device.submission_state.lock().unwrap();

    let mut wait_semaphores = [vk::Semaphore::null(); MAX_SYNC_COUNT];
    let mut signal_semaphores = [vk::Semaphore::null(); MAX_SYNC_COUNT];
    let mut wait_semaphore_values = [0u64; MAX_SYNC_COUNT];
    let mut signal_semaphore_values = [0u64; MAX_SYNC_COUNT];
    let mut wait_semaphore_dst_stages = [vk::PipelineStageFlags::empty(); MAX_SYNC_COUNT];

    for (i, signal) in signals.iter().enumerate() {
        signal_semaphores[i] = signal.semaphore;
        signal_semaphore_values[i] = signal.value;
    }

    for (i, w) in waits.iter().enumerate() {
        wait_semaphore_dst_stages[i] = vk::PipelineStageFlags::ALL_COMMANDS;
        wait_semaphores[i] = w.semaphore;
        wait_semaphore_values[i] = w.value;
    }

    let timeline_submit_info = vk::TimelineSemaphoreSubmitInfo {
        wait_semaphore_value_count: waits.len() as u32,
        p_wait_semaphore_values: wait_semaphore_values.as_ptr(),
        signal_semaphore_value_count: signals.len() as u32,
        p_signal_semaphore_values: signal_semaphore_values.as_ptr(),
        ..Default::default()
    };

    let submit_info = vk::SubmitInfo {
        p_next: &timeline_submit_info as *const _ as *const c_void,
        wait_semaphore_count: waits.len() as u32,
        p_wait_semaphores: wait_semaphores.as_ptr(),
        p_wait_dst_stage_mask: wait_semaphore_dst_stages.as_ptr(),
        command_buffer_count: 0,
        p_command_buffers: ptr::null(),
        signal_semaphore_count: signals.len() as u32,
        p_signal_semaphores: signal_semaphores.as_ptr(),
        ..Default::default()
    };

    unsafe {
        trace!("GPU: QueueSubmit (synchronization)");
        match device.raw.queue_submit(submission_state.queue, &[submit_info], vk::Fence::null()) {
            Ok(()) => {}
            Err(e) => {
                error!("QueueSubmit (synchronization) failed: {:?}", e);
            }
        }
    }
}

/// Waits on the given timeline semaphore until it reaches the given value.
pub fn wait(semaphore: vk::Semaphore, value: u64) {
    sync(&[SyncWait { semaphore, value }], &[]);
}

/// Signals the given timeline semaphore with the given value.
///
/// The value is ignored if `semaphore` is a binary semaphore.
pub fn signal(semaphore: vk::Semaphore, value: u64) {
    sync(&[], &[SyncSignal { semaphore, value }]);
}

/// Submits the given commands for execution on the GPU.
///
/// This implicitly calls [`flush`](flush) to ensure that all pending commands on the default
/// command buffer are submitted before this command buffer.
#[inline(never)]
pub fn submit(mut cmd: CommandBuffer) -> VkResult<()> {
    // Submit the default command buffer.
    flush()?;

    let device = Device::instance();

    //----------------------
    // /!\ Lock the device for command submission.
    // This effectively synchronizes submissions on the device.
    //----------------------
    let mut submission_state = device.submission_state.lock().unwrap();

    // Verify that the command streams are submitted in the order in which they were created.
    // Timeline semaphore values depend on this.
    assert!(!cmd.submitted);

    let frame_index_submitted = device.frame_index.load(Relaxed);
    assert_eq!(
        cmd.frame_index_created, frame_index_submitted,
        "a command buffer was submitted in a different frame than the one it was created in"
    );

    trace!(
        "GPU: submit CommandStream, frame_index_created={}, frame_index_submitted={}",
        cmd.frame_index_created, frame_index_submitted
    );

    // flush pending writes
    cmd.barrier(BarrierFlags::empty());

    // finish recording the command buffer
    unsafe {
        device.raw.end_command_buffer(cmd.cmdbuf).unwrap();
    }

    // Put the command buffer up for deletion
    command_pool::defer_free_command_buffer(cmd.cmdbuf, frame_index_submitted);

    //----------------------
    // submit
    let signal_semaphores = vec![];
    let signal_semaphore_values = vec![];
    let timeline_submit_info = vk::TimelineSemaphoreSubmitInfo {
        signal_semaphore_value_count: signal_semaphore_values.len() as u32,
        p_signal_semaphore_values: signal_semaphore_values.as_ptr(),
        ..Default::default()
    };
    let submit_info = vk::SubmitInfo {
        p_next: &timeline_submit_info as *const _ as *const c_void,
        command_buffer_count: 1,
        p_command_buffers: &cmd.cmdbuf,
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
        result = device.raw.queue_submit(submission_state.queue, &[submit_info], vk::Fence::null());

        submission_state.active_submissions.push_back(ActiveSubmission {
            frame_index: frame_index_submitted,
            timestamp_query_pool: cmd.timestamp_query_pool,
            timestamp_query_count: cmd.timestamp_query_count,
            timestamp_callbacks: mem::take(&mut cmd.timestamp_callbacks),
        });
    };

    cmd.submitted = true;
    result
}

/// Presents the given swap chain image to the screen.
pub fn present(swap_chain: &mut SwapChain, index: usize) -> VkResult<()> {
    let image = &swap_chain.images[index];

    // Automatically flush the default command buffer before presenting.
    flush()?;

    // transition image to PRESENT_SRC
    let mut cmd = CommandBuffer::new();
    unsafe {
        cmd.image_barrier(&vk::ImageMemoryBarrier2 {
            src_stage_mask: vk::PipelineStageFlags2::ALL_COMMANDS,
            src_access_mask: vk::AccessFlags2::MEMORY_WRITE,
            dst_stage_mask: vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
            dst_access_mask: vk::AccessFlags2::NONE,
            old_layout: vk::ImageLayout::GENERAL,
            new_layout: vk::ImageLayout::PRESENT_SRC_KHR,
            image: image.image.handle,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: vk::REMAINING_MIP_LEVELS,
                base_array_layer: 0,
                layer_count: vk::REMAINING_ARRAY_LAYERS,
            },
            ..Default::default()
        });
    }
    submit(cmd)?;

    // NOTE: submission state is unlocked here, so it's possible that another thread submits
    //       commands to the image that was transitioned to PRESENT layout before we present it.
    //       This is up to the caller to avoid doing that.

    let device = Device::instance();
    signal(image.render_finished, 0);
    unsafe {
        let submission_state = device.submission_state.lock().unwrap();
        let present_info = vk::PresentInfoKHR {
            wait_semaphore_count: 1,
            p_wait_semaphores: &image.render_finished,
            swapchain_count: 1,
            p_swapchains: &swap_chain.handle,
            p_image_indices: &[index as u32] as *const u32,
            p_results: ptr::null_mut(),
            ..Default::default()
        };
        device.ext.swapchain.queue_present(submission_state.queue, &present_info).map(|_| ())
    }
}

// TODO is this safe with multiple threads?
thread_local! {
    static DEFAULT_COMMAND_BUFFER: RefCell<Option<CommandBuffer>> = const { RefCell::new(None) };
}

#[inline]
pub fn with_cmdbuf<R>(f: impl FnOnce(&mut CommandBuffer) -> R) -> R {
    // This has better codegen than calling `f` inside `with`.
    let cmdbuf = DEFAULT_COMMAND_BUFFER.with(|cmdbuf| cmdbuf as *const RefCell<Option<CommandBuffer>>);
    // SAFETY: `cmdbuf` has 'static lifetime (at least within this thread), and we keep it in this thread.
    unsafe {
        let mut cb = (*cmdbuf).borrow_mut();
        let cb = cb.get_or_insert_with(|| CommandBuffer::new());
        f(cb)
    }
}

pub(crate) fn take_cmdbuf() -> Option<CommandBuffer> {
    DEFAULT_COMMAND_BUFFER.take()
}

/// Dispatches compute work items.
///
/// This uses the default command buffer.
///
/// # Arguments
///
/// * `group_count_x` - Number of workgroups to dispatch in the X dimension.
/// * `group_count_y` - Number of workgroups to dispatch in the Y dimension.
/// * `group_count_z` - Number of workgroups to dispatch in the Z dimension.
/// * `root_params` - Root parameters to bind for the dispatch.
pub fn dispatch<'params, T: Copy + 'static>(
    pipeline: &ComputePipeline,
    group_count_x: u32,
    group_count_y: u32,
    group_count_z: u32,
    root_params: impl Into<PushDataSource<'params, T>>,
) {
    with_cmdbuf(|cb| {
        cb.bind_compute_pipeline(pipeline);
        cb.dispatch(group_count_x, group_count_y, group_count_z, root_params);
    });
}

/// Encodes a render pass on the default command buffer.
pub fn render(
    color_attachments: &[ColorAttachment],
    depth_stencil_attachment: Option<DepthStencilAttachment>,
    encoder_fn: impl FnOnce(&mut RenderEncoder),
) {
    with_cmdbuf(|cb| {
        let mut encoder = cb.begin_rendering(color_attachments, depth_stencil_attachment);
        encoder_fn(&mut encoder);
        encoder.finish();
    });
}

/// Starts a debug group on the default command buffer.
#[inline(never)]
pub fn push_debug_group(label: &str) {
    with_cmdbuf(|cb| {
        cb.push_debug_group(label);
    });
}

/// Ends a debug group on the default command buffer.
#[inline(never)]
pub fn pop_debug_group() {
    with_cmdbuf(|cb| {
        cb.pop_debug_group();
    });
}

/// Defines a barrier ordering memory transactions.
///
/// `barrier` ensures that memory writes issued prior to this barrier
/// are visible to operations of the kind specified in `flags` after this barrier.
///
/// # Implementation details
///
/// This calls `vkCmdPipelineBarrier`, with an `ALL_COMMANDS -> ALL_COMMANDS` execution dependency
/// and a single memory barrier.
/// All previous writes are made visible unconditionally (i.e. `srcAccessMask = MEMORY_WRITE`).
/// `flags` define which memory access types are concerned for subsequent command (it determines `dstAccessMask`).
#[inline(never)]
pub fn barrier(flags: BarrierFlags) {
    with_cmdbuf(|cb| {
        cb.barrier(flags);
    });
}

#[inline(never)]
pub fn update_buffer(buffer: &BufferUntyped, offset: usize, data: &[u8]) {
    with_cmdbuf(|cb| {
        unsafe {
            // TODO figure out why we needed unsafe here
            cb.update_buffer(buffer, offset, data);
        }
    });
}

#[inline(never)]
pub fn upload_image_data(image: ImageCopyView, size: Size3D, data: &[u8]) {
    with_cmdbuf(|cb| {
        cb.upload_image_data(image, size, data);
    });
}

#[inline(never)]
pub fn create_image_with_data(create_info: &ImageCreateInfo, aspect: ImageAspect, data: &[u8]) -> Image {
    with_cmdbuf(|cb| cb.create_image_with_data(create_info, aspect, data))
}

#[inline(never)]
pub fn blit_full_image_top_mip_level(src: &Image, dst: &Image) {
    with_cmdbuf(|cb| {
        cb.blit_full_image_top_mip_level(src, dst);
    });
}

#[inline(never)]
pub fn fill_buffer(range: &BufferRangeUntyped, data: u32) {
    with_cmdbuf(|cb| {
        cb.fill_buffer(range, data);
    });
}

#[inline(never)]
pub fn clear_image(image: &Image, clear_color_value: ClearColorValue) {
    with_cmdbuf(|cb| {
        cb.clear_image(image, clear_color_value);
    });
}

#[inline(never)]
pub fn clear_depth_image(image: &Image, depth: f32) {
    with_cmdbuf(|cb| {
        cb.clear_depth_image(image, depth);
    });
}

#[inline(never)]
pub fn copy_image_to_image(source: ImageCopyView<'_>, destination: ImageCopyView<'_>, copy_size: vk::Extent3D) {
    with_cmdbuf(|cb| {
        cb.copy_image_to_image(source, destination, copy_size);
    });
}

/// Copies data from one buffer to another.
#[inline(never)]
pub fn copy_buffer(source: &BufferUntyped, src_offset: u64, destination: &BufferUntyped, dst_offset: u64, size: u64) {
    with_cmdbuf(|cb| {
        cb.copy_buffer(source, src_offset, destination, dst_offset, size);
    });
}

/// Copies data from a buffer to an image.
///
/// TODO copy to layer other than 0
#[inline(never)]
pub fn copy_buffer_to_image(source: ImageCopyBuffer<'_>, destination: ImageCopyView<'_>, copy_size: vk::Extent3D) {
    with_cmdbuf(|cb| {
        cb.copy_buffer_to_image(source, destination, copy_size);
    });
}

/// Copies data from an image to a buffer.
#[inline(never)]
pub fn copy_image_to_buffer(source: ImageCopyView<'_>, destination: ImageCopyBuffer<'_>, copy_size: Size3D) {
    with_cmdbuf(|cb| {
        cb.copy_image_to_buffer(source, destination, copy_size);
    });
}

#[inline(never)]
pub fn blit_image(
    src: &Image,
    src_subresource: ImageSubresourceLayers,
    src_region: Rect3D,
    dst: &Image,
    dst_subresource: ImageSubresourceLayers,
    dst_region: Rect3D,
    filter: vk::Filter,
) {
    with_cmdbuf(|cb| {
        cb.blit_image(src, src_subresource, src_region, dst, dst_subresource, dst_region, filter);
    });
}

/// Submits commands in the default command buffer for execution on the GPU.
#[inline(never)]
pub fn flush() -> VkResult<()> {
    if let Some(cb) = take_cmdbuf() {
        submit(cb)
    } else {
        // Nothing to submit.
        Ok(())
    }
}

/// Writes a timestamp.
#[inline(never)]
pub fn write_timestamp(callback: impl FnOnce(u64) + Send + 'static) {
    with_cmdbuf(|cb| cb.write_timestamp(callback))
}
