//! Manages command watches & data captures.
use crate::descriptor::ImageViewInfo;
use crate::event::EId;
use crate::helper::{Buffer, HOST_VISIBLE_CACHED_BITS, Image};
use crate::load_chain::{LoadChain, LoadChainBase};
use crate::wrappers::command::Command;
use crate::wrappers::image::ImageInfo;
use crate::{Device, DeviceAddressRangeSet, QueueSubmit};
use slotmap::{SlotMap, new_key_type};
use std::hash::Hash;
use std::{mem, ptr, slice};
use vulkan::*;

/// Debugger-related state.
///
/// Holds the list of watches.
pub struct DebuggerState {
    pub(crate) watches: Vec<CommandWatch>,
}

impl DebuggerState {
    pub(crate) fn new() -> DebuggerState {
        DebuggerState { watches: vec![] }
    }

    /// Finds an existing watch by key.
    pub(crate) fn get_or_insert_watch(&mut self, eid: EId) -> &mut CommandWatch {
        for watch in self.watches.iter_mut() {
            if watch.eid == eid {
                return watch;
            }
        }
        self.watches.push(CommandWatch {
            eid,
            stale: true,
            captures: SlotMap::with_key(),
        });
        self.watches.last_mut().unwrap()
    }

    /// Adds a debugger watch.
    fn add_capture(&mut self, eid: EId, query: CaptureTarget) -> Option<&Capture> {
        let mut watch = self.get_or_insert_watch(eid);
        if watch.stale {
            return None;
        }
        for lc in watch.captures.values_mut() {
            if lc.target == query {
                lc.abandoned = false;
                return Some(lc);
            }
        }
        let id = watch.captures.insert(Capture {
            hash: 0, // TODO compute hash
            target: query,
            result: None,
            transient: true,
            abandoned: false,
            stale: true,
        });
        Some(&watch.captures[id])
    }

    // TODO: this should return a reference.
    fn capture_command_data(&mut self, eid: EId, query: CaptureTarget) -> Option<Vec<u8>> {
        if let Some(lc) = self.add_capture(eid, query)
            && let Some(ref result_buffer) = lc.result
        {
            match result_buffer {
                CaptureResult::Buffer(buf) => {
                    let result_slice = unsafe { std::slice::from_raw_parts(buf.buffer.ptr as *const u8, buf.captured_byte_size) };
                    Some(result_slice.to_vec())
                }
                CaptureResult::Image(img) => {
                    todo!("image capture not implemented");
                    //let result_slice = unsafe { std::slice::from_raw_parts(img.image.ptr as *const u8, image_buffer_size(&img.info)) };
                    //return Some(result_slice.to_vec());
                }
            }
        } else {
            // no data captured this frame
            None
        }
    }

    /// Captures data from a pointer indirection chain that starts in push data.
    pub fn capture_command_push_data(&mut self, eid: EId, load_chain: LoadChain, size: usize) -> Option<Vec<u8>> {
        self.capture_command_data(eid, CaptureTarget::PushData { load_chain, byte_size: Some(size) })
    }

    /// Captures data from a pointer indirection chain that starts in global memory.
    pub fn capture_command_mem_data(&mut self, eid: EId, load_chain: LoadChain, size: usize) -> Option<Vec<u8>> {
        self.capture_command_data(eid, CaptureTarget::GlobalMemory { load_chain, byte_size: size })
    }

    pub fn capture_resource_descriptor_heap(&mut self, eid: EId) -> Option<Vec<u8>> {
        self.capture_command_data(eid, CaptureTarget::ResourceDescriptorHeap { size: None })
    }

    pub fn capture_sampler_descriptor_heap(&mut self, eid: EId) -> Option<Vec<u8>> {
        self.capture_command_data(eid, CaptureTarget::SamplerDescriptorHeap { size: None })
    }

    pub fn capture_image(
        &mut self,
        eid: EId,
        descriptor_load_chain: LoadChain,
        image_info: ImageViewInfo,
    ) -> Option<CapturedImage> {
        todo!()
    }
}

// on NV:
// - bufferDescriptorSize=16
// - imageDescriptorSize=32
// - samplerDescriptorSize=32
// So set the max size to 64 to have some margin
pub const DESCRIPTOR_BLOB_MAX_SIZE: usize = 64;
pub type DescriptorBlob = [u8; DESCRIPTOR_BLOB_MAX_SIZE];

pub enum ResourceDescriptorInfo {
    Image { view: VkImageViewCreateInfo, layout: VkImageLayout },
    TexelBuffer(VkTexelBufferDescriptorInfoEXT),
    AddressRange(VkDeviceAddressRangeEXT),
    TensorARM(VkTensorViewCreateInfoARM),
}

impl Device {
    unsafe fn add_submission(&self, mut commands: Vec<Command>) {
        let mut submissions = self.submissions.lock();
        // store submission & command index in each Command
        let submission_index = submissions.len();
        for (i, cmd) in commands.iter_mut().enumerate() {
            cmd.idx.sub = submission_index as u32;
            cmd.idx.cmd = i as u32;
        }
        submissions.push(QueueSubmit { commands });
    }

    pub unsafe fn hook_queue_submit(
        &self,
        queue: VkQueue,
        submit_count: u32,
        p_submits: *const VkSubmitInfo,
        fence: VkFence,
    ) -> VkResult {
        let submit_infos = slice::from_raw_parts(p_submits, submit_count as usize);
        for s in submit_infos {
            if s.commandBufferCount == 0 {
                continue;
            }
            let mut commands = vec![];
            for cmdbuf in slice::from_raw_parts(s.pCommandBuffers, s.commandBufferCount as usize) {
                let pdata = self.get_private_data_mut(*cmdbuf).unwrap();
                commands.extend(mem::take(&mut pdata.commands));
            }
            self.add_submission(commands);
        }
        self.QueueSubmit(queue, submit_count, p_submits, fence)
    }

    pub unsafe fn hook_queue_submit_2(
        &self,
        queue: VkQueue,
        submit_count: u32,
        p_submits: *const VkSubmitInfo2,
        fence: VkFence,
    ) -> VkResult {
        let submit_infos = slice::from_raw_parts(p_submits, submit_count as usize);
        for s in submit_infos {
            if s.commandBufferInfoCount == 0 {
                continue;
            }
            let mut commands = vec![];
            for cbinfo in slice::from_raw_parts(s.pCommandBufferInfos, s.commandBufferInfoCount as usize) {
                let pdata = self.get_private_data_mut(cbinfo.commandBuffer).unwrap();
                commands.extend(mem::take(&mut pdata.commands));
            }
            self.add_submission(commands);
        }
        self.QueueSubmit2(queue, submit_count, p_submits, fence)
    }

    /// Returns the size of a descriptor of the specified type,
    /// as written by `vkWriteResourceDescriptorsEXT` or `vkWriteSamplerDescriptorsEXT`.
    pub fn descriptor_blob_size(&self, ty: VkDescriptorType) -> usize {
        match ty {
            VK_DESCRIPTOR_TYPE_SAMPLER => self.descriptor_heap_properties.samplerDescriptorSize as usize,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
            | VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE
            | VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
            | VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER
            | VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER => self.descriptor_heap_properties.imageDescriptorSize as usize,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
            | VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
            | VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC
            | VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC => {
                self.descriptor_heap_properties.bufferDescriptorSize as usize
            }
            _ => panic!("unsupported descriptor type: {}", ty),
        }
    }

    /// Records the mapping between VkResourceDescriptorInfoEXT and the corresponding descriptor blob written by `vkWriteResourceDescriptorsEXT`.
    pub unsafe fn record_resource_descriptors(
        &self,
        count: u32,
        resources: *const VkResourceDescriptorInfoEXT,
        descriptors: *const VkHostAddressRangeEXT,
    ) {
        let mut ds = self.descriptor_state.lock();
        let resources = slice::from_raw_parts(resources, count as usize);
        let descriptors = slice::from_raw_parts(descriptors, count as usize);
        for (resource, descriptor) in resources.iter().zip(descriptors.iter()) {
            let size = self.descriptor_blob_size(resource.r#type);
            let mut blob = [0u8; DESCRIPTOR_BLOB_MAX_SIZE];
            ptr::copy_nonoverlapping(descriptor.address as *const u8, blob.as_mut_ptr(), size);
            let resource = match resource.r#type {
                VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE | VK_DESCRIPTOR_TYPE_STORAGE_IMAGE => ResourceDescriptorInfo::Image {
                    view: *(*resource.data.pImage).pView,
                    layout: (*resource.data.pImage).layout,
                },
                VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER | VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER => {
                    ResourceDescriptorInfo::TexelBuffer(*resource.data.pTexelBuffer)
                }
                VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER | VK_DESCRIPTOR_TYPE_STORAGE_BUFFER => {
                    ResourceDescriptorInfo::AddressRange(*resource.data.pAddressRange)
                }
                VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR => {
                    ResourceDescriptorInfo::AddressRange(*resource.data.pAddressRange)
                }
                VK_DESCRIPTOR_TYPE_TENSOR_ARM => ResourceDescriptorInfo::TensorARM(*resource.data.pTensorARM),
                _ => {
                    eprintln!("write_resource_descriptors: unsupported resource type {}", resource.r#type);
                    continue;
                }
            };
            ds.resource_descriptors.insert(blob, resource);
        }
    }

    /// Records the mapping between VkSamplerCreateInfo and the corresponding descriptor blob written by `vkWriteResourceDescriptorsEXT`.
    pub unsafe fn record_sampler_descriptors(
        &self,
        count: u32,
        samplers: *const VkSamplerCreateInfo,
        descriptors: *const VkHostAddressRangeEXT,
    ) {
        let mut ds = self.descriptor_state.lock();
        let samplers = slice::from_raw_parts(samplers, count as usize);
        let descriptors = slice::from_raw_parts(descriptors, count as usize);
        for (sampler, descriptor) in samplers.iter().zip(descriptors.iter()) {
            let size = self.descriptor_blob_size(VK_DESCRIPTOR_TYPE_SAMPLER);
            let mut blob = [0u8; DESCRIPTOR_BLOB_MAX_SIZE];
            ptr::copy_nonoverlapping(descriptor.address as *const u8, blob.as_mut_ptr(), size);
            ds.sampler_descriptors.insert(blob, *sampler);
        }
    }
}

impl Device {
    pub fn update_watches(&self) {
        //let mut submissions = self.submissions.lock();
        let mut dbg = self.dbg.lock();
        // Iterate over all watches, and clean up any abandoned captures
        // (those which were not requested during the last frame).
        // At the same time, reset the `abandoned` flags to true for the coming frame.
        for w in dbg.watches.iter_mut() {
            w.captures.retain(|_id, cap| {
                if cap.transient && cap.abandoned {
                    self.destroy_capture(cap);
                    false
                } else {
                    cap.abandoned = true;
                    true
                }
            });
            // stale for the coming frame
            w.stale = true;
        }
    }

    fn destroy_capture(&self, capture: &mut Capture) {
        if let Some(result) = capture.result.take() {
            match result {
                CaptureResult::Buffer(buf) => unsafe {
                    self.destroy_buffer_helper(buf.buffer);
                },
                CaptureResult::Image(img) => unsafe {
                    self.destroy_image_helper(img.image);
                },
            }
        }
    }

    unsafe fn update_watches_after_command(&self, cmd_buf: VkCommandBuffer, command: &Command) {
        let mut dbg = self.dbg.lock();
        for w in dbg.watches.iter_mut() {
            if w.eid == command.eid {
                w.stale = false;
                for lc in w.captures.values_mut() {
                    self.do_capture_command_data(cmd_buf, command, lc);
                }
            }
        }
    }

    unsafe fn allocate_capture_result_buffer<'a>(&self, byte_size: usize, capture: &'a mut Capture) -> &'a Buffer {
        let result_buffer = match capture.result {
            Some(CaptureResult::Buffer(ref mut result_buffer)) => {
                if result_buffer.captured_byte_size >= byte_size {
                    // existing buffer is big enough
                    &result_buffer.buffer
                } else {
                    // reallocate buffer
                    self.destroy_buffer_helper(mem::take(&mut result_buffer.buffer));
                    result_buffer.buffer = self.create_buffer_helper(
                        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                        HOST_VISIBLE_CACHED_BITS,
                        byte_size,
                        None,
                    );
                    result_buffer.captured_byte_size = byte_size;
                    &result_buffer.buffer
                }
            }
            None => {
                let buffer = self.create_buffer_helper(
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                    HOST_VISIBLE_CACHED_BITS,
                    byte_size,
                    None,
                );
                capture.result =
                    Some(CaptureResult::Buffer(CaptureResultBuffer { buffer, captured_byte_size: byte_size }));
                &capture.result.as_ref().unwrap().as_buffer().unwrap().buffer
            }
            _ => {
                panic!("capture result is an image, but expected a buffer");
            }
        };
        result_buffer
    }

    unsafe fn copy_command_memory<'a>(
        &self,
        cmd_buf: VkCommandBuffer,
        base: LoadChainBase,
        chain: &LoadChain,
        byte_size: usize,
        capture: &'a mut Capture,
    ) -> Option<&'a Buffer> {
        if byte_size == 0 {
            return None;
        }
        let result_buffer = self.allocate_capture_result_buffer(byte_size, capture);
        self.read_load_chain(cmd_buf, base, chain, byte_size, result_buffer);
        Some(result_buffer)
    }

    unsafe fn copy_command_image_to_memory<'a>(
        &self,
        command: &Command,
        cmd_buf: VkCommandBuffer,
        base: LoadChainBase,
        descriptor_chain: &LoadChain,
        image_info: &ImageViewInfo,
        capture: &'a mut Capture,
    ) -> Option<&'a Buffer> {
        let byte_size = image_buffer_size(image_info);
        if byte_size == 0 {
            return None;
        }
        let result_buffer = self.allocate_capture_result_buffer(byte_size, capture);
        self.copy_image_indirect(cmd_buf, base, descriptor_chain, image_info, &command.resource_heap, result_buffer);
        Some(result_buffer)
    }

    unsafe fn do_capture_command_data(&self, cmd_buf: VkCommandBuffer, command: &Command, capture: &mut Capture) {
        match capture.target {
            CaptureTarget::GlobalMemory { load_chain, byte_size } => {
                self.copy_command_memory(cmd_buf, LoadChainBase::DeviceAddress(0), &load_chain, byte_size, capture);
            }
            CaptureTarget::PushData { load_chain, byte_size } => {
                let byte_size = byte_size.unwrap_or(command.push_data.len());
                self.copy_command_memory(
                    cmd_buf,
                    LoadChainBase::Slice(&command.push_data),
                    &load_chain,
                    byte_size,
                    capture,
                );
            }
            CaptureTarget::ResourceDescriptorHeap { size } => {
                let byte_size = size.unwrap_or(command.resource_heap.heapRange.size as usize);
                self.copy_command_memory(
                    cmd_buf,
                    LoadChainBase::DeviceAddress(command.resource_heap.heapRange.address),
                    &LoadChain::new(),
                    byte_size,
                    capture,
                );
            }
            CaptureTarget::SamplerDescriptorHeap { size } => {
                let size = size.unwrap_or(command.sampler_heap.heapRange.size as usize);
                if size == 0 {
                    return;
                }
                self.copy_command_memory(
                    cmd_buf,
                    LoadChainBase::DeviceAddress(command.sampler_heap.heapRange.address),
                    &LoadChain::new(),
                    size,
                    capture,
                );
            }
            CaptureTarget::Image { descriptor, info } => {
                let pixel_size = image_buffer_size(&info);
                if pixel_size == 0 {
                    return;
                }
                self.copy_command_memory(cmd_buf, LoadChainBase::DeviceAddress(0), &descriptor, pixel_size, capture);
            }
        };
    }
}

/// A command for which we have capture requests.
pub struct CommandWatch {
    pub eid: EId,
    /// If true, no data was captured for this command in the last frame.
    pub stale: bool,
    pub captures: SlotMap<CaptureId, Capture>,
}

/// Represents data to capture.
#[derive(Clone, Eq, PartialEq, Hash)]
enum CaptureTarget {
    /// Capture memory contents reachable from push data.
    PushData {
        /// Pointer chain to the data.
        load_chain: LoadChain,
        /// Number of bytes to capture. If none, captures the whole push data buffer.
        byte_size: Option<usize>,
    },
    /// Capture an image.
    Image {
        /// Access chain to the image descriptor.
        descriptor: LoadChain,
        /// Information about the image to capture.
        info: ImageViewInfo,
    },
    /// Capture contents of global memory.
    GlobalMemory {
        /// Pointer chain to the data.
        load_chain: LoadChain,
        /// Number of bytes to capture.
        byte_size: usize,
    },
    /// Capture contents of the resource descriptor heap bound for the command.
    ResourceDescriptorHeap { size: Option<usize> },
    /// Capture contents of the sampler descriptor heap bound for the command.
    SamplerDescriptorHeap { size: Option<usize> },
}

pub struct CaptureResultBuffer {
    pub buffer: Buffer,
    pub captured_byte_size: usize,
}

pub struct CaptureResultImage {
    pub image: Image,
    pub info: ImageViewInfo,
}

/// Holds the result of a command data capture.
pub enum CaptureResult {
    Buffer(CaptureResultBuffer),
    Image(CaptureResultImage),
}

impl CaptureResult {
    pub fn as_buffer(&self) -> Option<&CaptureResultBuffer> {
        match self {
            CaptureResult::Buffer(buffer) => Some(buffer),
            _ => None,
        }
    }

    pub fn as_image(&self) -> Option<&Image> {
        match self {
            CaptureResult::Image(image) => Some(image),
            _ => None,
        }
    }
}

pub struct Capture {
    pub(crate) hash: u64, // Unique hash
    pub(crate) target: CaptureTarget,
    pub(crate) result: Option<CaptureResult>, // Result buffer for holding the result of the query
    pub(crate) transient: bool,               // Whether this watch is temporary (removed if not read in the last frame)
    pub(crate) abandoned: bool,               // If true at the end of the frame, means that the watch wasn't requested
    pub(crate) stale: bool,                   // If true, no data was captured for this watch in the last frame
}

new_key_type! {
    pub struct CaptureId;
}

impl Device {
    /// Called after every command to update relevant watches.
    pub fn after_command(&self, command: &Command) {
        unsafe { self.update_watches_after_command(command.cmd_buf, command) }
    }
}

//--------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------
// helpers

fn image_buffer_size(image_info: &ImageViewInfo) -> usize {
    let format_info = vk_format_info::get_format_info(image_info.format);
    let pixel_size = image_info.width as usize
        * image_info.height as usize
        * image_info.depth as usize
        * format_info.block_size as usize;
    pixel_size
}
