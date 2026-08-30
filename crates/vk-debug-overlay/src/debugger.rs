use crate::Device;
use crate::event::EId;
use crate::format::format_info;
use crate::helper::{Buffer, DeviceHelper, Image, Pipeline, include_bytes_as_u32};
use crate::state_tracker::command::Command;
use crate::state_tracker::image::ImageInfo;
use ash::vk;
use core::fmt;
use rustc_hash::FxHasher;
use slotmap::{SlotMap, new_key_type};
use std::hash::{Hash, Hasher};
use std::ptr;

/// Represents a sequence of pointer indirections from a base address, (e.g. `base->field->field2 ...`).
#[derive(Clone, Hash, Eq, PartialEq)]
pub struct LoadChain {
    // Chain of offsets for each pointer indirection.
    //
    // This establishes a series of addresses (denoted `address[i]`), defined by the following
    // recurrence relation:
    //
    // - `address[0] = <base> + offsets[0]`
    // - `base[N] = *address[N-1]`
    // - `address[N] = base[N] + offsets[N]`
    pub offsets: Vec<usize>,
}

impl LoadChain {
    pub fn new() -> LoadChain {
        LoadChain { offsets: vec![0] }
    }

    /// Pushes a new indirection on the load chain.
    ///
    /// Concretely, if this load chain represents some address `ADDR`,
    /// then after this function it will point to the address at `*(ADDR + offset)`
    pub fn deref_at(&mut self, offset: usize) {
        let len = self.offsets.len() - 1;
        self.offsets[len] = offset;
        self.offsets.push(0);
    }

    pub fn with_deref(&self, offset: usize) -> LoadChain {
        let mut c = self.clone();
        c.deref_at(offset);
        c
    }
}

impl Default for LoadChain {
    fn default() -> Self {
        LoadChain::new()
    }
}

impl fmt::Debug for LoadChain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[base")?;
        for (i, offset) in self.offsets.iter().enumerate() {
            if i > 0 {
                write!(f, ".0x{:x}", offset)?;
            } else {
                write!(f, "+0x{:x}", offset)?;
            }
        }
        write!(f, "]")?;
        Ok(())
    }
}

//--------------------------------------------------------------------------------------------------

pub struct DebuggerResources {
    copy_indirect_1d: Pipeline,
}

impl DebuggerResources {
    pub unsafe fn new(device_helper: &DeviceHelper) -> DebuggerResources {
        let copy_indirect_1d = device_helper.create_compute_pipeline_helper(
            COPY_1D_SHADER,
            c"copy_indirect_1d",
            &[],
            size_of::<CopyIndirect1DParams>(),
        );
        DebuggerResources { copy_indirect_1d }
    }
}

static COPY_1D_SHADER: &[u32] = include_bytes_as_u32!("copy.spv");
const MAX_INDIRECTIONS: usize = 8;
const COPY_1D_WORKGROUP_SIZE: u32 = 32;

#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
struct CopyIndirect1DParams {
    base: vk::DeviceAddress,
    dst: vk::DeviceAddress,
    byte_size: u32,
    count: u32,
    offset: [u32; MAX_INDIRECTIONS],
}

//--------------------------------------------------------------------------------------------------

/// Manages the capture of data (buffer data & images) between commands.
pub struct Debugger {
    pub watches: SlotMap<WatchId, Watch>,
}

impl Debugger {
    pub fn new() -> Debugger {
        Debugger { watches: SlotMap::with_key() }
    }

    pub fn end_frame(&mut self, d: &Device) {
        // cleanup abandoned watches
        self.watches.retain(|_id, watch| {
            if watch.transient && watch.abandoned {
                match watch.capture {
                    CaptureKind::Buffer(ref mut c) => {
                        if let Some(result_buffer) = c.result.take() {
                            unsafe { d.destroy_buffer_helper(result_buffer) };
                        }
                    }
                    CaptureKind::Image(ref mut c) => {
                        if let Some(result) = c.result.take() {
                            unsafe {
                                d.destroy_image_helper(result.image);
                            }
                        }
                    }
                }
                false
            } else {
                true
            }
        });

        // reset the abandoned flag
        for watch in self.watches.values_mut() {
            watch.abandoned = true;
            watch.stale = true;
        }
    }

    unsafe fn update_watches_after_command(
        &mut self,
        d: &Device,
        eid: EId,
        cmd_buf: vk::CommandBuffer,
        push_data: &[u8],
    ) {
        for (id, watch) in self.watches.iter_mut() {
            if watch.eid == eid {
                watch.stale = false;
                Self::do_capture(d, watch, cmd_buf, push_data);
            }
        }
    }

    unsafe fn do_capture(d: &Device, watch: &mut Watch, cmd_buf: vk::CommandBuffer, push_data: &[u8]) {
        match watch.capture {
            CaptureKind::Buffer(ref mut cap) => {
                Self::do_capture_command_data(d, cmd_buf, push_data, cap);
            }
            CaptureKind::Image(ref mut cap) => {
                //unsafe {
                //    self.update_image_watch(cmd_buf, cap);
                //}
                // TODO
            }
        }
    }

    unsafe fn do_capture_command_data(
        d: &Device,
        cmd_buf: vk::CommandBuffer,
        push_data: &[u8],
        cap: &mut BufferCapture,
    ) {
        if cap.size == 0 {
            // nothing to copy.
            return;
        }

        // Allocate the result buffer.
        let result_buffer = cap.result.get_or_insert_with(|| {
            d.create_buffer_helper(
                vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::STORAGE_BUFFER,
                cap.size,
                None,
            )
        });

        let addr0 = push_data.as_ptr().add(cap.load_chain.offsets[0]);

        if cap.load_chain.offsets.len() == 1 {
            // there are no indirections into device memory, we can just copy the data from the push
            // data buffer
            ptr::copy_nonoverlapping(addr0, result_buffer.ptr as *mut u8, cap.size);
            return;
        }

        // The data spills in device memory, so dispatch a shader to copy from the rest of the load chain
        // into the provided buffer.

        let base1 = *(addr0 as *const u64);
        let addr1 = base1 + cap.load_chain.offsets[1] as u64;
        let count = cap.load_chain.offsets.len() - 2;

        let mut params = CopyIndirect1DParams {
            base: addr1,
            dst: result_buffer.device_address as u64,
            byte_size: cap.size as u32,
            count: count as u32,
            offset: [Default::default(); MAX_INDIRECTIONS],
        };
        for i in 0..count {
            params.offset[i] = cap.load_chain.offsets[i + 2] as u32;
        }

        //eprintln!("read_load_chain:");
        //eprintln!("   offset[0]={}", load_chain.offsets[0]);
        //eprintln!("   base1=0x{:016x}", base1);
        //eprintln!("   addr1=0x{:016x}", addr1);
        //eprintln!("   offsets={:?}", &load_chain.offsets[..]);
        //eprintln!("   count={}", count);
        //eprintln!("   device offsets={:?}", &params.offset[..]);

        let n_workgroups = cap.size.div_ceil(COPY_1D_WORKGROUP_SIZE as usize) as u32;
        d.cmd_bind_pipeline(cmd_buf, vk::PipelineBindPoint::COMPUTE, d.debugger_resources.copy_indirect_1d.pipeline);
        d.push_constants_helper(
            cmd_buf,
            d.debugger_resources.copy_indirect_1d.pipeline_layout,
            vk::ShaderStageFlags::COMPUTE,
            &params,
        );
        d.cmd_dispatch(cmd_buf, n_workgroups, 1, 1);
    }


    // Finds an existing watch by key.
    fn get_or_insert_watch(&mut self, eid: EId, key: impl Hash, f: impl FnOnce() -> CaptureKind) -> (WatchId, &mut Watch) {
        let mut h = FxHasher::default();
        (eid, key).hash(&mut h);
        let hash = h.finish();
        for (id, watch) in self.watches.iter_mut() {
            if watch.hash == hash {
                // found a matching watch, mark it as live, return its id
                watch.abandoned = false;
                //watch.last_request = self.frame_index;
                return (id, watch);
            }
        }
        let id = self.watches.insert(Watch {
            hash,
            eid,
            transient: true,
            abandoned: false,
            stale: true,
            capture: f(),
        });
        (id, &mut self.watches[id])
    }

    // Adds a debugger watch on command push data.
    fn add_data_watch(&mut self, eid: EId, load_chain: &LoadChain, byte_size: usize) -> (WatchId, &mut Watch) {
        self.get_or_insert_watch(eid, (0, load_chain, byte_size), || {
            CaptureKind::Buffer(BufferCapture { load_chain: load_chain.clone(), size: byte_size, result: None })
        })
    }

    fn add_image_watch(&mut self, eid: EId, image: vk::Image) -> (WatchId, &mut Watch) {
        self.get_or_insert_watch(eid, (1, image), || CaptureKind::Image(ImageCapture { image, result: None }))
    }

    pub fn capture_load_chain(&mut self, eid: EId, load_chain: &LoadChain, byte_size: usize) -> Option<Vec<u8>> {
        let (_watch_id, watch) = self.add_data_watch(eid, load_chain, byte_size);
        if watch.stale {
            // no data captured on this frame
            return None;
        }
        let CaptureKind::Buffer(ref capture) = watch.capture else { unreachable!() };
        if let Some(ref result_buffer) = capture.result {
            let result_slice = unsafe { std::slice::from_raw_parts(result_buffer.ptr as *const u8, capture.size) };
            Some(result_slice.to_vec())
        } else {
            None
        }
    }

    pub fn capture_image(&mut self, eid: EId, image: vk::Image) -> Option<CapturedImage> {
        let (_watch_id, watch) = self.add_image_watch(eid, image);
        if watch.stale {
            return None;
        }
        let CaptureKind::Image(ref capture) = watch.capture else { unreachable!() };
        if let Some(ref result) = capture.result { Some(result.clone()) } else { None }
    }
}

/// Debugger watch: a query that should run before/after a specified command.
pub struct BufferCapture {
    pub(crate) load_chain: LoadChain,  // Load chain to the data being inspected
    pub(crate) size: usize,            // Size in bytes to load from the chain
    pub(crate) result: Option<Buffer>, // Result buffer for holding the result of the query
}

pub struct ImageCapture {
    //

    pub(crate) image: vk::Image,
    pub(crate) result: Option<CapturedImage>,
}

pub enum CaptureKind {
    Buffer(BufferCapture),
    Image(ImageCapture),
}

pub struct Watch {
    pub eid: EId,
    pub(crate) hash: u64,       // Unique hash
    pub(crate) transient: bool, // Whether this watch is temporary (removed if not read in the last frame)
    pub(crate) abandoned: bool, //
    pub(crate) stale: bool,     // If true, no data was captured for this watch in the last frame
    pub capture: CaptureKind,
}

new_key_type! {
    pub struct WatchId;
}

#[derive(Clone)]
pub struct CapturedImage {
    pub data: Vec<u8>,
    pub image: Image,
    pub info: ImageInfo,
}

impl Device {
    /// Called after every command to update relevant watches.
    pub fn after_command(&self, command: &Command) {
        let mut dbg = self.debugger.lock();
        unsafe {
            dbg.update_watches_after_command(self, command.eid, command.cmd_buf, command.push.as_slice());
        }
    }
}

fn image_buffer_size(image_info: &ImageInfo) -> usize {
    let format_info = format_info(image_info.format).unwrap();
    let pixel_size = image_info.size.width as usize
        * image_info.size.height as usize
        * image_info.size.depth as usize
        * format_info.block_size as usize;
    pixel_size
}
