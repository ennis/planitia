use crate::helper::Buffer;
use crate::{COPY_1D_MAX_INDIRECTIONS, COPY_1D_WORKGROUP_SIZE, CopyIndirect1DParams, Device};
use core::fmt;
use std::ptr;
use vulkan::{VkBindHeapInfoEXT, VkCommandBuffer, VkDeviceAddress, VK_PIPELINE_BIND_POINT_COMPUTE, VK_SHADER_STAGE_COMPUTE_BIT};
use crate::descriptor::ImageViewInfo;

/// Represents a sequence of pointer indirections from some base location, (e.g. `base->field->field2 ...`).
// TODO Rename to `PointerChain`? IndirectionChain?
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

/// Represents the base of a load chain.
pub enum LoadChainBase<'a> {
    /// The base address is an address in global memory.
    DeviceAddress(VkDeviceAddress),
    /// Resolves the load chain relative to the provided buffer.
    ///
    /// The first indirection of the chain is read from this buffer.
    /// Subsequent indirections are read from device memory.
    Slice(&'a [u8]),
}

impl Device {
    /// Copies data from the device, at the memory location reachable from `base` by a pointer indirection `chain`.
    ///
    /// This function walks the pointer indirection chain to the target address,
    /// and copies `size` bytes from that address into `out_buffer`.
    ///
    /// The type indirection depends on the value of `base`:
    /// - [`LoadChainBase::DeviceAddress(base_address)`](LoadChainBase::DeviceAddress):
    ///   the offsets in `chain` are interpreted as offsets relative to a device memory address, starting with `base_address`
    /// - [`LoadChainBase::Slice(slice)`](LoadChainBase::Slice):
    ///   the first offset is interpreted an as offset into the specified byte slice, the rest are interpreted as offsets relative to a device memory address.
    ///
    /// # Pseudocode
    /// ```
    /// if base is DeviceAddress(base_address):
    ///     addr = base_address
    ///     for i in 0..N-1:
    ///         addr = *(addr + offsets[i])    // device-memory load
    ///     device_memcpy(out_buffer, addr + offsets[N-1], size)
    /// else if base is Slice(slice):
    ///     addr = slice.as_ptr()
    ///     if N < 2:
    ///         memcpy(out_buffer, addr + offsets[0], size) // host copy
    ///     else:
    ///         addr = *(addr + offsets[0])   // host load
    ///         for i in 1..N-1:
    ///             addr = *(addr + offsets[i])   // device-memory load
    ///         device_memcpy(out_buffer, addr + offsets[N-1], size)
    /// ```
    pub unsafe fn read_load_chain(
        &self,
        cmd_buf: VkCommandBuffer,
        base: LoadChainBase,
        chain: &LoadChain,
        size: usize,
        out_buffer: &Buffer,
    ) {
        // offsets for the part of the chain in device memory
        let offsets;
        // first address to load from in device memory
        let addr0;
        match base {
            LoadChainBase::DeviceAddress(base_address) => {
                addr0 = base_address;
                offsets = &chain.offsets[..];
            }
            LoadChainBase::Slice(data) => {
                let host_addr = data.as_ptr().add(chain.offsets.first().cloned().unwrap_or(0));
                if chain.offsets.len() < 2 {
                    // there are no indirections into device memory, the target is in the provided host buffer
                    ptr::copy_nonoverlapping(host_addr, out_buffer.ptr as *mut u8, size);
                    return;
                }
                addr0 = *(host_addr as *const u64);
                offsets = &chain.offsets[1..];
            }
        }

        let addr1 = addr0 + offsets.first().cloned().unwrap_or(0) as u64;
        let count = offsets.len().saturating_sub(1);
        let mut params = CopyIndirect1DParams {
            base: addr1,
            dst: out_buffer.device_address as u64,
            byte_size: size as u32,
            count: count as u32,
            offset: [Default::default(); COPY_1D_MAX_INDIRECTIONS],
        };
        for i in 1..count {
            params.offset[i] = offsets[i] as u32;
        }
        let n_workgroups = size.div_ceil(COPY_1D_WORKGROUP_SIZE as usize) as u32;
        self.CmdBindPipeline(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, self.copy_indirect_1d.pipeline);
        self.push_constants_helper(
            cmd_buf,
            self.copy_indirect_1d.pipeline_layout,
            VK_SHADER_STAGE_COMPUTE_BIT,
            &params,
        );
        self.CmdDispatch(cmd_buf, n_workgroups, 1, 1);

        //eprintln!("read_load_chain:");
        //eprintln!("   offset[0]={}", load_chain.offsets[0]);
        //eprintln!("   base1=0x{:016x}", base1);
        //eprintln!("   addr1=0x{:016x}", addr1);
        //eprintln!("   offsets={:?}", &load_chain.offsets[..]);
        //eprintln!("   count={}", count);
        //eprintln!("   device offsets={:?}", &params.offset[..]);
    }

    pub fn copy_image_indirect(&self,
                               cmd_buf: VkCommandBuffer,
                               base: LoadChainBase,
                               descriptor: &LoadChain,
                               image_view_info: &ImageViewInfo,
                               resource_descriptors_heap: &VkBindHeapInfoEXT,
                                out_buffer: &Buffer,
    ) {

        // issue: capturing images is complicated
        // - the only way to reliably identify an image for the current frame is a load chain pointing to a texture/image descriptor index
        // - vkCmdCopy* commands cannot be used because they need a VkImage object
        // - we don't know the size of the image in advance for the current frame
        //      - this would require reading back the descriptor heap "in the middle" of a frame,
        //        i.e. sync with CPU in the middle of a command buffer, which is annoying do to
        //      - or, it would require tracking host writes to descriptor heap memory so that we
        //        know what's in the descriptor heap without having to read it back; also annoying
        // - it's possible to assume that the image is the same on every frame for a command,
        //   but that's not necessarily the case
        //
        // On NV, the descriptor blob contains all the necessary metadata and it is accessible
        // from shaders.
        // It's the same on AMD.

        // Problem: how to allocate an image of the correct size for readback when we don't know
        // the image size in advance?
        // - Option A: use the size that is known when making the first request


    }
}
