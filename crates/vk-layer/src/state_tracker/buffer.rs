use crate::helper::HasPrivateData;
use crate::DeviceState;
use ash::vk;
use std::slice;

#[derive(Copy, Clone, Debug)]
struct AddressRange {
    base: vk::DeviceAddress,
    size: u64,
    buffer: vk::Buffer,
}

pub struct AddressMap {
    // sorted by base
    ranges: Vec<AddressRange>,
}

impl AddressMap {
    pub fn new() -> AddressMap {
        AddressMap { ranges: Vec::new() }
    }

    pub fn insert(&mut self, buffer: vk::Buffer, base: vk::DeviceAddress, size: u64) {
        let new_range = AddressRange { base, size, buffer };
        let pos = self.ranges.binary_search_by(|r| r.base.cmp(&base)).unwrap_or_else(|e| e);
        self.ranges.insert(pos, new_range);
    }

    pub fn remove(&mut self, buffer: vk::Buffer) {
        if let Some(pos) = self.ranges.iter().position(|r| r.buffer == buffer) {
            self.ranges.remove(pos);
        }
    }

    pub fn lookup(&mut self, address: vk::DeviceAddress) -> Option<(vk::Buffer, u64)> {
        // NOTE: aliasing is possible, this returns the first buffer that matches
        let pos = self
            .ranges
            .binary_search_by(|r| {
                if address < r.base {
                    std::cmp::Ordering::Greater
                } else if address >= r.base + r.size {
                    std::cmp::Ordering::Less
                } else {
                    std::cmp::Ordering::Equal
                }
            })
            .ok()?;
        let range = &self.ranges[pos];
        Some((range.buffer, address - range.base))
    }
}

pub struct BufferData {
    pub(crate) name: String,
    device_address: vk::DeviceAddress,
    size: u64,
}

impl HasPrivateData for vk::Buffer {
    type PrivateData = BufferData;
}

impl DeviceState {
    pub unsafe fn hook_create_buffer(
        &self,
        device: vk::Device,
        p_create_info: *const vk::BufferCreateInfo,
        p_allocator: *const vk::AllocationCallbacks,
        p_buffer: *mut vk::Buffer,
    ) -> vk::Result {
        let r = (self.fp_v1_0().create_buffer)(device, p_create_info, p_allocator, p_buffer);
        if r != vk::Result::SUCCESS {
            return r;
        }

        self.set_private_data(*p_buffer, BufferData { name: Default::default(), device_address: 0, size: (*p_create_info).size });
        vk::Result::SUCCESS
    }

    unsafe fn register_buffer_address_range(&self, buffer: vk::Buffer) {
        // SAFETY: safe because buffer is an externally-synchronized parameter
        let buf_data = self.get_private_data_mut(buffer).unwrap();

        // Get buffer device address and register range
        buf_data.device_address =
            self.get_buffer_device_address(&vk::BufferDeviceAddressInfo { buffer, ..Default::default() });

        if buf_data.device_address != 0 {
            //eprintln!(
            //    "[planitia-layer] add mapping {:016x}->{:016x}: {:p} ({} bytes) ",
            //    buf_data.device_address,
            //    buf_data.device_address + buf_data.size,
            //    buffer,
            //    buf_data.size,
            //);
            self.addrmap.lock().insert(buffer, buf_data.device_address, buf_data.size);
        }
    }

    pub unsafe fn hook_bind_buffer_memory_2(
        &self,
        device: vk::Device,
        bind_info_count: u32,
        p_bind_infos: *const vk::BindBufferMemoryInfo<'_>,
    ) -> vk::Result {
        let r = (self.fp_v1_1().bind_buffer_memory2)(device, bind_info_count, p_bind_infos);
        if r != vk::Result::SUCCESS {
            return r;
        }

        let bind_infos = slice::from_raw_parts(p_bind_infos, bind_info_count as usize);
        for bind_info in bind_infos {
            self.register_buffer_address_range(bind_info.buffer);
        }

        vk::Result::SUCCESS
    }

    pub unsafe fn hook_bind_buffer_memory(
        &self,
        device: vk::Device,
        buffer: vk::Buffer,
        memory: vk::DeviceMemory,
        memory_offset: vk::DeviceSize,
    ) -> vk::Result {
        let r = (self.fp_v1_0().bind_buffer_memory)(device, buffer, memory, memory_offset);
        if r != vk::Result::SUCCESS {
            return r;
        }

        self.register_buffer_address_range(buffer);
        vk::Result::SUCCESS
    }

    pub unsafe fn hook_destroy_buffer(
        &self,
        device: vk::Device,
        buffer: vk::Buffer,
        p_allocator: *const vk::AllocationCallbacks,
    ) {
        let buf_data = self.take_private_data(buffer).unwrap();

        if buf_data.device_address != 0 {
            //eprintln!(
            //    "[planitia-layer] remove mapping {:016x}->{:016x}: {:p}",
            //    buf_data.device_address,
            //    buf_data.device_address + buf_data.size,
            //    buffer,
            //);
            self.addrmap.lock().remove(buffer);
        }

        (self.fp_v1_0().destroy_buffer)(device, buffer, p_allocator);
    }
}
