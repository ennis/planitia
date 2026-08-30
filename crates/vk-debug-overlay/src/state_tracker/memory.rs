use crate::Device;
use ash::vk;

#[derive(Copy, Clone, Debug)]
pub struct AddressRange {
    pub base: vk::DeviceAddress,
    pub size: u64,
    pub handle: vk::Buffer,
}

pub struct AddressMap {
    // sorted by base
    pub ranges: Vec<AddressRange>,
}

impl AddressMap {
    pub fn new() -> AddressMap {
        AddressMap { ranges: Vec::new() }
    }

    pub fn insert_buffer(&mut self, buffer: vk::Buffer, base: vk::DeviceAddress, size: u64) {
        let new_range = AddressRange { base, size, handle: buffer };
        let pos = self.ranges.binary_search_by(|r| r.base.cmp(&base)).unwrap_or_else(|e| e);
        self.ranges.insert(pos, new_range);
    }

    pub fn remove_buffer(&mut self, buffer: vk::Buffer) {
        if let Some(pos) = self.ranges.iter().position(|r| r.handle == buffer) {
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
        Some((range.handle, address - range.base))
    }
}

impl Device {
    pub unsafe fn register_buffer_address_range(&self, buffer: vk::Buffer) {
        // SAFETY: safe because buffer is an externally-synchronized parameter
        let buf_data = self.get_private_data_mut(buffer).unwrap();

        // Get buffer device address and register range
        buf_data.device_address =
            self.get_buffer_device_address(&vk::BufferDeviceAddressInfo { buffer, ..Default::default() });

        if buf_data.device_address != 0 {
            self.addrmap.lock().insert_buffer(buffer, buf_data.device_address, buf_data.size);
        }
    }
}
