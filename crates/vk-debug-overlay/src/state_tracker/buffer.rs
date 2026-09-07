use crate::helper::HasPrivateData;
use crate::Device;
use ash::vk;
use std::slice;
use ash::VkHandle;

pub struct BufferData {
    pub  name: String,
    pub device_address: VkDeviceAddress,
    pub size: u64,
}

impl HasPrivateData for VkBuffer {
    type PrivateData = BufferData;
}

impl Device {
    pub unsafe fn hook_create_buffer(
        &self,
        device: VkDevice,
        p_create_info: *const VkBufferCreateInfo,
        p_allocator: *const VkAllocationCallbacks,
        p_buffer: *mut VkBuffer,
    ) -> VkResult {
        let r = (self.fp_v1_0().create_buffer)(device, p_create_info, p_allocator, p_buffer);
        if r != VK_SUCCESS {
            return r;
        }

        self.set_private_data(*p_buffer, BufferData { name: format!("Buffer_{:016x}", (*p_buffer).as_raw()), device_address: 0, size: (*p_create_info).size });
        VK_SUCCESS
    }


    pub unsafe fn hook_bind_buffer_memory_2(
        &self,
        device: VkDevice,
        bind_info_count: u32,
        p_bind_infos: *const VkBindBufferMemoryInfo<'_>,
    ) -> VkResult {
        let r = (self.fp_v1_1().bind_buffer_memory2)(device, bind_info_count, p_bind_infos);
        if r != VkResult::SUCCESS {
            return r;
        }

        let bind_infos = slice::from_raw_parts(p_bind_infos, bind_info_count as usize);
        for bind_info in bind_infos {
            self.register_buffer_address_range(bind_info.buffer);
        }

        VK_SUCCESS
    }

    pub unsafe fn hook_bind_buffer_memory(
        &self,
        device: VkDevice,
        buffer: VkBuffer,
        memory: VkDeviceMemory,
        memory_offset: VkDeviceSize,
    ) -> VkResult {
        let r = (self.fp_v1_0().bind_buffer_memory)(device, buffer, memory, memory_offset);
        if r != VK_SUCCESS {
            return r;
        }

        self.register_buffer_address_range(buffer);
        VK_SUCCESS
    }

    pub unsafe fn hook_destroy_buffer(
        &self,
        device: VkDevice,
        buffer: VkBuffer,
        p_allocator: *const VkAllocationCallbacks,
    ) {
        let buf_data = self.take_private_data(buffer).unwrap();

        if buf_data.device_address != 0 {
            //eprintln!(
            //    "[planitia-layer] remove mapping {:016x}->{:016x}: {:p}",
            //    buf_data.device_address,
            //    buf_data.device_address + buf_data.size,
            //    buffer,
            //);
            self.addrmap.lock().remove_buffer(buffer);
        }

        (self.fp_v1_0().destroy_buffer)(device, buffer, p_allocator);
    }
}
