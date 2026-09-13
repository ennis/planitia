//! Instance & device initialization code.

// Contains code from ash_layer
//
// Copyright (c) 2022 Huang-Huang Bao
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

use crate::dispatch::InstanceDispatch;
use crate::{DEVICE_STATE, Device, DeviceDispatchableHandle, INSTANCE_MAP, PHY_TO_INSTANCE};
use vulkan::vk_layer::*;
use vulkan::*;
use std::mem;
use std::ptr::NonNull;

const _: PFN_vkCreateInstance = layer_vkCreateInstance;
const _: PFN_vkDestroyInstance = layer_vkDestroyInstance;
const _: PFN_vkCreateDevice = layer_vkCreateDevice;
const _: PFN_vkDestroyDevice = layer_vkDestroyDevice;

// Adapted from ash_layer
pub unsafe fn get_instance_chain_info(
    create_info: &VkInstanceCreateInfo,
    function: VkLayerFunction,
) -> Option<NonNull<VkLayerInstanceCreateInfo>> {
    let mut chain_info_ptr = create_info.pNext.cast::<VkLayerInstanceCreateInfo>();
    while !chain_info_ptr.is_null() {
        let chain_info = chain_info_ptr.read();
        if chain_info.sType == VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO && chain_info.function == function {
            return Some(NonNull::new_unchecked(chain_info_ptr as _));
        }
        chain_info_ptr = chain_info.pNext.cast()
    }
    None
}

pub unsafe fn get_device_chain_info(
    create_info: &VkDeviceCreateInfo,
    function: VkLayerFunction,
) -> Option<NonNull<VkLayerDeviceCreateInfo>> {
    let mut chain_info_ptr = create_info.pNext.cast::<VkLayerDeviceCreateInfo>();
    while !chain_info_ptr.is_null() {
        let chain_info = chain_info_ptr.read();
        if chain_info.sType == VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO && chain_info.function == function {
            return Some(NonNull::new_unchecked(chain_info_ptr as _));
        }
        chain_info_ptr = chain_info.pNext.cast()
    }
    None
}

#[unsafe(no_mangle)]
pub(crate) unsafe extern "system" fn layer_vkCreateInstance(
    p_create_info: *const VkInstanceCreateInfo,
    p_allocator: *const VkAllocationCallbacks,
    p_instance: *mut VkInstance,
) -> VkResult {
    let create_info = *p_create_info;
    let chain_info = match get_instance_chain_info(&create_info, VK_LAYER_LINK_INFO) {
        Some(mut p) => p.as_mut(),
        None => return VK_ERROR_INITIALIZATION_FAILED,
    };
    // Consume the head of the layer-info linked list.
    let layer_info = *chain_info.u.pLayerInfo;
    chain_info.u.pLayerInfo = layer_info.pNext;
    let gipa = layer_info.pfnNextGetInstanceProcAddr.unwrap();
    let gpdpa = layer_info.pfnNextGetPhysicalDeviceProcAddr.unwrap();
    // Call down the chain.
    let create_instance: PFN_vkCreateInstance = mem::transmute(gipa(VkInstance::null(), c"vkCreateInstance".as_ptr()));
    let res = create_instance(p_create_info, p_allocator, p_instance);
    if res < 0 {
        return res;
    }
    let instance = *p_instance;
    // Load ash instance function pointers (next layer's pointers).
    eprintln!("[planitia-layer] vkCreateInstance {:?}", instance);
    let dispatch = InstanceDispatch::new(gipa, gpdpa, instance);
    // Map every physical device to its parent instance for vkCreateDevice lookup.
    vkarraycall!(dispatch.EnumeratePhysicalDevices(instance, @count let count, @out let phy_devices));
    for pd in phy_devices {
        PHY_TO_INSTANCE.insert(pd, instance);
    }
    INSTANCE_MAP.insert(instance, dispatch);
    res
}

#[unsafe(no_mangle)]
pub(crate) unsafe extern "system" fn layer_vkDestroyInstance(
    instance: VkInstance,
    p_allocator: *const VkAllocationCallbacks,
) {
    if let Some((_, layer_instance)) = INSTANCE_MAP.remove(&instance) {
        vkarraycall!(layer_instance.fns.EnumeratePhysicalDevices(instance, @count let count, @out let phy_devices));
        for pd in phy_devices {
            PHY_TO_INSTANCE.remove(&pd);
        }
        eprintln!("[planitia-layer] vkDestroyInstance {:?}", instance);
        layer_instance.fns.DestroyInstance(instance, p_allocator);
    }
}

// ---------------------------------------------------------------------------
// vkCreateDevice / vkDestroyDevice
// ---------------------------------------------------------------------------

#[unsafe(no_mangle)]
pub(crate) unsafe extern "system" fn layer_vkCreateDevice(
    physical_device: VkPhysicalDevice,
    p_create_info: *const VkDeviceCreateInfo,
    p_allocator: *const VkAllocationCallbacks,
    p_device: *mut VkDevice,
) -> VkResult {
    let instance = *PHY_TO_INSTANCE.get(&physical_device).expect("unknown physical device");
    let instance_dispatch = INSTANCE_MAP.get(&instance).expect("unknown instance");
    let chain_info = match get_device_chain_info(&*p_create_info, VK_LAYER_LINK_INFO) {
        Some(mut p) => p.as_mut(),
        None => return VK_ERROR_INITIALIZATION_FAILED,
    };
    let layer_info = *chain_info.u.pLayerInfo;
    chain_info.u.pLayerInfo = layer_info.pNext;

    //let next_get_instance_proc_addr = layer_info.pfn_next_get_instance_proc_addr.expect("pfnNextGetInstanceProcAddr is null");
    let next_get_device_proc_addr = layer_info.pfnNextGetDeviceProcAddr.unwrap();

    let set_device_loader_data = match get_device_chain_info(&*p_create_info, VK_LOADER_DATA_CALLBACK) {
        Some(mut p) => p.as_mut().u.pfnSetDeviceLoaderData.unwrap(),
        None => return VK_ERROR_INITIALIZATION_FAILED,
    };

    // Create the device.
    let res = instance_dispatch.CreateDevice(physical_device, p_create_info, p_allocator, p_device);
    if res < 0 {
        return res;
    }

    let device = *p_device;
    let create_info = &*p_create_info;
    let device_state = Device::new(
        &instance_dispatch,
        device,
        create_info,
        physical_device,
        next_get_device_proc_addr,
        set_device_loader_data,
    );
    DEVICE_STATE.insert(device.key(), device_state);
    //eprintln!("[planitia-layer] vkCreateDevice {:?}", device);
    res
}

#[unsafe(no_mangle)]
pub(crate) unsafe extern "system" fn layer_vkDestroyDevice(
    device: VkDevice,
    p_allocator: *const VkAllocationCallbacks,
) {
    if let Some((_, device_state)) = DEVICE_STATE.remove(&device.key()) {
        //eprintln!("[planitia-layer] vkDestroyDevice {:?}", device);
        device_state.DestroyDevice(device, p_allocator);
    }
}
