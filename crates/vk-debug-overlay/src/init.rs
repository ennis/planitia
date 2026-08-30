//! Instance & device initialization code.
use crate::dispatch::InstanceDispatch;
use crate::{DeviceDispatchableHandle, Device, DEVICE_STATE, INSTANCE_MAP,
            PHY_TO_INSTANCE,
};
use ash::vk;
use ash::vk::{PFN_vkCreateDevice, PFN_vkCreateInstance, PFN_vkDestroyDevice, PFN_vkDestroyInstance};
use ash_layer::{get_device_chain_info, get_instance_chain_info, LayerFunction};
use std::mem;

const _: PFN_vkCreateInstance = layer_vkCreateInstance;
const _: PFN_vkDestroyInstance = layer_vkDestroyInstance;
const _: PFN_vkCreateDevice = layer_vkCreateDevice;
const _: PFN_vkDestroyDevice = layer_vkDestroyDevice;

#[unsafe(no_mangle)]
pub(crate) unsafe extern "system" fn layer_vkCreateInstance(
    p_create_info: *const vk::InstanceCreateInfo,
    p_allocator: *const vk::AllocationCallbacks,
    p_instance: *mut vk::Instance,
) -> vk::Result {
    let create_info = *p_create_info;

    let chain_info = match get_instance_chain_info(&create_info, LayerFunction::LAYER_LINK_INFO) {
        Some(mut p) => p.as_mut(),
        None => return vk::Result::ERROR_INITIALIZATION_FAILED,
    };

    // Consume the head of the layer-info linked list.
    let layer_info = *chain_info.u.p_layer_info;
    chain_info.u.p_layer_info = layer_info.p_next;

    let gipa = layer_info.pfn_next_get_instance_proc_addr.expect("pfnNextGetInstanceProcAddr is null");
    let gpdpa = layer_info.pfn_next_get_physical_device_proc_addr.expect("pfnNextGetPhysicalDeviceProcAddr is null");

    // Call down the chain.
    let create_instance: vk::PFN_vkCreateInstance =
        mem::transmute(gipa(vk::Instance::null(), c"vkCreateInstance".as_ptr()));
    let res = create_instance(p_create_info, p_allocator, p_instance);
    if res != vk::Result::SUCCESS {
        return res;
    }

    let instance = *p_instance;

    // Load ash instance function pointers (next layer's pointers).
    {
        eprintln!("[planitia-layer] vkCreateInstance {:?}", instance);
    };
    let dispatch = InstanceDispatch::new(gipa, gpdpa, instance);

    // Map every physical device to its parent instance for vkCreateDevice lookup.
    if let Ok(phy_devices) = dispatch.enumerate_physical_devices() {
        for pd in phy_devices {
            PHY_TO_INSTANCE.insert(pd, instance);
        }
    }

    INSTANCE_MAP.insert(instance, dispatch);

    vk::Result::SUCCESS
}

#[unsafe(no_mangle)]
pub(crate) unsafe extern "system" fn layer_vkDestroyInstance(
    instance: vk::Instance,
    p_allocator: *const vk::AllocationCallbacks,
) {
    if let Some((_, layer_instance)) = INSTANCE_MAP.remove(&instance) {
        if let Ok(phy_devices) = layer_instance.d.enumerate_physical_devices() {
            for pd in phy_devices {
                PHY_TO_INSTANCE.remove(&pd);
            }
        }
        {
            eprintln!("[planitia-layer] vkDestroyInstance {:?}", instance);
        };
        (layer_instance.d.fp_v1_0().destroy_instance)(instance, p_allocator);
    }
}

// ---------------------------------------------------------------------------
// vkCreateDevice / vkDestroyDevice
// ---------------------------------------------------------------------------

#[unsafe(no_mangle)]
pub(crate) unsafe extern "system" fn layer_vkCreateDevice(
    physical_device: vk::PhysicalDevice,
    p_create_info: *const vk::DeviceCreateInfo,
    p_allocator: *const vk::AllocationCallbacks,
    p_device: *mut vk::Device,
) -> vk::Result {
    let instance = *PHY_TO_INSTANCE.get(&physical_device).expect("unknown physical device");
    let instance_dispatch = INSTANCE_MAP.get(&instance).expect("unknown instance");

    let chain_info = match get_device_chain_info(&*p_create_info, LayerFunction::LAYER_LINK_INFO) {
        Some(mut p) => p.as_mut(),
        None => return vk::Result::ERROR_INITIALIZATION_FAILED,
    };

    let layer_info = *chain_info.u.p_layer_info;
    chain_info.u.p_layer_info = layer_info.p_next;

    //let next_get_instance_proc_addr = layer_info.pfn_next_get_instance_proc_addr.expect("pfnNextGetInstanceProcAddr is null");
    let next_get_device_proc_addr = layer_info.pfn_next_get_device_proc_addr.expect("pfnNextGetDeviceProcAddr is null");

    let set_device_loader_data = match get_device_chain_info(&*p_create_info, LayerFunction::LOADER_DATA_CALLBACK) {
        Some(mut p) => p.as_mut().u.pfn_set_device_loader_data.expect("pfnSetDeviceLoaderData is null"),
        None => return vk::Result::ERROR_INITIALIZATION_FAILED,
    };

    // Create the device.
    let res = (instance_dispatch.d.fp_v1_0().create_device)(physical_device, p_create_info, p_allocator, p_device);
    if res != vk::Result::SUCCESS {
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
    vk::Result::SUCCESS
}

#[unsafe(no_mangle)]
pub(crate) unsafe extern "system" fn layer_vkDestroyDevice(
    device: vk::Device,
    p_allocator: *const vk::AllocationCallbacks,
) {
    if let Some((_, device_state)) = DEVICE_STATE.remove(&device.key()) {
        //eprintln!("[planitia-layer] vkDestroyDevice {:?}", device);
        (device_state.fp_v1_0().destroy_device)(device, p_allocator);
    }
}
