//! Vulkan debug overlay layer.
//!
//! This layer intercepts Vulkan API calls and renders a debug overlay on top of the application.
//! Currently displayed:
//! - draw/compute command parameters
//! - push data and data reachable via device addresses in push data

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

#![allow(non_snake_case)]
#![allow(unsafe_op_in_unsafe_fn, reason = "too verbose")]
#![feature(default_field_values)]
extern crate core;

mod bump;
mod debugger;
mod descriptor;
mod dispatch;
mod event;
mod helper;
mod load_chain;
mod overlay;
mod util;
mod wrappers;

use crate::bump::BumpAllocator;
use crate::debugger::{CommandWatch, DebuggerState, DescriptorBlob, ResourceDescriptorInfo};
use crate::dispatch::{DeviceDispatch, DeviceDispatchableHandle, DispatchKey, InstanceDispatch};
use crate::event::EventTimeline;
use crate::helper::{DeviceHelper, Pipeline, include_bytes_as_u32};
use crate::overlay::gui::GuiState;
use crate::overlay::input::InputState;
use crate::overlay::renderer::OverlayResources;
use crate::wrappers::command::Command;
use crate::wrappers::memory::AddressMap;
use bumpalo::Bump;
use core::ffi::{CStr, c_char};
use core::mem;
use dashmap::DashMap;
use parking_lot::Mutex;
use range_set::RangeSet;
use slotmap::{SlotMap, new_key_type};
use spirv_reflect::Module;
use std::collections::HashMap;
use std::ffi::c_void;
use std::ops::{Deref, RangeInclusive};
use std::slice;
use std::sync::LazyLock;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering::Relaxed;
use vulkan::*;

// High-level module overview:
// - wrappers: vulkan command wrappers
// - capture: command data capture (uniforms, resources, etc.)
// - overlay: GUI
//
// Helper modules:
// - dispatch: device/instance function pointer tables
// - spirv_reflect (external crate): SPIR-V reflection


/// Maximum number of frames in flight.
///
/// Used as an upper bound for the number of in-flight resources allocated for the overlay.
const FRAMES_IN_FLIGHT: usize = 3;

/// Device state.
///
/// This holds the state tracker and resources for rendering the overlay.
pub struct Device {
    pub helper: DeviceHelper,                      // device dispatch tables + vulkan helpers
    pub tracked_objects: Mutex<TrackedObjects>,    // various tracked objects
    pub addrmap: Mutex<AddressMap>,                // Maps device addresses to buffer objects
    pub event_timeline: Mutex<EventTimeline>, // Used to generate event IDs (EIDs) that are coherent between frames.
    pub gui: Mutex<GuiState>,                 // GUI state
    pub overlay: OverlayResources,            // Overlay-related state and resources
    pub modules: Mutex<SlotMap<ModuleId, Module>>, // SPIR-V modules
    pub input: Mutex<InputState>,             // Tracks inputs to the main window
    pub bump: Mutex<BumpAllocator>,           // FIXME Not sure what this is for
    pub frame_index: AtomicU64,               // Frame counter
    pub dbg: Mutex<DebuggerState>,

    pub descriptor_heap_properties: VkPhysicalDeviceDescriptorHeapPropertiesEXT,
    pub descriptor_state: Mutex<DescriptorState>,
    // Queue submissions for the current frame
    pub submissions: Mutex<Vec<QueueSubmit>>,

    // Pipelines
    copy_indirect_1d: Pipeline,
    copy_image_indirect: Pipeline,
}

impl Device {
    unsafe fn new(
        instance_dispatch: &InstanceDispatch,
        device: VkDevice,
        create_info: &VkDeviceCreateInfo,
        physical_device: VkPhysicalDevice,
        next_get_device_proc_addr: PFN_vkGetDeviceProcAddr,
        set_device_loader_data: vk_layer::PFN_vkSetDeviceLoaderData,
    ) -> Device {
        let dispatch = DeviceDispatch::new(device, next_get_device_proc_addr, set_device_loader_data).unwrap();
        let first_queue_family = {
            let qcis = slice::from_raw_parts(create_info.pQueueCreateInfos, create_info.queueCreateInfoCount as usize);
            qcis.first().map_or(0, |q| q.queueFamilyIndex)
        };
        vkcallnc!(instance_dispatch.fns.GetPhysicalDeviceMemoryProperties(physical_device, @out let mem_props));
        let mut descriptor_heap_properties: VkPhysicalDeviceDescriptorHeapPropertiesEXT = unsafe { mem::zeroed() };
        descriptor_heap_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_HEAP_PROPERTIES_EXT;
        let mut physical_device_properties: VkPhysicalDeviceProperties2 = unsafe { mem::zeroed() };
        physical_device_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        physical_device_properties.pNext = &mut descriptor_heap_properties as *mut _ as *mut c_void;
        instance_dispatch.fns.GetPhysicalDeviceProperties2(physical_device, &mut physical_device_properties);
        let helper = DeviceHelper::new(dispatch, mem_props, first_queue_family);
        let tracked_resources = TrackedObjects { pipelines: Vec::new(), swapchains: Vec::new() };
        let overlay_resources = OverlayResources::new(&helper);
        let event_timeline = EventTimeline::new();
        let descriptor_state =
            DescriptorState { resource_descriptors: HashMap::new(), sampler_descriptors: HashMap::new() };
        let debugger_state = DebuggerState::new();

        // Pipelines
        static COPY_SHADER: &[u32] = include_bytes_as_u32!("copy.spv");
        let copy_indirect_1d = helper.create_compute_pipeline_helper(
            COPY_SHADER,
            c"copy_indirect_1d",
            &[],
            size_of::<CopyIndirect1DParams>(),
        );
        let copy_image_indirect = helper.create_compute_pipeline_helper(
            COPY_SHADER,
            c"copy_image_indirect",
            &[],
            size_of::<CopyIndirect1DParams>(),
        );

        Device {
            helper,
            tracked_objects: Mutex::new(tracked_resources),
            addrmap: Mutex::new(AddressMap::new()),
            modules: Mutex::new(SlotMap::with_key()),
            input: Mutex::new(InputState::new()),
            bump: Mutex::new(BumpAllocator::new()),
            gui: Mutex::new(GuiState::new()),
            event_timeline: Mutex::new(event_timeline),
            overlay: overlay_resources,
            frame_index: Default::default(),
            descriptor_heap_properties,
            descriptor_state: Mutex::new(descriptor_state),
            submissions: Default::default(),
            dbg: Mutex::new(debugger_state),
            copy_indirect_1d,
            copy_image_indirect,
        }
    }

    unsafe fn end_frame(&self) {
        let mut bump = self.bump.lock();
        bump.reset();
        self.update_watches();
        // reset the list of submissions
        self.submissions.lock().clear();
        self.frame_index.fetch_add(1, Relaxed);
    }

    fn get_frame_index(&self) -> u64 {
        self.frame_index.load(Relaxed)
    }
}

// Deref chain for device:
// Device
// -> DeviceHelper (command submission helper)
// -> DeviceDispatch (layer loader helper methods)
// -> DeviceDispatchCombined (vulkan dispatch tables)
impl Deref for Device {
    type Target = DeviceHelper;
    fn deref(&self) -> &Self::Target {
        &self.helper
    }
}

pub struct DescriptorState {
    // Map from device opaque descriptor blob to the VkResourceDescriptorInfo/VkSamplerCreateInfo
    pub resource_descriptors: HashMap<DescriptorBlob, ResourceDescriptorInfo>,
    pub sampler_descriptors: HashMap<DescriptorBlob, VkSamplerCreateInfo>,
}

/// Represents a call to vkQueueSubmit.
pub struct QueueSubmit {
    //pub cmd_bufs: Vec<VkCommandBuffer>,
    pub commands: Vec<Command>,
    // Recorded descriptor heap device address ranges.
    //pub resource_heap_ranges: DeviceAddressRangeSet,
    //pub sampler_heap_ranges: DeviceAddressRangeSet,
    // Captured descriptor heap memory contents; each element corresponds to a range in the corresponding DeviceAddressRangeSet
    //pub resource_heap_memory: Vec<Buffer>,
    //pub sampler_heap_memory: Vec<Buffer>,
}

new_key_type! {
    /// SPIR-V module identifier.
    pub struct ModuleId;
}

pub type ModuleMap = SlotMap<ModuleId, Module>;
pub type DeviceAddressRangeSet = RangeSet<[RangeInclusive<VkDeviceAddress>; 4]>;

/// Information about a swapchain.
pub struct SwapchainInfo {
    surface: VkSurfaceKHR,
    device: VkDevice,
    format: VkFormat,
    extent: VkExtent2D,
    swapchain: VkSwapchainKHR,
    images: Vec<VkImage>,
    image_views: Vec<VkImageView>,
    render_to_present: Vec<VkSemaphore>,
}

pub struct TrackedObjects {
    pipelines: Vec<VkPipeline>,
    swapchains: Vec<SwapchainInfo>,
}

//--------------------------------------------------------------------------------------------------
// copy_indirect_1d shader params

const COPY_1D_MAX_INDIRECTIONS: usize = 8;
const COPY_1D_WORKGROUP_SIZE: u32 = 32;

/// Push constants passed to copy_indirect_1d
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
struct CopyIndirect1DParams {
    base: VkDeviceAddress,
    dst: VkDeviceAddress,
    byte_size: u32,
    count: u32,
    offset: [u32; COPY_1D_MAX_INDIRECTIONS],
}

// ---------------------------------------------------------------------------
// Global state
// ---------------------------------------------------------------------------

static DEVICE_STATE: LazyLock<DashMap<DispatchKey, Device>> = LazyLock::new(DashMap::new);
static INSTANCE_MAP: LazyLock<DashMap<VkInstance, InstanceDispatch>> = LazyLock::new(DashMap::new);
static PHY_TO_INSTANCE: LazyLock<DashMap<VkPhysicalDevice, VkInstance>> = LazyLock::new(DashMap::new);

/// Returns the [`Device`] for the given `VkDevice`.
fn device_state<T: DeviceDispatchableHandle>(handle: T) -> dashmap::mapref::one::Ref<'static, DispatchKey, Device> {
    DEVICE_STATE.get(&unsafe { handle.key() }).expect("unknown device")
}
