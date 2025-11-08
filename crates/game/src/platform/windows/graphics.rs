use crate::context::Context;
use crate::platform::windows::{D3D12CommandQueue, D3D12Device, DXGIFactory4, GpuFenceData};
use log::info;
use std::cell::{Cell, OnceCell};
use std::ffi::OsString;
use threadbound::ThreadBound;
use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
use windows::Win32::Graphics::Direct3D12::{
    D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_COMMAND_QUEUE_DESC, D3D12_FENCE_FLAG_NONE, D3D12CreateDevice,
    ID3D12CommandAllocator, ID3D12CommandQueue, ID3D12Device, ID3D12Fence,
};
use windows::Win32::Graphics::DirectComposition::{
    DCompositionCreateDevice3, IDCompositionDesktopDevice, IDCompositionDeviceDebug,
};
use windows::Win32::Graphics::Dxgi::{
    CreateDXGIFactory2, DXGI_ADAPTER_FLAG_SOFTWARE, DXGI_CREATE_FACTORY_FLAGS, IDXGIAdapter1, IDXGIFactory4,
};
use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};
use windows::core::{IUnknown, Interface, Owned};

pub(super) struct GraphicsContext {
    pub(super) adapter: IDXGIAdapter1,
    pub(super) d3d_device: D3D12Device,      // thread safe
    pub(super) cmd_queue: D3D12CommandQueue, // thread safe
    pub(super) cmd_alloc: ThreadBound<ID3D12CommandAllocator>,
    pub(super) compositor: IDCompositionDesktopDevice,
    pub(super) dxgi_factory: DXGIFactory4,
    pub(super) fence: GpuFenceData,
}

impl GraphicsContext {
    pub(super) fn new() -> Self {
        // XXX: create the vulkan device first so that it is picked up as the main API by
        //      renderdoc (otherwise it picks D3D12 which is not what we want)
        let _vk_device = gpu::Device::global();

        //=========================================================
        // DXGI Factory and adapter enumeration

        // SAFETY: the paramters are valid
        let dxgi_factory =
            unsafe { DXGIFactory4(CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS::default()).unwrap()) };

        // --- Enumerate adapters
        let mut adapters = Vec::new();
        unsafe {
            let mut i = 0;
            while let Ok(adapter) = dxgi_factory.EnumAdapters1(i) {
                adapters.push(adapter);
                i += 1;
            }
        };

        let mut chosen_adapter = None;
        let mut chosen_adapter_name = String::new();
        for adapter in adapters.iter() {
            let desc = unsafe { adapter.GetDesc1().unwrap() };
            use std::os::windows::ffi::OsStringExt;
            let name = &desc.Description[..];
            let name_len = name.iter().take_while(|&&c| c != 0).count();
            let name = OsString::from_wide(&desc.Description[..name_len])
                .to_string_lossy()
                .into_owned();
            let is_software = (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32) != 0;
            info!(
                "DXGI adapter: {} (LUID:{:08x}{:08x}{})",
                name,
                desc.AdapterLuid.HighPart,
                desc.AdapterLuid.LowPart,
                if is_software { ", software" } else { "" }
            );
            if is_software {
                continue;
            }
            if chosen_adapter.is_none() {
                chosen_adapter = Some(adapter.clone());
                chosen_adapter_name = name;
            }
        }
        let adapter = chosen_adapter.expect("no suitable video adapter found");
        info!("using DXGI adapter: {}", chosen_adapter_name);

        //=========================================================
        // D3D12 stuff

        //let debug = unsafe { DXGIGetDebugInterface1(0).unwrap() };

        let d3d_device = unsafe {
            let mut d3d12_device: Option<ID3D12Device> = None;
            D3D12CreateDevice(
                // pAdapter:
                &adapter.cast::<IUnknown>().unwrap(),
                // MinimumFeatureLevel:
                D3D_FEATURE_LEVEL_12_0,
                // ppDevice:
                &mut d3d12_device,
            )
            .expect("D3D12CreateDevice failed");
            D3D12Device(d3d12_device.unwrap())
        };

        let cmd_queue = unsafe {
            let cq: ID3D12CommandQueue = d3d_device
                .0
                .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                    Priority: 0,
                    Flags: Default::default(),
                    NodeMask: 0,
                })
                .expect("CreateCommandQueue failed");
            D3D12CommandQueue(cq)
        };

        let cmd_alloc = unsafe {
            let command_allocator = d3d_device
                .0
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                .unwrap();
            ThreadBound::new(command_allocator)
        };

        //=========================================================
        // Compositor

        let compositor: IDCompositionDesktopDevice =
            unsafe { DCompositionCreateDevice3(None).expect("failed to create composition device") };

        #[cfg(debug_assertions)]
        {
            let composition_device_debug: IDCompositionDeviceDebug = compositor.cast().unwrap();
            unsafe {
                composition_device_debug.EnableDebugCounters().unwrap();
            }
        }

        let fence = unsafe {
            let fence = d3d_device
                .CreateFence::<ID3D12Fence>(0, D3D12_FENCE_FLAG_NONE)
                .expect("CreateFence failed");
            let event = Owned::new(CreateEventW(None, false, false, None).unwrap());

            GpuFenceData {
                fence,
                event,
                value: Cell::new(0),
            }
        };

        Self {
            adapter,
            d3d_device,
            cmd_queue,
            cmd_alloc,
            compositor,
            dxgi_factory,
            fence,
        }
    }

    /// Waits for submitted GPU commands to complete.
    pub(super) fn wait_for_gpu(&self) {
        let _span = tracy_client::span!("wait_for_gpu");
        unsafe {
            let mut val = self.fence.value.get();
            val += 1;
            self.fence.value.set(val);
            self.cmd_queue
                .Signal(&self.fence.fence, val)
                .expect("ID3D12CommandQueue::Signal failed");
            if self.fence.fence.GetCompletedValue() < val {
                self.fence
                    .fence
                    .SetEventOnCompletion(val, *self.fence.event)
                    .expect("SetEventOnCompletion failed");
                WaitForSingleObject(*self.fence.event, 0xFFFFFFFF);
            }
        }
    }

    pub(super) fn current() -> &'static GraphicsContext {
        GRAPHICS_CONTEXT.with(|g| {
            *g.get_or_init(|| {
                let ctx = GraphicsContext::new();
                Box::leak(Box::new(ctx))
            })
        })
    }
}

thread_local! {
    static GRAPHICS_CONTEXT: OnceCell<&'static GraphicsContext> = OnceCell::new();
}
