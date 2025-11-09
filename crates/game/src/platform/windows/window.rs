use crate::platform::RenderTargetImage;
use crate::platform::windows::graphics::GraphicsContext;
use crate::platform::windows::swap_chain::DxgiVulkanInteropSwapChain;
use crate::platform::windows::{Error, get_hwnd};
use windows::Win32::Graphics::DirectComposition::{IDCompositionTarget, IDCompositionVisual2};
use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_FORMAT_B8G8R8A8_TYPELESS, DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_R8G8B8A8_TYPELESS,
    DXGI_FORMAT_R8G8B8A8_UNORM,
};
use winit::dpi::PhysicalSize;
use winit::event_loop::ActiveEventLoop;
use winit::platform::windows::WindowAttributesExtWindows;
use winit::raw_window_handle::HasWindowHandle;
use winit::window::WindowAttributes;

/// Win32 window with associated composition swap chain and vulkan interop.
pub(super) struct Window {
    pub(super) inner: winit::window::Window,
    pub(super) composition_target: IDCompositionTarget,
    pub(super) root_visual: IDCompositionVisual2,
    pub(super) swap_chain: DxgiVulkanInteropSwapChain,
    first_present: bool,
    last_size: PhysicalSize<u32>,
}

impl Window {
    pub(super) fn new(event_loop: &ActiveEventLoop, title: &str, width: u32, height: u32) -> Result<Window, Error> {
        let gfx = GraphicsContext::current();
        let attrs = WindowAttributes::default()
            .with_title(title.to_string())
            .with_no_redirection_bitmap(true)
            .with_inner_size(winit::dpi::LogicalSize::new(width, height));
        let inner = event_loop.create_window(attrs)?;
        let hwnd = get_hwnd(inner.window_handle().unwrap().as_raw());

        unsafe {
            // Create a DirectComposition target for the window.
            // SAFETY: the HWND handle is valid
            let composition_target = gfx.compositor.CreateTargetForHwnd(hwnd, false).unwrap();
            // Create the root visual and attach it to the composition target.
            // SAFETY: FFI call
            let root_visual = gfx.compositor.CreateVisual().unwrap();
            composition_target.SetRoot(&root_visual).unwrap();

            let swap_chain = DxgiVulkanInteropSwapChain::new(
                DXGI_FORMAT_R8G8B8A8_UNORM,
                width,
                height,
                gpu::ImageUsage::COLOR_ATTACHMENT | gpu::ImageUsage::TRANSFER_DST,
            );

            // Bind the swap chain to the root visual.
            //root_visual.SetContent(&swap_chain.dxgi).unwrap();
            //gfx.compositor.Commit().unwrap();

            Ok(Window {
                inner,
                composition_target,
                root_visual,
                swap_chain,
                first_present: true,
                last_size: Default::default(),
            })
        }
    }

    pub(super) fn get_swap_chain_image(&self) -> RenderTargetImage<'_> {
        self.swap_chain.get_image()
    }

    pub(super) fn present(&mut self) {

        self.swap_chain.present();

        // If this is the first present, the swap chain was just created and not yet
        // bound to the visual, so bind it now.
        if self.first_present {
            let gfx = GraphicsContext::current();
            // see https://chromium.googlesource.com/chromium/src/+/027183a19bdc7fc48d75dd046e9c8ab8be1700d5/gpu/ipc/service/direct_composition_surface_win.cc#1128
            gfx.wait_gpu_idle();
            self.first_present = false;
            unsafe {
                self.root_visual.SetContent(&self.swap_chain.dxgi).unwrap();
                gfx.compositor.Commit().unwrap();
            }
        }
    }

    pub(super) fn resize(&mut self, width: u32, height: u32) {
        self.swap_chain = DxgiVulkanInteropSwapChain::new(
            DXGI_FORMAT_R8G8B8A8_UNORM,
            width,
            height,
            gpu::ImageUsage::COLOR_ATTACHMENT | gpu::ImageUsage::TRANSFER_DST,
        );
        self.first_present = true;
    }
}
