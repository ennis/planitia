use crate::{gpu_span, WindowInputState, span};
use crate::platform::RenderTargetImage;
use crate::platform::win32::graphics::GraphicsContext;
use crate::platform::win32::swap_chain::{DxgiVulkanInteropSwapChain, dxgi_to_vk_format};
use crate::platform::win32::{Error, get_hwnd};
use crate::util::env_flag;
use gpu::vk;
use log::{error, info};
use std::cell::Cell;
use std::env;
use std::time::Duration;
use windows::Win32::Foundation::HWND;
use windows::Win32::Graphics::DirectComposition::{IDCompositionTarget, IDCompositionVisual2};
use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_FORMAT, DXGI_FORMAT_B8G8R8A8_TYPELESS, DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_R8G8B8A8_TYPELESS,
    DXGI_FORMAT_R8G8B8A8_UNORM,
};
use winit::dpi::PhysicalSize;
use winit::event_loop::ActiveEventLoop;
use winit::platform::windows::WindowAttributesExtWindows;
use winit::raw_window_handle::HasWindowHandle;
use winit::window::WindowAttributes;

struct DCompState {
    composition_target: IDCompositionTarget,
    root_visual: IDCompositionVisual2,
}

const SWAP_CHAIN_FORMAT: DXGI_FORMAT = DXGI_FORMAT_R8G8B8A8_UNORM;

/// Swap chain implementation details.
pub(super) enum SwapChainImpl {
    /// Vulkan layered over a DXGI swap chain bound to a DirectComposition visual.
    ///
    /// Use this if:
    /// - you want to make use of DXGI flip models as documented in https://learn.microsoft.com/en-us/windows/win32/direct3ddxgi/for-best-performance--use-dxgi-flip-model
    /// - you are comfortable layering things through 3 under-documented APIs, and trust
    ///   things to synchronize without issues
    DirectComposition(DxgiVulkanInteropSwapChain),
    /// Vulkan swap chains (VK_KHR_swapchain).
    Vulkan(gpu::SwapChain),
}

/// Win32 window with associated swap chain.
pub(super) struct Window {
    /// Winit window object.
    pub(super) inner: winit::window::Window,
    /// DirectComposition state, if using DirectComposition.
    dcomp: Option<DCompState>,
    /// Swap chain
    swap_chain_impl: SwapChainImpl,
    /// Index of the currently acquired swap chain image (Vulkan swap chain only).
    swap_chain_image_index: Cell<usize>,
    /// `true` before the first present, `false` afterwards.
    first_present: bool,
    /// Input state for this window.
    pub(super) input_state: WindowInputState,
}

impl Window {
    pub(super) fn new(event_loop: &ActiveEventLoop, title: &str, width: u32, height: u32) -> Result<Window, Error> {
        let use_directcomposition = env_flag("USE_DXGI");

        if use_directcomposition {
            info!("Using DXGI composition swap chain");
        } else {
            info!("Using Vulkan swap chain");
        }

        let gfx = GraphicsContext::current();
        let attrs = WindowAttributes::default()
            .with_title(title.to_string())
            .with_no_redirection_bitmap(use_directcomposition)
            .with_inner_size(winit::dpi::LogicalSize::new(width, height));
        let inner = event_loop.create_window(attrs)?;
        let hwnd = get_hwnd(inner.window_handle().unwrap().as_raw());

        unsafe {
            let mut dcomp = None;
            if use_directcomposition {
                // Create a DirectComposition target for the window.
                // SAFETY: the HWND handle is valid
                let composition_target = gfx.compositor.CreateTargetForHwnd(hwnd, false).unwrap();
                // Create the root visual and attach it to the composition target.
                // SAFETY: FFI call
                let root_visual = gfx.compositor.CreateVisual().unwrap();
                composition_target.SetRoot(&root_visual).unwrap();
                dcomp = Some(DCompState {
                    composition_target: composition_target.clone(),
                    root_visual: root_visual.clone(),
                });
            }

            let swap_chain_mode = if use_directcomposition {
                SwapChainImpl::DirectComposition(DxgiVulkanInteropSwapChain::new(
                    SWAP_CHAIN_FORMAT,
                    width,
                    height,
                    gpu::ImageUsage::COLOR_ATTACHMENT | gpu::ImageUsage::TRANSFER_DST,
                ))
            } else {
                let device = gpu::Device::instance();
                let surface = gpu::get_vulkan_surface(inner.window_handle().unwrap().as_raw());
                let swapchain = device.create_swapchain(
                    surface,
                    vk::SurfaceFormatKHR {
                        format: dxgi_to_vk_format(SWAP_CHAIN_FORMAT),
                        color_space: gpu::vk::ColorSpaceKHR::SRGB_NONLINEAR,
                    },
                    width,
                    height,
                );
                SwapChainImpl::Vulkan(swapchain)
            };

            // Bind the swap chain to the root visual.
            // NOTE: this is done after the first present instead.
            //root_visual.SetContent(&swap_chain.dxgi).unwrap();
            //gfx.compositor.Commit().unwrap();

            Ok(Window {
                inner,
                dcomp,
                first_present: true,
                swap_chain_impl: swap_chain_mode,
                swap_chain_image_index: Default::default(),
                input_state: Default::default(),
            })
        }
    }

    pub(super) fn get_swap_chain_image(&self) -> Option<&gpu::Image> {
        match &self.swap_chain_impl {
            SwapChainImpl::DirectComposition(swap_chain) => Some(swap_chain.get_image()),
            SwapChainImpl::Vulkan(swap_chain) => {
                let device = gpu::Device::instance();
                unsafe {
                    let (index, image) =
                        match device.acquire_next_swapchain_image(swap_chain, Duration::from_millis(1000)) {
                            Ok(result) => result,
                            Err(err) => {
                                error!("failed to acquire next swap chain image: {err}");
                                return None;
                            }
                        };
                    self.swap_chain_image_index.set(index);
                    Some(image)
                }
            }
        }
    }

    pub(super) fn present(&mut self) {
        let _span = span!("present");
        let _gpu_span = gpu_span!("present");
        match self.swap_chain_impl {
            SwapChainImpl::DirectComposition(ref mut swap_chain) => {
                swap_chain.present();

                // If this is the first present, the swap chain was just created and not yet
                // bound to the visual, so bind it now.
                if self.first_present {
                    let gfx = GraphicsContext::current();
                    // see https://chromium.googlesource.com/chromium/src/+/027183a19bdc7fc48d75dd046e9c8ab8be1700d5/gpu/ipc/service/direct_composition_surface_win.cc#1128
                    gfx.wait_gpu_idle();
                    self.first_present = false;

                    if let Some(dcomp) = &self.dcomp {
                        unsafe {
                            dcomp.root_visual.SetContent(&swap_chain.dxgi_swap_chain).unwrap();
                            gfx.compositor.Commit().unwrap();
                        }
                    }
                }
            }
            SwapChainImpl::Vulkan(ref mut swap_chain) => {
                gpu::present(swap_chain, self.swap_chain_image_index.get());
            }
        }
    }

    pub(super) fn resize(&mut self, width: u32, height: u32) {
        match self.swap_chain_impl {
            SwapChainImpl::DirectComposition(ref mut swap_chain) => {
                *swap_chain = DxgiVulkanInteropSwapChain::new(
                    SWAP_CHAIN_FORMAT,
                    width,
                    height,
                    gpu::ImageUsage::COLOR_ATTACHMENT | gpu::ImageUsage::TRANSFER_DST,
                );
            }
            SwapChainImpl::Vulkan(ref mut swap_chain) => {
                let device = gpu::Device::instance();
                unsafe {
                    device.resize_swapchain(swap_chain, width, height);
                }
            }
        }

        self.first_present = true;
    }
}
