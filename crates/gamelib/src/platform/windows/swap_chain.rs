use crate::platform::RenderTargetImage;
use crate::platform::windows::SWAP_CHAIN_BUFFER_COUNT;
use crate::platform::windows::graphics::GraphicsContext;
use gpu::{Device, SyncWait, vk};
use log::warn;
use std::cell::Cell;
use std::mem::ManuallyDrop;
use windows::Win32::Foundation::{CloseHandle, GENERIC_ALL, HANDLE, HWND};
use windows::Win32::Graphics::Direct3D12::{
    D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_FENCE_FLAG_SHARED, D3D12_RESOURCE_BARRIER, D3D12_RESOURCE_BARRIER_0,
    D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_BARRIER_TYPE, D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
    D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_TRANSITION_BARRIER,
    ID3D12CommandQueue, ID3D12Fence, ID3D12GraphicsCommandList, ID3D12Resource,
};
use windows::Win32::Graphics::Dxgi::Common::{DXGI_ALPHA_MODE_IGNORE, DXGI_FORMAT, DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_R8G8B8A8_TYPELESS, DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_SAMPLE_DESC, DXGI_ALPHA_MODE_UNSPECIFIED, DXGI_ALPHA_MODE_STRAIGHT};
use windows::Win32::Graphics::Dxgi::{DXGI_ERROR_WAS_STILL_DRAWING, DXGI_PRESENT_DO_NOT_WAIT, DXGI_SCALING_STRETCH, DXGI_SWAP_CHAIN_DESC1, DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING, DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL, DXGI_USAGE_RENDER_TARGET_OUTPUT, IDXGIFactory4, IDXGISwapChain3, DXGI_SWAP_EFFECT_SEQUENTIAL, DXGI_SCALING_NONE};
use windows::core::{Interface, Owned};

/// DXGI swap chain that provides facilities for interoperation with Vulkan.
///
/// This holds a composition DXGI swap chain, whose images are imported as Vulkan images.
/// This uses VK_EXT_external_memory_win32 to import the swap chain images.
pub(super) struct DxgiVulkanInteropSwapChain {
    pub(super) dxgi_swap_chain: IDXGISwapChain3,
    /// Imported vulkan images for the swap chain buffers.
    pub(super) images: Vec<VulkanInteropImage>,
    /// Whether a swap chain surface has been acquired and not released yet.
    pub(super) acquired: Cell<bool>,

    // Fence state for synchronizing between D3D12 presentation and vulkan
    /// Vulkan side of the presentation fence
    pub(super) fence_semaphore: vk::Semaphore,
    /// D3D12 side of the presentation fence
    pub(super) fence: ID3D12Fence,
    /// Fence shared handle (imported to vulkan)
    pub(super) fence_shared_handle: HANDLE,
    /// Presentation fence value
    pub(super) fence_value: Cell<u64>,

    // frame latency waitable
    pub(super) frame_latency_waitable: Owned<HANDLE>,
}

pub(super) struct VulkanInteropImage {
    /// Shared handle to DXGI swap chain buffer.
    pub(super) shared_handle: HANDLE,
    /// Imported DXGI swap chain buffer.
    pub(super) image: gpu::Image,
    // Dummy command list for synchronization with vulkan.
    //
    // We need to push some commands to the D3D12 queue after acquiring a buffer from the swap chain and before signalling the DXGI/VK fence,
    // to force some implicit synchronization with the presentation engine.
    //
    // Suggested by a user on the DirectX discord.
    //
    // Don't remove it, we get artifacts otherwise, especially when moving the window.
    pub(super) dummy_cmd_list: ID3D12GraphicsCommandList,
}

impl Drop for DxgiVulkanInteropSwapChain {
    fn drop(&mut self) {
        let gfx = GraphicsContext::current();
        // Before releasing the buffers, we must make sure that the swap chain is not in use
        // We don't bother with setting up fences around the swap chain, we just wait for all commands to complete.
        // We could use fences to avoid unnecessary waiting, but not sure that it's worth the complication.
        gfx.wait_gpu_idle();

        unsafe {
            // Release the swap chain resources
            // FIXME: there should be a RAII wrapper for semaphores probably
            gpu::wait_idle();
            gpu::Device::global()
                .raw()
                .destroy_semaphore(self.fence_semaphore, None);
            CloseHandle(self.fence_shared_handle).unwrap();
            for img in self.images.iter() {
                CloseHandle(img.shared_handle).unwrap();
            }
        }
    }
}

impl DxgiVulkanInteropSwapChain {
    pub(super) fn new(format: DXGI_FORMAT, width: u32, height: u32, usage: gpu::ImageUsage) -> DxgiVulkanInteropSwapChain {
        let gfx = GraphicsContext::current();
        let vk_format = dxgi_to_vk_format(format);

        // create the DXGI swap chain
        let swap_chain = create_swap_chain(&gfx.dxgi_factory, &gfx.cmd_queue.0, format, width, height);
        let mut images = vec![];

        unsafe {
            let frame_latency_waitable = Owned::new(swap_chain.GetFrameLatencyWaitableObject());
            //assert!(!frame_latency_waitable.is_invalid());

            gfx.cmd_alloc.get_ref().unwrap().Reset().unwrap();

            // wrap swap chain buffers as vulkan images
            for i in 0..SWAP_CHAIN_BUFFER_COUNT {
                // obtain the ID3D12Resource of each swap chain buffer and create a shared handle for them
                let swap_chain_buffer = swap_chain.GetBuffer::<ID3D12Resource>(i).unwrap();
                // NOTE: I'm not sure if CreateSharedHandle is supposed to work on swap chain
                //       buffers. It didn't work with D3D11 if I remember correctly, but
                //       D3D12 doesn't seem to mind. If this breaks at some point, we may work
                //       around that by using a staging texture and copying it to the swap chain
                //       on the D3D12 side.
                //       Also, I can't find the code on GitHub that I used as a reference for this.
                let shared_handle = gfx
                    .d3d_device
                    .CreateSharedHandle(&swap_chain_buffer, None, GENERIC_ALL.0, None)
                    .unwrap();

                // import the buffer to a vulkan image with memory imported from the shared handle
                let imported_image = Device::global().create_imported_image_win32(
                    &gpu::ImageCreateInfo {
                        memory_location: gpu::MemoryLocation::GpuOnly,
                        type_: gpu::ImageType::Image2D,
                        usage,
                        format: vk_format,
                        width,
                        height,
                        depth: 1,
                        mip_levels: 1,
                        array_layers: 1,
                        samples: 1,
                        ..
                    },
                    Default::default(),
                    Default::default(),
                    vk::ExternalMemoryHandleTypeFlags::D3D12_RESOURCE_KHR,
                    shared_handle.0 as vk::HANDLE,
                    None,
                );

                // Create the dummy command list that is executed just before signalling the fence
                // for synchronization from D3D12 to Vulkan. Doing "something" on the D3D side
                // before signalling the fence is necessary to properly synchronize with
                // the presentation engine.
                // In our case we just call DiscardResource on the swap chain buffer.
                let discard_cmd_list: ID3D12GraphicsCommandList = gfx
                    .d3d_device
                    .CreateCommandList(
                        0,
                        D3D12_COMMAND_LIST_TYPE_DIRECT,
                        gfx.cmd_alloc.get_ref().unwrap(),
                        None,
                    )
                    .unwrap();

                discard_cmd_list.ResourceBarrier(&[D3D12_RESOURCE_BARRIER {
                    Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                    Flags: Default::default(),
                    Anonymous: D3D12_RESOURCE_BARRIER_0 {
                        Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                            // This transmute_copy and ManuallyDrop is messed up but apparently
                            // that's the intended way. God forbid using an actual raw pointer in the API,
                            // gotta use the RAII wrappers everywhere, even where it makes no sense.
                            pResource: std::mem::transmute_copy(&swap_chain_buffer),
                            Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                            StateBefore: D3D12_RESOURCE_STATE_COMMON,
                            StateAfter: D3D12_RESOURCE_STATE_RENDER_TARGET,
                        }),
                    },
                }]);
                discard_cmd_list.DiscardResource(&swap_chain_buffer, None);
                discard_cmd_list.ResourceBarrier(&[D3D12_RESOURCE_BARRIER {
                    Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                    Flags: Default::default(),
                    Anonymous: D3D12_RESOURCE_BARRIER_0 {
                        Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                            pResource: std::mem::transmute_copy(&swap_chain_buffer),
                            Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                            StateBefore: D3D12_RESOURCE_STATE_RENDER_TARGET,
                            StateAfter: D3D12_RESOURCE_STATE_COMMON,
                        }),
                    },
                }]);
                discard_cmd_list.Close().unwrap();

                images.push(VulkanInteropImage {
                    shared_handle,
                    image: imported_image,
                    dummy_cmd_list: discard_cmd_list,
                });
            }

            // Create & share a D3D12 fence for VK/DXGI sync
            let fence = gfx.d3d_device.CreateFence(0, D3D12_FENCE_FLAG_SHARED).unwrap();
            let fence_shared_handle = gfx
                .d3d_device
                .CreateSharedHandle(&fence, None, GENERIC_ALL.0, None)
                .unwrap();
            let fence_semaphore = gpu::Device::global().create_imported_semaphore_win32(
                vk::SemaphoreImportFlags::empty(),
                vk::ExternalSemaphoreHandleTypeFlags::D3D12_FENCE,
                fence_shared_handle.0 as vk::HANDLE,
                None,
            );

            // SAFETY: no concurrent access possible just after creation
            gpu::set_debug_name_raw(fence_semaphore, "DxgiVulkanSharedFence");

            DxgiVulkanInteropSwapChain {
                dxgi_swap_chain: swap_chain,
                images,
                fence_value: Cell::new(1),
                fence_semaphore,
                fence,
                fence_shared_handle,
                acquired: Cell::new(false),
                frame_latency_waitable,
            }
        }
    }


    /// Acquires the next image from the swap chain for rendering.
    ///
    /// This must be followed by a call to `present()`.
    pub(super) fn get_image(&self) -> &gpu::Image {
        assert!(!self.acquired.get(), "surface already acquired");

        let gfx = GraphicsContext::current();
        let index = unsafe { self.dxgi_swap_chain.GetCurrentBackBufferIndex() };
        let image = &self.images[index as usize];

        let fence_value = self.fence_value.get();
        self.fence_value.set(fence_value + 1);

        // Synchronization: D3D12 -> Vulkan
        unsafe {
            // dummy rendering to synchronize with the presentation engine before signalling the fence
            // needed! there's some implicit synchronization being done here
            gfx.cmd_queue
                .ExecuteCommandLists(&[Some(image.dummy_cmd_list.cast().unwrap())]);

            gfx.cmd_queue.Signal(&self.fence, fence_value).unwrap();
            gpu::sync_wait(self.fence_semaphore, fence_value);
        }

        self.acquired.set(true);

        &image.image
    }

    /// Presents the image that was last acquired with `get_image()`.
    pub(super) fn present(&self) {
        let gfx = GraphicsContext::current();
        let fence_value = self.fence_value.get();
        self.fence_value.set(fence_value + 1);

        // Synchronization: Vulkan -> D3D12
        unsafe {
            // Synchronize with vulkan rendering before presenting
            gpu::sync_signal(self.fence_semaphore, fence_value);
            gfx.cmd_queue.Wait(&self.fence, fence_value).unwrap();

            // Present the image.
            //
            // NOTE: the Vulkan layout of the image should be GENERAL before being handed off to
            // D3D12 via external memory objects, according to VK_KHR_external_memory spec:
            //
            //       [...] Other APIs will fall into two categories: Those that are
            //       Vulkan-compatible, and those that are Vulkan-incompatible.
            //       Vulkan-incompatible APIs will require the image to be in the GENERAL layout
            //       whenever they are accessing them.
            //
            // FIXME: Make sure that the Vulkan layout is GENERAL before the handoff.
            //        This won't be a problem once we get rid of layout transitions and use GENERAL
            //        everywhere.
            let r = self.dxgi_swap_chain.Present(0, DXGI_PRESENT_DO_NOT_WAIT);
            if r == DXGI_ERROR_WAS_STILL_DRAWING {
                //warn!("[IDXGISwapChain::Present] DXGI_ERROR_WAS_STILL_DRAWING");
            }
        }

        self.acquired.set(false);
    }
}

/// Creates a DXGI swap chain for use with DirectComposition (`CreateSwapChainForComposition`).
///
/// The swap chain is created with the following parameters:
/// * `AlphaMode = DXGI_ALPHA_MODE_IGNORE`
/// * `BufferCount = 2`
/// * `Scaling = DXGI_SCALING_STRETCH`
/// * `SwapEffect = DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL`
/// * No multisampling (`SampleDesc.Count = 1`)
///
/// # Arguments
///
/// * `width` - width in physical pixels of the swap chain buffers.
/// * `height` - height in physical pixels of the swap chain buffers.
///
/// # Panics
///
/// Panics if `width` or `height` are zero (zero-sized swap chains are not supported).
fn create_swap_chain(
    dxgi_factory: &IDXGIFactory4,
    command_queue: &ID3D12CommandQueue,
    dxgi_format: DXGI_FORMAT,
    width: u32,
    height: u32,
    //mode: SwapChainMode,
) -> IDXGISwapChain3 {
    // CreateSwapChainForComposition fails if width or height are zero.
    // Catch this early to avoid a cryptic error message from the system.
    assert!(
        width != 0 && height != 0,
        "swap chain width and height must be non-zero"
    );

    // NOTE: using too few buffers can lead to contention on the present queue since frames
    // must wait for a buffer to be available.
    //
    // 3 buffers is the minimum to avoid contention.

    // SAFETY: FFI calls
    unsafe {
        // Create the swap chain.
        let swap_chain = dxgi_factory
            .CreateSwapChainForComposition(
                command_queue,
                &DXGI_SWAP_CHAIN_DESC1 {
                    Width: width,
                    Height: height,
                    Format: dxgi_format,
                    Stereo: false.into(),
                    SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                    BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
                    BufferCount: SWAP_CHAIN_BUFFER_COUNT,
                    Scaling: DXGI_SCALING_STRETCH,
                    SwapEffect: DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL,
                    AlphaMode: DXGI_ALPHA_MODE_IGNORE,
                    Flags: DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING.0 as u32,
                },
                None,
            )
            .expect("CreateSwapChainForComposition failed")
            // This shouldn't fail (IDXGISwapChain3 is DXGI 1.4 / Windows 10)
            .cast::<IDXGISwapChain3>()
            .unwrap();
        swap_chain

        /*swap_chain = dxgi_factory
            .CreateSwapChainForHwnd(
                command_queue,
                hwnd,
                &DXGI_SWAP_CHAIN_DESC1 {
                    Width: width,
                    Height: height,
                    Format: dxgi_format,
                    Stereo: false.into(),
                    SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                    BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
                    BufferCount: SWAP_CHAIN_BUFFER_COUNT,
                    Scaling: DXGI_SCALING_STRETCH,
                    SwapEffect: DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL,
                    AlphaMode: DXGI_ALPHA_MODE_IGNORE,
                    Flags: 0,
                },
                None,
                None,
            )
            .expect("CreateSwapChainForHwnd failed")
            // This shouldn't fail (IDXGISwapChain3 is DXGI 1.4 / Windows 10)
            .cast::<IDXGISwapChain3>()
            .unwrap();*/

    }
}

pub(super) fn dxgi_to_vk_format(format: DXGI_FORMAT) -> vk::Format {
    match format {
        DXGI_FORMAT_R8G8B8A8_TYPELESS => vk::Format::R8G8B8A8_UNORM,
        DXGI_FORMAT_R8G8B8A8_UNORM => vk::Format::R8G8B8A8_UNORM,
        DXGI_FORMAT_B8G8R8A8_UNORM => vk::Format::B8G8R8A8_UNORM,
        DXGI_FORMAT_R16G16B16A16_FLOAT => vk::Format::R16G16B16A16_SFLOAT,
        _ => panic!("Unsupported DXGI format: {:?}", format),
    }
}
