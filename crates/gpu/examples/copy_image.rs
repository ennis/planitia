use image::DynamicImage;
use std::path::Path;
use std::ptr;
use std::time::Duration;

use gpu::{
    vk, BufferUsage, CommandStream, Image, ImageCopyBuffer, ImageCopyView, ImageCreateInfo, ImageDataLayout,
    ImageSubresourceLayers, ImageType, ImageUsage, MemoryLocation, Point3D, Rect3D, SemaphoreWait, SwapChain,
};
use raw_window_handle::HasWindowHandle;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowAttributes, WindowId};

fn load_image(cmd: &mut CommandStream, path: impl AsRef<Path>, usage: ImageUsage) -> Image {
    let path = path.as_ref();
    let device = cmd.device().clone();

    let dyn_image = image::open(path).expect("could not open image file");

    let (vk_format, bpp) = match dyn_image {
        DynamicImage::ImageLuma8(_) => (vk::Format::R8_UNORM, 1usize),
        DynamicImage::ImageLumaA8(_) => (vk::Format::R8G8_UNORM, 2usize),
        DynamicImage::ImageRgb8(_) => (vk::Format::R8G8B8_SRGB, 3usize),
        DynamicImage::ImageRgba8(_) => (vk::Format::R8G8B8A8_SRGB, 4usize),
        _ => unimplemented!(),
    };

    let width = dyn_image.width();
    let height = dyn_image.height();

    let mip_levels = gpu::mip_level_count(width, height);

    // create the texture
    let image = device.create_image(&ImageCreateInfo {
        memory_location: MemoryLocation::GpuOnly,
        type_: ImageType::Image2D,
        usage: usage | ImageUsage::TRANSFER_DST,
        format: vk_format,
        width,
        height,
        depth: 1,
        mip_levels,
        array_layers: 1,
        samples: 1,
    });

    let byte_size = width as u64 * height as u64 * bpp as u64;

    // create a staging buffer
    let staging_buffer = device.create_buffer(BufferUsage::TRANSFER_SRC, MemoryLocation::CpuToGpu, byte_size);

    // read image data
    unsafe {
        ptr::copy_nonoverlapping(
            dyn_image.as_bytes().as_ptr(),
            staging_buffer.as_mut_ptr() as *mut u8,
            byte_size as usize,
        );

        cmd.copy_buffer_to_image(
            ImageCopyBuffer {
                buffer: &staging_buffer,
                layout: ImageDataLayout::new(width, height),
            },
            ImageCopyView {
                image: &image,
                mip_level: 0,
                origin: vk::Offset3D { x: 0, y: 0, z: 0 },
                aspect: vk::ImageAspectFlags::COLOR,
            },
            vk::Extent3D {
                width,
                height,
                depth: 1,
            },
        );
    }

    image
}

struct VulkanWindow {
    window: Window,
    surface: gpu::vk::SurfaceKHR,
    format: gpu::vk::SurfaceFormatKHR,
    swap_chain: SwapChain,
    width: u32,
    height: u32,
}

struct App {
    device: gpu::RcDevice,
    window: Option<VulkanWindow>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let window = event_loop.create_window(WindowAttributes::default()).unwrap();
            let size = window.inner_size();
            let surface = gpu::get_vulkan_surface(window.window_handle().unwrap().as_raw());
            let surface_format = unsafe { self.device.get_preferred_surface_format(surface) };
            let mut swap_chain = unsafe {
                self.device
                    .create_swapchain(surface, surface_format, size.width, size.height)
            };

            self.window = Some(VulkanWindow {
                window,
                surface,
                format: surface_format,
                swap_chain,
                width: size.width,
                height: size.height,
            })
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, window_id: WindowId, event: WindowEvent) {
        let window = self.window.as_mut().unwrap();
        let device = &self.device;

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                window.width = size.width;
                window.height = size.height;
                unsafe {
                    self.device
                        .resize_swapchain(&mut window.swap_chain, window.width, window.height);
                }
            }

            WindowEvent::RedrawRequested => {
                // SAFETY: swapchain is valid
                let (swapchain_image, swapchain_ready) = unsafe {
                    device
                        .acquire_next_swapchain_image(&window.swap_chain, Duration::from_millis(100))
                        .unwrap()
                };

                let mut cmd = device.create_command_stream();
                let image = load_image(
                    &mut cmd,
                    "crates/graal/examples/yukari.png",
                    ImageUsage::TRANSFER_SRC | ImageUsage::SAMPLED,
                );

                let blit_w = image.size().width.min(window.width);
                let blit_h = image.size().height.min(window.height);

                cmd.blit_image(
                    &image,
                    ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    Rect3D {
                        min: Point3D { x: 0, y: 0, z: 0 },
                        max: Point3D {
                            x: blit_w as i32,
                            y: blit_h as i32,
                            z: 1,
                        },
                    },
                    &swapchain_image.image,
                    ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    Rect3D {
                        min: Point3D { x: 0, y: 0, z: 0 },
                        max: Point3D {
                            x: blit_w as i32,
                            y: blit_h as i32,
                            z: 1,
                        },
                    },
                    vk::Filter::NEAREST,
                );

                cmd.present(&[swapchain_ready.wait()], &swapchain_image).unwrap();
                device.cleanup();
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.window.request_redraw();
        }
    }
}

fn main() {
    tracing_subscriber::fmt()
        .with_target(false)
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_span_events(tracing_subscriber::fmt::format::FmtSpan::ACTIVE)
        .init();

    let event_loop = EventLoop::new().expect("failed to create event loop");
    let device = gpu::create_device().expect("failed to create device");
    let mut app = App { device, window: None };
    event_loop.run_app(&mut app).unwrap();
}
