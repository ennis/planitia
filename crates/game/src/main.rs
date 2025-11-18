#![expect(unused, reason = "noisy")]
#![feature(default_field_values)]

use gamelib::asset::{AssetCache, Handle};
use gamelib::camera_control::{CameraControl, CameraControlInput};
use gamelib::context::{App, AppHandler, LoopHandler};
use gamelib::egui::Color32;
use gamelib::egui;
use gamelib::input::{InputEvent, PointerButton};
use gamelib::paint::{DrawGlyphRunOptions, PaintRenderParams, PaintScene, Painter, TextFormat, TextLayout};
use gamelib::pipeline_cache::get_graphics_pipeline;
use gamelib::platform::{EventToken, InitOptions, RenderTargetImage, UserEvent};

use color::{Srgba32, srgba32};
use egui_demo_lib::{View, WidgetGallery};
use gpu::PrimitiveTopology::TriangleList;
use gpu::{DeviceAddress, Image, RenderPassInfo, push_constants};
use log::debug;
use math::geom::Camera;

mod experiments;

/// Global application singleton.
static APP: App<Game> = App::new();

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;

#[repr(C)]
#[derive(Clone, Copy)]
struct SceneUniforms {
    view_matrix: [[f32; 4]; 4],
    proj_matrix: [[f32; 4]; 4],
    screen_size: [f32; 2],
    time: f32,
    frame: u32,
}

struct Game {
    width: u32,
    height: u32,
    demo: WidgetGallery,
    color: Color32,
    painter: Painter,
    camera_control: CameraControl,
    depth_stencil_buffer: gpu::Image,
    grid_shader: Handle<gpu::GraphicsPipeline>,
    background_shader: Handle<gpu::GraphicsPipeline>,
    frame_count: u32,
    start_time: std::time::Instant,
}

fn create_depth_buffer(width: u32, height: u32) -> Image {
    Image::new(gpu::ImageCreateInfo {
        width,
        height,
        format: gpu::Format::D32_SFLOAT_S8_UINT,
        usage: gpu::ImageUsage::DEPTH_STENCIL_ATTACHMENT | gpu::ImageUsage::TRANSFER_DST,
        ..
    })
}

impl Default for Game {
    fn default() -> Self {
        Self {
            demo: WidgetGallery::default(),
            color: Default::default(),
            painter: Painter::new(gpu::Format::R8G8B8A8_UNORM, None),
            depth_stencil_buffer: create_depth_buffer(WIDTH, HEIGHT),
            camera_control: CameraControl::default(),
            grid_shader: get_graphics_pipeline("/shaders/pipelines.parc#grid"),
            background_shader: get_graphics_pipeline("/shaders/pipelines.parc#background"),
            frame_count: 0,
            width: WIDTH,
            height: HEIGHT,
            start_time: std::time::Instant::now(),
        }
    }
}

impl Game {
    fn render_scene(
        &mut self,
        encoder: &mut gpu::RenderEncoder,
        _camera: &Camera,
        scene_uniforms: DeviceAddress<SceneUniforms>,
    ) {
        //----------------------------------
        // Draw background
        {
            let background_shader = self.background_shader.read();
            if let Ok(background_shader) = background_shader.try_get() {
                encoder.bind_graphics_pipeline(background_shader);
                encoder.push_constants(push_constants! {
                    scene_uniforms: DeviceAddress<SceneUniforms> = scene_uniforms,
                    bottom_color: Srgba32 = srgba32(20, 20, 40, 255),
                    top_color: Srgba32 = srgba32(100, 150, 255, 255)
                });
                encoder.draw(TriangleList, 0..6, 0..1);
            }
        }

        //----------------------------------
        // Draw grid
        {
            let grid_shader = self.grid_shader.read();
            if let Ok(grid_shader) = grid_shader.try_get() {
                encoder.bind_graphics_pipeline(grid_shader);
                encoder.push_constants(push_constants! {
                    scene_uniforms: DeviceAddress<SceneUniforms> = scene_uniforms,
                    grid_scale: f32 = 100.0
                });
                encoder.draw(TriangleList, 0..6, 0..1);
            }
        }

        //----------------------------------
        // Draw editor
    }

    fn render_overlay(&mut self, cmd: &mut gpu::CommandStream, target: &gpu::Image) {
        experiments::painting_test(
            &mut self.painter,
            cmd,
            target,
            Srgba32::from(self.color.to_srgba_unmultiplied()),
        );
    }
}

impl AppHandler for Game {
    fn input(&mut self, input_event: InputEvent) {
        // --- SHORTCUTS ---

        // App exit
        if input_event.is_shortcut("Ctrl+Q") {
            debug!("Quit requested via Ctrl+Q");
            APP.quit();
        }

        // Home camera
        if input_event.is_shortcut("Home") {
            debug!("Home camera");
            //self.camera_control.home();
        }

        // --- CAMERA ---
        self.camera_control.handle_input(&input_event);
    }

    fn event(&mut self, event: UserEvent) {}

    fn resized(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
        self.camera_control.resize(width, height);
        self.depth_stencil_buffer = create_depth_buffer(width, height);
    }

    fn vsync(&mut self) {}

    fn render(&mut self, target: RenderTargetImage) {
        // TODO: that's not the right time to reload assets.
        //       Ideally this should be done asynchronously, on another thread, so as not
        //       to block the GUI and rendering.
        AssetCache::instance().do_reload();

        let mut cmd = gpu::CommandStream::new();
        let time = self.start_time.elapsed().as_secs_f32();
        let frame = self.frame_count;
        self.frame_count += 1;

        //-------------------------------
        // Render 3D scene
        {
            let camera = self.camera_control.camera();
            let scene_info = cmd.upload_temporary(&SceneUniforms {
                view_matrix: camera.view.to_cols_array_2d(),
                proj_matrix: camera.projection.to_cols_array_2d(),
                screen_size: camera.screen_size.as_vec2().to_array(),
                time,
                frame,
            });

            let mut encoder = cmd.begin_rendering(RenderPassInfo {
                color_attachments: &[gpu::ColorAttachment {
                    image: target.image,
                    clear_value: Some([0.0, 0.0, 0.0, 1.0]),
                }],
                depth_stencil_attachment: Some(gpu::DepthStencilAttachment {
                    image: &self.depth_stencil_buffer,
                    depth_clear_value: Some(1.0),
                    stencil_clear_value: Some(0),
                }),
            });

            self.render_scene(&mut encoder, &camera, scene_info);
            encoder.finish();
        }

        //-------------------------------
        // Render 2D overlays
        self.render_overlay(&mut cmd, &target.image);

        //-------------------------------
        // Render GUI
        APP.render_imgui(&mut cmd, &target.image);

        cmd.flush(&[target.ready], &[target.rendering_finished], None).unwrap();
    }

    fn close_requested(&mut self) {
        APP.quit();
    }

    fn imgui(&mut self, ctx: &egui::Context) {
        egui::Window::new("imgui").show(ctx, |ui| {
            ui.color_edit_button_srgba(&mut self.color);
            self.demo.ui(ui);
        });
    }
}

fn main() {
    // register asset directories
    AssetCache::register_directory(concat!(env!("CARGO_MANIFEST_DIR"), "/assets"));
    gamelib::register_asset_directory();

    APP.run(&InitOptions {
        width: WIDTH,
        height: HEIGHT,
        window_title: "Planitia",
        ..Default::default()
    });
}
