#![expect(unused, reason = "noisy")]
#![feature(default_field_values)]

// debug! and error! macros are used frequently enough so that it's convenient to have them available
// without having to import them in every file.
#[macro_use]
extern crate log;

use color::{Srgba8, srgba8};
use gamelib::asset::{AssetCache, Handle};
use gamelib::camera_control::{CameraControl, CameraControlInput};
use gamelib::input::{InputEvent, PointerButton};
use gamelib::paint::{DrawGlyphRunOptions, PaintScene, Painter, TextFormat, TextLayout};
use gamelib::platform::RenderTargetImage;
use gamelib::render::pipeline_cache::get_graphics_pipeline;
use gamelib::{App, AppHandler, PluginHost, UserEvent, WindowCreateInfo, WindowHandle, imgui};
use std::ops::Deref;
use std::path::Path;
//use egui_demo_lib::{View, WidgetGallery};
use crate::project::Project;
use crate::scene::{SceneInfo, SceneInfoUniforms};
use gamelib::imgui::Condition::Always;
use gamelib::imgui::StyleVar;
use gpu::PrimitiveTopology::TriangleList;
use gpu::vulkan::*;
use gpu::{Image, Ptr, PushDataSource, root_params};
use log::debug;
use math::{Camera, Mat4, Vec2, Vec3, rect_xywh, vec2};
use ron::ser::PrettyConfig;
use serde::{Deserialize, Serialize};
use crate::gui::Gui;

mod error;
mod experiments;
mod gui;
mod project;
mod scene;
mod shader;

/// Global application singleton.
pub(crate) static APP: App<Game> = App::new();

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;

/// Application configuration.
#[derive(Clone, Serialize, Deserialize)]
struct Config {
    show_grid: bool,
    show_painting_demo: bool,
    show_background: bool,
    show_imgui: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self { show_grid: false, show_painting_demo: false, show_background: false, show_imgui: false }
    }
}

const CONFIG_FILE: &str = "config.ron";

impl Config {
    fn load() -> Self {
        let ron_str = std::fs::read_to_string(CONFIG_FILE).unwrap_or_default();
        ron::from_str(&ron_str).unwrap_or_default()
    }

    fn save(&self) {
        if let Ok(ron_str) = ron::ser::to_string_pretty(self, PrettyConfig::default()) {
            let _ = std::fs::write(CONFIG_FILE, ron_str);
        }
    }
}

struct Game {
    slang_global_session: slang::GlobalSession,
    width: u32,
    height: u32,
    cfg: Config,
    gui: Gui,
    project: Project,
    camera_control: CameraControl,
    depth_stencil_buffer: gpu::Image,
    grid_shader: Handle<gpu::GraphicsPipeline>,
    background_shader: Handle<gpu::GraphicsPipeline>,
    frame_count: u32,
    start_time: std::time::Instant,
    /// Current message in the status bar.
    status: String,
}

//--------------------------------------------------------------------------------------------------

impl Default for Game {
    fn default() -> Self {
        Self {
            slang_global_session: slang::GlobalSession::new().unwrap(),
            depth_stencil_buffer: create_depth_buffer(WIDTH, HEIGHT),
            camera_control: CameraControl::default(),
            grid_shader: get_graphics_pipeline("/shaders/grid.sharc"),
            background_shader: get_graphics_pipeline("/shaders/background.sharc"),
            frame_count: 0,
            width: WIDTH,
            height: HEIGHT,
            start_time: std::time::Instant::now(),
            cfg: Config::load(),
            status: String::new(),
            project: Default::default(),
            gui: Default::default(),
        }
    }
}

impl Game {
    fn render_scene(&mut self, target: &Image, scene_info: &SceneInfo) {
        gpu::render(
            &[gpu::ColorAttachment { image: target, clear: Some([0.0, 0.0, 0.0, 1.0]) }],
            Some(gpu::DepthStencilAttachment {
                image: &self.depth_stencil_buffer,
                depth_clear: Some(1.0),
                stencil_clear: Some(0),
            }),
            |encoder| {
                // Draw background
                //let bottom_color = Srgba8::from(self.bg_bottom_color.to_srgba_unmultiplied());
                //let top_color = Srgba8::from(self.bg_top_color.to_srgba_unmultiplied());
                if self.cfg.show_background {
                    if let Ok(background_shader) = self.background_shader.read() {
                        encoder.bind_graphics_pipeline(&*background_shader);
                        encoder.draw(
                            TriangleList,
                            None,
                            0..6,
                            0..1,
                            root_params! {
                                scene_uniforms: Ptr<SceneInfoUniforms> = scene_info.gpu,
                                bottom_color: Srgba8 = Srgba8::new(0x19, 0x19, 0x70, 0xFF),
                                top_color: Srgba8 = Srgba8::new(0x64, 0x7A, 0xEE, 0xFF)
                            },
                        );
                    }
                }

                // Draw grid
                if self.cfg.show_grid {
                    if let Ok(grid_shader) = self.grid_shader.read() {
                        encoder.bind_graphics_pipeline(&*grid_shader);
                        encoder.draw(
                            TriangleList,
                            None,
                            0..6,
                            0..1,
                            root_params! {
                                scene_uniforms: Ptr<SceneInfoUniforms> = scene_info.gpu,
                                grid_scale: f32 = 100.0
                            },
                        );
                    }
                }
            },
        );
    }

    fn render_overlay(&mut self, target: &gpu::Image) {
        let _span = gamelib::span!("render_overlay");
        let mut scene = PaintScene::new(Srgba8::TRANSPARENT);
        scene.draw_text(
            vec2(target.width() as f32 - 340.0, 10.0),
            concat!(
                "        [Home] Home camera\n",
                "           [G] Toggle grid\n",
                "           [H] Toggle background\n",
                "           [P] Toggle painting demo\n",
                "           [I] Toggle imgui\n",
                "[Ctrl+Shift+G] Reload plugin\n",
            ),
            &TextFormat { size: 20.0, ..Default::default() },
            Srgba8::WHITE,
        );
        scene.render(target);
    }
}

impl AppHandler for Game {
    fn started(&mut self) {
        // create the window
        let _ = gamelib::create_window(&WindowCreateInfo { width: WIDTH, height: HEIGHT, title: "Planitia", .. });
    }

    fn input(&mut self, _window: WindowHandle, input_event: &InputEvent) {
        // --- SHORTCUTS ---

        // App exit
        if input_event.is_shortcut("Ctrl+Q") {
            debug!("Quit requested via Ctrl+Q");
            gamelib::quit();
        }

        // Home camera
        if input_event.is_shortcut("Home") {
            debug!("Home camera");
            //self.camera_control.home();
        }

        if input_event.is_shortcut("G") {
            self.cfg.show_grid = !self.cfg.show_grid;
        }
        if input_event.is_shortcut("H") {
            self.cfg.show_background = !self.cfg.show_background;
        }
        if input_event.is_shortcut("P") {
            self.cfg.show_painting_demo = !self.cfg.show_painting_demo;
        }
        if input_event.is_shortcut("I") {
            self.cfg.show_imgui = !self.cfg.show_imgui;
        }

        // --- CAMERA ---
        if self.camera_control.handle_input(&input_event) {
            return;
        }
    }

    fn event(&mut self, event: UserEvent) {}

    fn resized(&mut self, window: WindowHandle, width: u32, height: u32) {
        self.width = width;
        self.height = height;
        self.camera_control.resize(width, height);
        self.depth_stencil_buffer = create_depth_buffer(width, height);
    }

    fn imgui(&mut self, ui: &mut imgui::Ui) {
        self.main_gui(ui);
    }

    fn vsync(&mut self) {}

    fn render(&mut self, _window: WindowHandle, target: RenderTargetImage) {
        let time = self.start_time.elapsed().as_secs_f32();
        let frame = self.frame_count;
        self.frame_count += 1;
        let scene_info = SceneInfo::new(&self.camera_control.camera(), time, frame);
        self.render_scene(target.image, &scene_info);
        gpu::flush();
    }

    fn file_changed(&mut self, _path: &Path) {}

    fn close_requested(&mut self, _window: WindowHandle) {
        gamelib::quit();
    }

    fn exiting(&mut self) {
        self.cfg.save();
    }
}

//--------------------------------------------------------------------------------------------------

fn create_depth_buffer(width: u32, height: u32) -> Image {
    Image::new(gpu::ImageCreateInfo {
        width,
        height,
        format: VK_FORMAT_D32_SFLOAT_S8_UINT,
        usage: gpu::ImageUsage::DEPTH_STENCIL_ATTACHMENT | gpu::ImageUsage::SAMPLED,
        ..
    })
}

//--------------------------------------------------------------------------------------------------
fn main() {
    APP.run(&Default::default());
}
