#![feature(default_field_values)]
use color::Srgba8;
use draft::{Draft, Font, TextFormat};
use gamelib::input::InputEvent;
use gamelib::platform::RenderTargetImage;
use gamelib::render::RenderTarget;
use gamelib::{App, AppHandler, WindowCreateInfo, WindowHandle};
use gpu::vulkan::VK_FORMAT_D32_SFLOAT_S8_UINT;
use math::{IRect, ivec2, vec2};

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;

static APP: App<Demo> = App::new();

struct Demo {
    width: u32,
    height: u32,
    depth_buffer: RenderTarget,
}

impl Default for Demo {
    fn default() -> Self {
        Self::new()
    }
}

impl Demo {
    fn new() -> Demo {
        Demo {
            width: WIDTH,
            height: HEIGHT,
            depth_buffer: RenderTarget::new(VK_FORMAT_D32_SFLOAT_S8_UINT, gpu::ImageUsage::DEPTH_STENCIL_ATTACHMENT),
        }
    }

    fn draw_demo(&mut self, color_target: &gpu::Image) {
        let depth_target = self.depth_buffer.image();
        gpu::clear_depth_image(depth_target, 1.0);
        gpu::clear_image(color_target, gpu::ClearColorValue::Float([0.1, 0.2, 0.4, 1.0]));
        draft::draw(color_target, Some(depth_target), |draft| {
            draft.set_viewport_transform(IRect::from_size(ivec2(self.width as i32, self.height as i32)));
            draft.draw_text(
                vec2(0.0, 0.0),
                "Hello, world!",
                &TextFormat { font: Font::default_regular().clone(), size: 32.0, color: Srgba8::WHITE },
                Srgba8::WHITE,
            );
        });
    }
}

impl AppHandler for Demo {
    fn started(&mut self) {
        self.width = WIDTH;
        self.height = HEIGHT;
        let _ = gamelib::create_window(&WindowCreateInfo { width: WIDTH, height: HEIGHT, title: "gamelib demo", .. });
    }

    fn input(&mut self, _window: WindowHandle, input_event: &InputEvent) {
        if input_event.is_shortcut("Escape") {
            gamelib::quit();
        }
    }

    fn resized(&mut self, _window: WindowHandle, width: u32, height: u32) {
        self.width = width;
        self.height = height;
        self.depth_buffer.setup(width, height);
    }

    fn render(&mut self, _window: WindowHandle, target: RenderTargetImage) {
        self.draw_demo(target.image);
    }

    fn close_requested(&mut self, _window: WindowHandle) {
        gamelib::quit();
    }
}

fn main() {
    gamelib::register_asset_directory();
    APP.run(&Default::default());
}
