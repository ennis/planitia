#![feature(default_field_values)]

// debug! and error! macros are used frequently enough so that it's convenient to have them available
// without having to import them in every file.
#[macro_use]
extern crate log;

use crate::editor::{Editor, EditorConfig};
use crate::node::{Document, DocumentBox};
use config::Config;
use gamelib::asset::AssetCache;
use gamelib::color::Srgba8;
use gamelib::input::{InputEvent, NamedKey};
use gamelib::paint::{Font, PaintScene};
use gamelib::platform::RenderTargetImage;
use gamelib::{App, AppHandler, WindowCreateInfo, WindowHandle};

mod config;
mod decl;
mod editor;
mod lang;
mod layout;
mod node;
mod style;

// ---------------------------------------------------------------------------
// Application singleton
// ---------------------------------------------------------------------------

/// Global application singleton.
static APP: App<AppState> = App::new();

// ---------------------------------------------------------------------------
// Initial window size
// ---------------------------------------------------------------------------

const INITIAL_WIDTH: u32 = 1280;
const INITIAL_HEIGHT: u32 = 720;

// ---------------------------------------------------------------------------
// Application state
// ---------------------------------------------------------------------------

struct AppState {
    width: u32,
    height: u32,
    frame_count: u32,
    start_time: std::time::Instant,
    cfg: Config,
    editor: Editor,
    font_size: i32 = 24,
}

impl Default for AppState {
    fn default() -> Self {
        Self::new()
    }
}

fn default_document() -> DocumentBox {
    Document::new(&lang::STRUCT)
}

impl AppState {
    fn new() -> Self {
        // Load configuration from file, or set up defaults.
        let cfg = Config::load();

        // Load editor font.
        let font = Font::load_static_font_from_bytes(include_bytes!("../assets/fonts/TX-02-Medium.otf"));

        let editor_config = EditorConfig { font, indent: 2, .. };
        let editor = Editor::new(default_document(), &editor_config);

        Self {
            width: INITIAL_WIDTH,
            height: INITIAL_HEIGHT,
            frame_count: 0,
            start_time: std::time::Instant::now(),
            cfg,
            editor,
            ..
        }
    }

    /// Renders the application.
    fn render_gui(&mut self, cmd: &mut gpu::CommandBuffer, target: &RenderTargetImage) {
        let mut scene = PaintScene::new(Srgba8::WHITE);
        self.editor.layout();
        self.editor.paint(&mut scene);
        scene.render(cmd, &target.image);
    }
}

// ---------------------------------------------------------------------------
// AppHandler implementation
// ---------------------------------------------------------------------------

impl AppHandler for AppState {
    fn started(&mut self) {
        let _ = gamelib::create_window(&WindowCreateInfo {
            width: INITIAL_WIDTH,
            height: INITIAL_HEIGHT,
            title: "Editor",
            ..
        });
        // load config
        self.cfg = Config::load();
    }

    fn input(&mut self, _window: WindowHandle, event: &InputEvent) {
        if event.is_shortcut("Ctrl+Q") {
            gamelib::quit();
        } else if event.is_shortcut("Ctrl++") {
            self.font_size += 1;
            info!("Font size: {}px", self.font_size);
            self.editor.set_font_size(self.font_size);
        } else if event.is_shortcut("Ctrl+-") {
            if self.font_size > 1 {
                self.font_size -= 1;
                info!("Font size: {}px", self.font_size);
            }
            self.editor.set_font_size(self.font_size);
        } else if event.is_shortcut("I") {
            self.cfg.show_imgui = !self.cfg.show_imgui;
        } else if event.is_shortcut(NamedKey::Enter) {
            // modify the document
            let doc = self.editor.document_mut();
            doc.root().insert_in(&lang::STRUCT::fields, doc.create_node(&lang::FIELD), 0);
            doc.apply_pending_changes();
        } else if event.is_shortcut(NamedKey::Backspace) {
            let doc = self.editor.document_mut();
            doc.root().remove_in(&lang::STRUCT::fields, 0);
            doc.apply_pending_changes();
        }
        else if event.is_shortcut("Ctrl+O") {
            //
        } else {
            // Pass input events to the editor for processing.
            self.editor.handle_input(event);
        }
    }

    fn resized(&mut self, _window: WindowHandle, width: u32, height: u32) {
        self.width = width;
        self.height = height;
    }

    fn vsync(&mut self) {
        // Trigger any per-frame logic that doesn't require a GPU render target
        // (e.g. animation ticks, physics, etc.).
        // TODO: update subsystems here.
    }

    fn render(&mut self, _window: WindowHandle, target: RenderTargetImage) {
        //#[cfg(feature = "hot_reload")]
        //AssetCache::instance().do_reload();

        let mut cmd = gpu::CommandBuffer::new();
        self.render_gui(&mut cmd, &target);
        self.frame_count += 1;
        gpu::submit(cmd).unwrap();
    }

    fn close_requested(&mut self, _window: WindowHandle) {
        gamelib::quit();
    }

    fn exiting(&mut self) {
        self.cfg.save();
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn main() {
    // Register asset directories so that assets can be loaded by VFS path.
    AssetCache::register_directory(concat!(env!("CARGO_MANIFEST_DIR"), "/assets"));
    gamelib::register_asset_directory();

    APP.run(&Default::default());
}
