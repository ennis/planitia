use clap::Parser;
use color_print::ceprintln;
use include_dir::{Dir, include_dir};
use log::error;
use std::borrow::Cow;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop, EventLoopProxy};
use winit::window::{Window, WindowId};
use wry::WebViewBuilder;

#[derive(Debug)]
enum WebViewEvent {
    DocumentTitleChanged(String),
}

struct App {
    evproxy: EventLoopProxy<WebViewEvent>,
    window: Option<Window>,
    webview: Option<wry::WebView>,
}

impl App {
    pub fn new(evproxy: EventLoopProxy<WebViewEvent>) -> Self {
        Self {
            evproxy,
            window: None,
            webview: None,
        }
    }
}

impl ApplicationHandler<WebViewEvent> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = event_loop
            .create_window(Window::default_attributes().with_title("Pipeline Editor"))
            .unwrap();
        let webview = WebViewBuilder::new()
            .with_custom_protocol("res".to_string(), move |_id, request| {
                let path = request.uri().path().strip_prefix("/").unwrap();
                eprintln!("request URI: {}", request.uri());
                eprintln!("{:?}", path);
                if let Some(file) = PROJECT_DIR.get_file(path) {
                    wry::http::Response::builder()
                        .body(Cow::Borrowed(file.contents()))
                        .unwrap()
                } else {
                    error!("file not found: {}", path);
                    wry::http::Response::builder()
                        .status(404)
                        .body(Cow::Borrowed(&[][..]))
                        .unwrap()
                }
            })
            .with_document_title_changed_handler({
                let evproxy = self.evproxy.clone();
                move |title| {
                    evproxy.send_event(WebViewEvent::DocumentTitleChanged(title)).unwrap();
                }
            })
            .with_url("res://./index.html")
            .build(&window)
            .unwrap();

        self.window = Some(window);
        self.webview = Some(webview);
    }

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, event: WebViewEvent) {
        match event {
            WebViewEvent::DocumentTitleChanged(title) => {
                self.window.as_ref().unwrap().set_title(title.as_str());
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _window_id: WindowId, event: WindowEvent) {
        if let WindowEvent::CloseRequested = event {
            event_loop.exit();
        }
    }
}

pub(crate) fn run_editor() {
    let event_loop = EventLoop::with_user_event().build().unwrap();
    let mut app = App::new(event_loop.create_proxy());
    event_loop.run_app(&mut app).unwrap();
}

static PROJECT_DIR: Dir = include_dir!("$CARGO_MANIFEST_DIR/html");

///////////////////////////////////////////////////////////////////////////////////

#[derive(Parser, Debug)]
struct Args {
    /// Path to build manifest.
    manifest_path: Option<String>,
    /// Don't print logs to stdout.
    #[clap(short, long)]
    quiet: bool,
    /// Print cargo dependency directives.
    #[clap(long)]
    emit_cargo_deps: bool,
    /// Emit shader debug information.
    #[clap(short, long)]
    debug: bool,
    /// Dump SPIR-V binaries to disk alongside the archive.
    #[clap(long)]
    dump_spirv: bool,
    /// Open graphical editor.
    #[clap(long)]
    editor: bool,
}

fn main() {
    let args = Args::parse();

    if args.editor {
        run_editor();
        return;
    } else if let Some(manifest_path) = args.manifest_path {
        let build_options = shadertool::BuildOptions {
            quiet: args.quiet,
            emit_cargo_deps: args.emit_cargo_deps,
            emit_debug_information: args.debug,
            emit_spirv_binaries: args.dump_spirv,
        };
        match shadertool::build_pipeline(&manifest_path, &build_options) {
            Ok(()) => {}
            Err(err) => {
                ceprintln!("<r,bold>error:</> {err}");
                std::process::exit(1);
            }
        }
    }
}
