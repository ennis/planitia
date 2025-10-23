
///////////////////////////////////////////////////////////////////////////////////

use std::borrow::Cow;
use log::error;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop, EventLoopProxy};
use winit::window::{Window, WindowId};
use wry::WebViewBuilder;
use crate::PROJECT_DIR;

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