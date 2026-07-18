#![feature(default_field_values)]

use std::path::PathBuf;
use std::time::Duration;
use gamelib::{FileDialogOptions, PluginEvent, PluginResult};
use gamelib::color::srgba8;
use gamelib::math::vec2;
use gamelib::paint::{ColorStop, GradientExtendMode, LinearGradientFill, PaintScene};
use gpu;

static LAST_TIME: std::sync::Mutex<Option<std::time::Instant>> = std::sync::Mutex::new(None);

#[unsafe(no_mangle)]
pub extern "C" fn plugin_entry(event: PluginEvent) -> PluginResult {
    let gpu_device_name = gpu::get_physical_device_name();

    /*match gamelib::pick_file("Image files", &["png", "jpg", "jpeg", "tiff"]) {
        Some(file) => {
            eprintln!("Picked file: {}", file.display());
        }
        None => {}
    }*/

    //eprintln!("GPU device name: {}", gpu_device_name);

    let time = std::time::Instant::now();
    let mut last_time = LAST_TIME.lock().unwrap();
    let delta = if let Some(last_time) = *last_time {
        time.duration_since(last_time)
    } else {
        Duration::ZERO
    };
    *last_time = Some(time);

    let fps = if delta.as_secs_f32() > 0.0 { 1.0 / delta.as_secs_f32() } else { 0.0 };

    gamelib::print_message(format!("GPU device name: {}\n", gpu_device_name));
    gamelib::print_message(format!("DT  : {delta:?}\nFPS : {fps:.1}\n"));

    match event {
        PluginEvent::VSync(image) => {
            let mut scene = PaintScene::new(srgba8(0, 0, 0, 255));
            scene.fill_circle(vec2(100.5, 100.5), 60.0, srgba8(255, 255, 255, 255));
            let gradient = scene.create_gradient_ramp(&[
                ColorStop { position: 0.0, color: srgba8(255, 0, 0, 255) },
                ColorStop { position: 0.5, color: srgba8(0, 255, 0, 255) },
                ColorStop { position: 1.0, color: srgba8(0, 0, 255, 255) },
            ]);
            scene.fill_circle(vec2(300.5, 100.5), 60.0, LinearGradientFill {
                start: vec2(240.5, 100.5),
                end: vec2(360.5, 100.5),
                ramp: gradient,
                extend_mode: GradientExtendMode::Repeat
            });
            scene.render(image);
        }
        _ => {}
    }

    PluginResult::WaitVSync
}
