use crate::overlay::renderer::{
    OverlayResources, RenderData, Shuffle, SHUFFLE_ONE, SHUFFLE_TEX1_NOALPHA, TEXID_SWAPCHAIN,
};
use crate::reflection::EntryPoint;
use crate::{DeviceState, SubmissionState, TrackedResources};
use color_print::{cwrite, cwriteln};
use imgui::{ImColor32, TextureId};
use std::fmt::Write;
use std::slice::from_raw_parts;

pub struct GuiState {
    selected: usize,
    lens_x: i32,
    lens_y: i32,
}

impl GuiState {
    pub fn new() -> GuiState {
        GuiState { selected: 0, lens_x: 0, lens_y: 0 }
    }
}

impl DeviceState {
    pub fn draw_imgui(&self, rd: &RenderData, ui: &mut imgui::Ui) {
        let mut st = self.gui.lock();

        ui.window("Debug Layer").size([400.0, 300.0], imgui::Condition::FirstUseEver).build(|| {
            let sbs = self.submissions.lock();
            let n_cmd = sbs.subs.iter().map(|sub| sub.commands.len()).sum::<usize>();
            ui.text(format!("Debug layer active"));
            ui.text(format!("Total commands: {}", n_cmd));
            {
                let cx = st.lens_x + rd.width / 2;
                let cy = st.lens_y + rd.height / 2;
                let scale = 10;
                let ws = 20;

                let hws = ws / 2;
                let (u0, v0) = rd.texel2uv(cx - hws, cy - hws);
                let (u1, v1) = rd.texel2uv(cx + hws, cy + hws);
                let x0 = 0;
                let x1 = ws * scale;
                let y0 = 0;
                let y1 = ws * scale;
                ui.get_foreground_draw_list()
                    .add_image(TEXID_SWAPCHAIN, [x0 as f32, y0 as f32], [x1 as f32, y1 as f32])
                    .uv_min([u0, v0])
                    .uv_max([u1, v1])
                    .col(0xFF_FF_FF_FF)
                    .build();
            }
        });
    }
}

fn hexdump(data: &[u8], columns: usize, colgroup: usize) -> String {
    let n = data.len();
    let mut addr = 0;
    let mut str = String::new();
    for chunk in data.chunks(columns) {
        let nck = chunk.len();
        cwrite!(str, "{addr:08x}:").unwrap();
        for byte in chunk {
            if addr % colgroup == 0 {
                str.push(' ');
            }
            cwrite!(str, "{byte:02x} ").unwrap();
            addr += 1;
        }
        for _ in nck..columns {
            if addr % colgroup == 0 {
                str.push(' ');
            }
            str.push_str("   ");
            addr += 1;
        }
        str.push_str("  ");
        for byte in chunk {
            if byte.is_ascii() && !byte.is_ascii_control() {
                cwrite!(str, "{}", *byte as char).unwrap();
            } else {
                cwrite!(str, "<K!>.</>").unwrap();
            }
        }
        writeln!(str).unwrap();
    }
    str
}
