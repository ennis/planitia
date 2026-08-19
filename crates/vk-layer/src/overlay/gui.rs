use crate::overlay::renderer::{OverlayRenderer, SHUFFLE_ONE, SHUFFLE_TEX1_NOALPHA};
use crate::{DeviceState, SubmissionState, TrackedResources};
use color_print::{cwrite, cwriteln};
use std::fmt::Write;
use std::slice::from_raw_parts;
use crate::reflection::EntryPoint;

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

    pub fn draw_imgui(&self, ui: &mut imgui::Ui) {
        ui.window("Debug Layer")
            .size([400.0, 300.0], imgui::Condition::FirstUseEver)
            .build(|| {
                let sbs = self.submissions.lock();
                let n_cmd = sbs.subs.iter().map(|sub| sub.commands.len()).sum::<usize>();
                ui.text(format!("Debug layer active"));
                ui.text(format!("Total commands: {}", n_cmd));
            });
    }

    pub unsafe fn draw_gui(&self, ovr: &mut OverlayRenderer, trk: &TrackedResources, sbs: &SubmissionState) {
        let mut gui = self.gui.lock();
        let input = self.input.lock();

        let pad = 30;
        ovr.set_text_offset(pad + 10, pad + 10);
        ovr.draw_quad(
            pad,
            pad,
            ovr.rd.width - pad,
            ovr.rd.height - pad,
            0.0,
            0.0,
            1.0,
            1.0,
            [0, 0, 0, 80],
            SHUFFLE_ONE,
        );

        let n_cmd = sbs.subs.iter().map(|sub| sub.commands.len()).sum::<usize>();

        //{
        //    let ctrl = input.key_down(KeyControl);
        //    if ctrl {
        //        if input.pressed(KeyDown) {
        //            gui.lens_y += 1;
        //        } else if input.pressed(KeyUp) {
        //            gui.lens_y -= 1;
        //        } else if input.pressed(KeyLeft) {
        //            gui.lens_x -= 1;
        //        } else if input.pressed(KeyRight) {
        //            gui.lens_x += 1;
        //        }
        //    } else {
        //        if input.pressed(KeyDown) {
        //            gui.selected = (gui.selected + 1) % n_cmd;
        //        } else if input.pressed(KeyUp) {
        //            gui.selected = (gui.selected + n_cmd - 1) % n_cmd;
        //        }
        //    }
        //}

        cwriteln!(ovr, "<red>Debug layer active</>");

        let mut i_sub_cmd = 0;
        let mut sel_sub = 0;
        let mut sel_cmd = 0;
        let mut sel_pipeline_data = None;
        for (i_sub, sub) in sbs.subs.iter().enumerate() {
            for (i_cmd, cmd) in sub.commands.iter().enumerate() {
                let pipeline_data = self.get_private_data_ref(cmd.key.pipeline).unwrap();
                if i_sub_cmd == gui.selected {
                    cwriteln!(ovr, " * sub({i_sub})cmd({i_cmd}) {}", pipeline_data.name);
                    sel_cmd = i_cmd;
                    sel_sub = i_sub;
                    sel_pipeline_data = Some(pipeline_data);
                } else {
                    cwriteln!(ovr, "   sub({i_sub})cmd({i_cmd}) {}", pipeline_data.name);
                }
                i_sub_cmd += 1;
            }
        }

        {

            let mut print_ep = |kind: &str, ep: Option<&EntryPoint>| {
                if let Some(ep) = ep {
                    for param in ep.params.iter() {
                        cwriteln!(ovr, "      {kind} param: {}: {:?}", param.name, param.ty.pretty());
                    }
                }
            };

            if let Some(pdata) = sel_pipeline_data {
                print_ep("vertex", pdata.vertex);
                print_ep("fragment", pdata.fragment);
                print_ep("compute", pdata.compute);
                print_ep("mesh", pdata.mesh);
                print_ep("task", pdata.task);
            }
        }

        //if input.pressed(KeyCode::KeyDown) {}

        // show zoomed-in pixels
        {
            let cx = gui.lens_x + ovr.rd.width / 2;
            let cy = gui.lens_y + ovr.rd.height / 2;
            let scale = 10;
            let ws = 10;

            let hws = ws / 2;
            let (u0, v0) = ovr.texel2uv(cx - hws, cy - hws);
            let (u1, v1) = ovr.texel2uv(cx + hws, cy + hws);
            let x0 = cx - hws * scale;
            let x1 = cx + hws * scale;
            let y0 = cy - hws * scale;
            let y1 = cy + hws * scale;

            ovr.draw_quad(x0, y0, x1, y1, u0, v0, u1, v1, [255, 255, 255, 255], SHUFFLE_TEX1_NOALPHA);
        }

        for sub in sbs.subs.iter() {
            for cmd in sub.commands.iter() {
                let pipeline_data = unsafe { self.get_private_data_ref(cmd.key.pipeline).unwrap() };
                writeln!(ovr, "{}", pipeline_data.name);
                ovr.print(&hexdump(&cmd.push, 16, 8));

                if let Some(readback) = cmd.readback.as_ref() {
                    writeln!(ovr, "----------------------------------------");
                    unsafe {
                        let data = from_raw_parts(readback.host_addr, 8);
                        ovr.print(&hexdump(data, 16, 8));
                    }
                }

                writeln!(ovr);
            }
        }
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
