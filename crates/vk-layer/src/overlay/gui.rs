use crate::Device;
use crate::debugger::{Debugger, PlaceExpr, Query, WatchId};
use crate::overlay::renderer::RenderData;
use crate::reflection::{EntryPoint, ScalarType, StructType, Type, TypeDesc, VectorType};
use crate::state_tracker::command::CmdKey;
use color_print::cwrite;
use imgui::Ui;
use std::collections::HashMap;
use std::fmt::Write;

pub struct GuiState {
    selected: usize,
    lens_x: i32,
    lens_y: i32,
    live_watches: HashMap<imgui::Id, WatchId>,
    prev_watches: HashMap<imgui::Id, WatchId>,
}

impl GuiState {
    pub fn new() -> GuiState {
        GuiState {
            selected: 0,
            lens_x: 0,
            lens_y: 0,
            live_watches: Default::default(),
            prev_watches: Default::default(),
        }
    }

    /*fn param_ui_node(&self, ui: &Ui, label: &str, type_desc: &TypeDesc, data: &[u8]) {
        match type_desc {
            TypeDesc::Bool => {
                let value = data.first().map(|&b| if b != 0 { "true" } else { "false" }).unwrap_or("?");
                let _t = ui.tree_node_config(label).leaf(true).push();
                ui.next_column();
                ui.text(value);
                ui.next_column();
            }
            TypeDesc::Scalar(scalar) => {
                let value = format_scalar_value(*scalar, data);
                let _t = ui.tree_node_config(label).leaf(true).push();
                ui.next_column();
                ui.text(&value);
                ui.next_column();
            }
            TypeDesc::Vector(scalar, n) => {
                let elem_size = scalar.byte_size();
                let summary: Vec<String> = (0..*n as usize)
                    .map(|i| format_scalar_value(*scalar, data.get(i * elem_size..).unwrap_or(&[])))
                    .collect();
                let token = ui.tree_node_config(label).push();
                ui.next_column();
                ui.text(format!("({})", summary.join(", ")));
                ui.next_column();
                if let Some(_t) = token {
                    let names = ["x", "y", "z", "w"];
                    for i in 0..*n as usize {
                        let _id = ui.push_id_usize(i);
                        let comp_name = names.get(i).copied().unwrap_or("?");
                        let value = format_scalar_value(*scalar, data.get(i * elem_size..).unwrap_or(&[]));
                        let _leaf = ui.tree_node_config(comp_name).leaf(true).push();
                        ui.next_column();
                        ui.text(&value);
                        ui.next_column();
                    }
                }
            }
            TypeDesc::Matrix { scalar, rows, cols, stride } => {
                let col_stride = stride.map(|s| s as usize).unwrap_or(scalar.byte_size() * *rows as usize);
                let elem_size = scalar.byte_size();
                let token = ui.tree_node_config(label).push();
                ui.next_column();
                ui.text(format!("mat{}x{}", rows, cols));
                ui.next_column();
                if let Some(_t) = token {
                    for c in 0..*cols as usize {
                        let _cid = ui.push_id_usize(c);
                        let col_label = format!("[{}]", c);
                        let col_token = ui.tree_node_config(&col_label).push();
                        ui.next_column();
                        ui.next_column();
                        if let Some(_ct) = col_token {
                            for r in 0..*rows as usize {
                                let _rid = ui.push_id_usize(r);
                                let row_label = format!("[{}]", r);
                                let offset = c * col_stride + r * elem_size;
                                let value = format_scalar_value(*scalar, data.get(offset..).unwrap_or(&[]));
                                let _leaf = ui.tree_node_config(&row_label).leaf(true).push();
                                ui.next_column();
                                ui.text(&value);
                                ui.next_column();
                            }
                        }
                    }
                }
            }
            TypeDesc::Array { element, len, stride } => {
                let elem_stride = stride.map(|s| s as usize).or_else(|| element.byte_size()).unwrap_or(0);
                let token = ui.tree_node_config(label).push();
                ui.next_column();
                ui.text(format!("[{}]", len));
                ui.next_column();
                if let Some(_t) = token {
                    for i in 0..*len as usize {
                        let _id = ui.push_id_usize(i);
                        let elem_label = format!("[{}]", i);
                        let elem_data = data.get(i * elem_stride..).unwrap_or(&[]);
                        param_ui_node(ui, &elem_label, element, elem_data);
                    }
                }
            }
            TypeDesc::Struct(s) => {
                let token = ui.tree_node_config(label).push();
                ui.next_column();
                ui.text(s.name);
                ui.next_column();
                if let Some(_t) = token {
                    for (i, field) in s.fields.iter().enumerate() {
                        let _id = ui.push_id_usize(i);
                        let field_data = data.get(field.offset as usize..).unwrap_or(&[]);
                        param_ui_node(ui, field.name, field.ty, field_data);
                    }
                }
            }
            _ => {
                // Opaque types (images, samplers, etc.) — show type name, no data
                let type_label = match type_desc {
                    TypeDesc::Void => "void",
                    TypeDesc::ImageHandle(_) => "image_handle",
                    TypeDesc::Image => "image",
                    TypeDesc::SampledImage => "sampled_image",
                    TypeDesc::Sampler => "sampler",
                    TypeDesc::Pointer(_) => "pointer",
                    TypeDesc::RuntimeArray { .. } => "[..]",
                    _ => "?",
                };
                let _t = ui.tree_node_config(label).leaf(true).push();
                ui.next_column();
                ui.text(type_label);
                ui.next_column();
            }
        }
    }

    fn param_ui<'a>(ui: &imgui::Ui, type_desc: &TypeDesc, data: &'a [u8]) -> &'a [u8] {
        match type_desc {
            TypeDesc::Struct(s) => {
                for (i, field) in s.fields.iter().enumerate() {
                    let _id = ui.push_id_usize(i);
                    let field_data = data.get(field.offset as usize..).unwrap_or(&[]);
                    param_ui_node(ui, field.name, field.ty, field_data);
                }
            }
            _ => param_ui_node(ui, "value", type_desc, data),
        }
        let consumed = type_desc.byte_size().unwrap_or(0).min(data.len());
        &data[consumed..]
    }

    fn entry_point_ui(ui: &imgui::Ui, ep: &EntryPoint, push_data: &[u8]) {
        if let Some(_eid) = ui.tree_node(ep.name) {
            ui.next_column();
            ui.next_column();
            for (i, param) in ep.params.iter().enumerate() {
                let _id = ui.push_id_usize(i);
                if let Some(_pid) = ui.tree_node(param.name) {
                    param_ui(ui, param.ty, push_data);
                }
            }
        }
    }*/
}

impl Device {
    /*fn param_ui(&self, ui: &mut Ui, state: &mut GuiState, cmd_key: &CmdKey, ty: Type, query: &Query) {
        // form the place expression for this param
        // get imgui current ID
        // look for corresponding WatchId
        // if not found, create watch and add to live map
        // otherwise, fetch watch, read data from buffer, and

        let current_id = ui.new_id(0);
        let watch_id = if let Some(id) = state.prev_watches.get(&current_id) {
            *id
        } else {
            self.add_watch(cmd_key, query.clone())
        };
        let data = self.read_watch(watch_id);
    }*/

    fn entry_point_ui(&self, ui: &mut Ui, entry_point: &EntryPoint, state: &mut GuiState, cmd_key: &CmdKey) {}

    pub unsafe fn draw_imgui(&self, rd: &RenderData, ui: &mut imgui::Ui) {
        let st = self.gui.lock();

        ui.window("Debug Layer")
            .position([20.0, 20.0], imgui::Condition::Always)
            .size([rd.width as f32 - 40.0, rd.height as f32 - 40.0], imgui::Condition::Always)
            .title_bar(false)
            .focused(true)
            .build(|| {
                let sbs = self.submissions.lock();
                let n_cmd = sbs.subs.iter().map(|sub| sub.commands.len()).sum::<usize>();
                ui.text(format!("Debug layer active"));
                ui.text(format!("Total commands: {}", n_cmd));

                //ui.set_keyboard_focus_here();

                ui.columns(2, "param_columns", true);
                for (isub, sub) in sbs.subs.iter().enumerate() {
                    let _guard = ui.push_id_usize(isub);
                    for (icmd, cmd) in sub.commands.iter().enumerate() {
                        let _guard = ui.push_id_usize(icmd);
                        let id = ui.new_id(0);
                        if let Some(_cid) = ui.tree_node_config(format!("command {icmd}")).push() {
                            ui.next_column();
                            ui.next_column();
                            if let Some(pipeline_data) = self.get_private_data_ref(cmd.key.pipeline) {
                                if let Some(vertex) = pipeline_data.vertex {
                                    //entry_point_ui(ui, &vertex, &cmd.push);
                                }
                            }
                        }
                    }
                }
                ui.columns(1, "param_columns_end", false);

                //{
                //    let cx = st.lens_x + rd.width / 2;
                //    let cy = st.lens_y + rd.height / 2;
                //    let scale = 10;
                //    let ws = 20;
                //
                //    let hws = ws / 2;
                //    let (u0, v0) = rd.texel2uv(cx - hws, cy - hws);
                //    let (u1, v1) = rd.texel2uv(cx + hws, cy + hws);
                //    let x0 = 0;
                //    let x1 = ws * scale;
                //    let y0 = 0;
                //    let y1 = ws * scale;
                //    ui.get_foreground_draw_list()
                //        .add_image(TEXID_SWAPCHAIN, [x0 as f32, y0 as f32], [x1 as f32, y1 as f32])
                //        .uv_min([u0, v0])
                //        .uv_max([u1, v1])
                //        .col(0xFF_FF_FF_FF)
                //        .build();
                //}
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

fn format_scalar_value(scalar: ScalarType, data: &mut &[u8]) -> String {
    let byte_size = scalar.byte_size();
    if byte_size < data.len() {
        return "<not enough data>".to_string();
    }
    let result = match scalar {
        ScalarType::Bool => {
            format!("{}", data[0] != 0)
        }
        ScalarType::I8 => {
            format!("{}", data[0] as i8)
        }
        ScalarType::U8 => {
            format!("{}", data[0])
        }
        ScalarType::I16 => {
            format!("{}", i16::from_le_bytes(data[0..2].try_into().unwrap()))
        }
        ScalarType::U16 => {
            format!("{}", u16::from_le_bytes(data[0..2].try_into().unwrap()))
        }
        ScalarType::I32 => {
            format!("{}", i32::from_le_bytes(data[0..4].try_into().unwrap()))
        }
        ScalarType::U32 => {
            format!("{}", u32::from_le_bytes(data[0..4].try_into().unwrap()))
        }
        ScalarType::I64 => {
            format!("{}", i64::from_le_bytes(data[0..8].try_into().unwrap()))
        }
        ScalarType::U64 => {
            format!("{}", u64::from_le_bytes(data[0..8].try_into().unwrap()))
        }
        ScalarType::F32 => {
            format!("{}", f32::from_le_bytes(data[0..4].try_into().unwrap()))
        }
    };
    *data = &data[byte_size..];
    result
}

fn format_vector_value(scalar_type: ScalarType, count: u8, data: &[u8]) -> String {
    let mut str = String::new();
    let mut data = data;
    for _ in 0..count {
        write!(str, "{},", format_scalar_value(scalar_type, &mut data));
    }
    str
}

/*
fn format_array(element_type: Type, count: usize, data: &[u8]) -> String {
    let mut str = String::new();
    let mut data = data;
    for _ in 0..count {
        write!(str, "{},", format_scalar_value(scalar_type, &mut data));
    }
    str
}
*/
