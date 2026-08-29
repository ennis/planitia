use crate::Device;
use crate::debugger::{Debugger, DebuggerContext, LoadChain, ModuleId, WatchId};
use crate::overlay::renderer::{FrameData, RenderData};
use crate::spirv::{Module, ScalarType, StructType, TypeId, TypeInfo, pretty_print_type, type_byte_size};
use crate::state_tracker::command::Command;
use crate::state_tracker::pipeline::ShaderStageInfo;
use ash::vk;
use color_print::cwrite;
use imgui::{TreeNodeFlags, Ui};
use slotmap::SlotMap;
use spirv::StorageClass;
use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt::Write;
use std::mem;

thread_local! {
    pub static IMGUI: RefCell<imgui::Context> = RefCell::new(imgui::Context::create());
}

pub fn with_imgui_context<R>(f: impl FnOnce(&mut imgui::Context) -> R) -> R {
    IMGUI.with(|cell| {
        let mut context = cell.borrow_mut();
        f(&mut context)
    })
}

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

struct CommandContext<'a> {
    d: &'a Device,
    dbg: &'a mut Debugger,
    modules: &'a mut SlotMap<ModuleId, Module>,
    cmd: &'a Command,
}

// Context for fetching GPU data.
struct ParamWalkContext<'a> {
    d: &'a Device,
    dbg: DebuggerContext<'a>,
    cmd: &'a Command,
    m: &'a Module,
    load_chain: LoadChain,
    data: Option<Vec<u8>>, // data
}

/// Returns whether the specified type can be expanded to child rows in the ui.
fn type_has_child_rows(ty: &TypeInfo) -> bool {
    match ty {
        TypeInfo::Void | TypeInfo::Bool | TypeInfo::Scalar(_) => false,
        TypeInfo::Struct(sty) if sty.fields.len() == 0 => false,
        _ => true,
    }
}

fn format_value(ty: &TypeInfo, data: &[u8]) -> String {
    match ty {
        TypeInfo::Void => "void".to_string(),
        TypeInfo::Bool => format!("{}", data[0] != 0),
        TypeInfo::Scalar(scalar_type) => match scalar_type {
            ScalarType::Bool => format!("{}", data[0] != 0),
            ScalarType::I8 => format!("{}", data[0] as i8),
            ScalarType::I16 => format!("{}", i16::from_le_bytes(data[0..2].try_into().unwrap())),
            ScalarType::I32 => format!("{}", i32::from_le_bytes(data[0..4].try_into().unwrap())),
            ScalarType::I64 => format!("{}", i64::from_le_bytes(data[0..8].try_into().unwrap())),
            ScalarType::U8 => format!("{}", data[0]),
            ScalarType::U16 => format!("{}", u16::from_le_bytes(data[0..2].try_into().unwrap())),
            ScalarType::U32 => format!("{}", u32::from_le_bytes(data[0..4].try_into().unwrap())),
            ScalarType::U64 => format!("{}", u64::from_le_bytes(data[0..8].try_into().unwrap())),
            ScalarType::F32 => format!("{}", f32::from_le_bytes(data[0..4].try_into().unwrap())),
        },
        TypeInfo::Vector(scalar, count) => {
            let mut str = String::new();
            let elem_size = scalar.byte_size();
            for i in 0..*count as usize {
                if i > 0 {
                    str.push_str(", ");
                }
                let value = format_value(&TypeInfo::Scalar(*scalar), &data[i * elem_size..]);
                str.push_str(&value);
            }
            format!("vec{}<{:?}>({})", count, scalar, str)
        }
        TypeInfo::Matrix { scalar, rows, cols, stride } => {
            let mut str = String::new();
            let elem_size = scalar.byte_size();
            let col_stride = stride.map(|s| s as usize).unwrap_or(elem_size * *rows as usize);
            for c in 0..*cols as usize {
                if c > 0 {
                    str.push_str(", ");
                }
                let mut col_str = String::new();
                for r in 0..*rows as usize {
                    if r > 0 {
                        col_str.push_str(", ");
                    }
                    let value = format_value(&TypeInfo::Scalar(*scalar), &data[c * col_stride + r * elem_size..]);
                    col_str.push_str(&value);
                }
                str.push_str(&format!("col[{}]({})", c, col_str));
            }
            format!("mat{}x{}<{:?}>({})", rows, cols, scalar, str)
        }
        TypeInfo::Pointer(_pointee) => {
            let device_address = u64::from_le_bytes(data[0..8].try_into().unwrap());
            format!("{:08x}", device_address)
        }
        _ => "--".to_string(),
    }
}

fn param_value_summary_ui(ctx: &mut ParamWalkContext, ui: &Ui, st: &mut GuiState, ty: &TypeInfo, offset: usize) {
    let Some(ref data) = ctx.data else {
        ui.text_disabled("--");
        return;
    };
    let data = &data[offset..];
    ui.text(format_value(ty, data));
}

fn pointer_param_child_rows_ui(
    ctx: &mut ParamWalkContext,
    ui: &Ui,
    st: &mut GuiState,
    pointee_type: TypeId,
    offset: usize,
) {
    let prev_load_chain = ctx.load_chain.clone();
    ctx.load_chain.deref_at(offset);

    let pointee_type = &ctx.m[pointee_type];
    let byte_size = type_byte_size(ctx.m, pointee_type).unwrap_or(0);
    let data = ctx.dbg.request_data(&ctx.load_chain, byte_size);
    let prev_data = mem::replace(&mut ctx.data, data);

    param_child_rows_ui(ctx, ui, st, pointee_type, 0);
    ctx.load_chain = prev_load_chain;
    ctx.data = prev_data;
}

fn vector_param_child_rows_ui(
    ctx: &mut ParamWalkContext,
    ui: &Ui,
    st: &mut GuiState,
    scalar_type: ScalarType,
    count: u8,
    offset: usize,
) {
    assert!(count <= 4);
    let elem_size = scalar_type.byte_size();
    let component_names = ["x", "y", "z", "w"];
    for i in 0..count as usize {
        ui.table_next_column();
        ui.tree_node_config(component_names[i]).leaf(true).push();
        ui.table_next_column();
        ui.text(format!("{:?}", scalar_type));
        ui.table_next_column();
        param_value_summary_ui(ctx, ui, st, &TypeInfo::Scalar(scalar_type), offset + i * elem_size);
    }
}

fn matrix_param_child_rows_ui(
    ctx: &mut ParamWalkContext,
    ui: &Ui,
    st: &mut GuiState,
    scalar_type: ScalarType,
    rows: u8,
    cols: u8,
    offset: usize,
) {
    let elem_size = scalar_type.byte_size();
    for c in 0..cols as usize {
        ui.table_next_column();
        ui.tree_node_config(format!("col[{}]", c)).leaf(true).push();
        ui.table_next_column();
        ui.text(format!("{}", pretty_print_type(ctx.m, &TypeInfo::Vector(scalar_type, rows))));
        ui.table_next_column();
        param_value_summary_ui(
            ctx,
            ui,
            st,
            &TypeInfo::Vector(scalar_type, rows),
            offset + c * rows as usize * elem_size,
        );
    }
}

fn struct_param_child_rows_ui(ctx: &mut ParamWalkContext, ui: &Ui, st: &mut GuiState, ty: &StructType, offset: usize) {
    for field in ty.fields.iter() {
        param_ui(ctx, ui, st, &field.name, &ctx.m[field.ty], offset + field.offset as usize);
    }
}

fn param_child_rows_ui(ctx: &mut ParamWalkContext, ui: &Ui, st: &mut GuiState, ty: &TypeInfo, offset: usize) {
    match ty {
        TypeInfo::Vector(scalar_type, count) => {
            vector_param_child_rows_ui(ctx, ui, st, *scalar_type, *count, offset);
        }
        TypeInfo::Matrix { scalar, rows, cols, .. } => {
            matrix_param_child_rows_ui(ctx, ui, st, *scalar, *rows, *cols, offset);
        }
        TypeInfo::Struct(sty) => {
            struct_param_child_rows_ui(ctx, ui, st, sty, offset);
        }
        TypeInfo::Pointer(pointee_type) => {
            if let Some(pointee) = pointee_type.pointee {
                pointer_param_child_rows_ui(ctx, ui, st, pointee, offset);
            }
        }
        _ => {}
    }
}

fn param_ui(ctx: &mut ParamWalkContext, ui: &Ui, st: &mut GuiState, name: &str, ty: &TypeInfo, offset: usize) {
    ui.table_next_column();
    let _id = ui.tree_node_config(name).flags(TreeNodeFlags::DEFAULT_OPEN).leaf(!type_has_child_rows(ty)).push();
    ui.table_next_column();
    ui.text(format!("{}", pretty_print_type(ctx.m, ty)));
    ui.table_next_column();
    param_value_summary_ui(ctx, ui, st, ty, offset);

    if _id.is_some() {
        param_child_rows_ui(ctx, ui, st, ty, offset);
    }
}

// command > shader stage > root param
fn entry_point_param_ui(
    ctx: &mut ParamWalkContext,
    ui: &Ui,
    st: &mut GuiState,
    name: &str,
    ty: &TypeInfo,
    offset: usize,
) {
    param_ui(ctx, ui, st, name, ty, offset);
}

// command > shader stages [i]
fn shader_stage_row_ui(
    ctx: &mut CommandContext,
    ui: &Ui,
    st: &mut GuiState,
    stage: vk::ShaderStageFlags,
    ep: &ShaderStageInfo,
) {
    let module = &ctx.modules[ep.module];
    let entry_point = &module[ep.entry_point];

    ui.table_next_column();
    let id = ui.tree_node(&entry_point.name);
    ui.table_next_column();
    ui.text_disabled("entry point");
    ui.table_next_column();
    if id.is_some() {
        for &param in entry_point.params.iter() {
            // filter out non push-constants
            let mut dbgctx = DebuggerContext::new(ctx.d, ctx.dbg, ctx.cmd.eid, stage);
            let mut wctx = ParamWalkContext {
                d: ctx.d,
                dbg: dbgctx,
                cmd: ctx.cmd,
                m: module,
                load_chain: Default::default(),
                data: None,
            };

            let param_info = &module[param];
            if param_info.sc != StorageClass::PushConstant {
                continue;
            }

            // this should be a pointer to push constants
            let TypeInfo::Pointer(pointee) = &module[param_info.ty] else {
                continue;
            };
            let pointee = pointee.pointee.unwrap();
            entry_point_param_ui(&mut wctx, ui, st, &param_info.name, &module[pointee], 0);
        }
    }
}

fn command_params_ui(ctx: &mut CommandContext, ui: &Ui, st: &mut GuiState) {
    if let Some(_t) = ui.begin_table_with_flags(
        "param_table",
        3,
        imgui::TableFlags::BORDERS | imgui::TableFlags::ROW_BG | imgui::TableFlags::RESIZABLE,
    ) {
        ui.table_setup_column("Name");
        ui.table_setup_column("Type");
        ui.table_setup_column("Value");
        ui.table_headers_row();
        unsafe {
            if let Some(pipeline) = ctx.d.get_private_data_ref(ctx.cmd.key.pipeline) {
                if let Some(ref vertex) = pipeline.vertex {
                    shader_stage_row_ui(ctx, ui, st, vk::ShaderStageFlags::VERTEX, vertex);
                }
                if let Some(ref mesh) = pipeline.mesh {
                    shader_stage_row_ui(ctx, ui, st, vk::ShaderStageFlags::MESH_EXT, mesh);
                }
                if let Some(ref task) = pipeline.task {
                    shader_stage_row_ui(ctx, ui, st, vk::ShaderStageFlags::TASK_EXT, task);
                }
                if let Some(ref fragment) = pipeline.fragment {
                    shader_stage_row_ui(ctx, ui, st, vk::ShaderStageFlags::FRAGMENT, fragment);
                }
                if let Some(ref compute) = pipeline.compute {
                    shader_stage_row_ui(ctx, ui, st, vk::ShaderStageFlags::COMPUTE, compute);
                }
            }
        }
    }
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

    fn main_gui(&self, ui: &Ui, st: &mut GuiState) {
        let sbs = self.submissions.lock();
        let mut dbg = self.debugger.lock();
        let mut modules = self.modules.lock();

        ui.dockspace_over_main_viewport();
        ui.main_menu_bar(|| {
            ui.menu("File", || {
                ui.menu_item("Exit");
            });
        });

        ui.window("Debug Layer").bg_alpha(0.5).build(|| {
            ui.text("Debug layer active");
            ui.text(format!("dear imgui version: {}", imgui::dear_imgui_version()));
            ui.text(format!("Total submissions: {}", sbs.submission_count));

            for (isub, sub) in sbs.subs.iter().enumerate() {
                let _guard = ui.push_id_usize(isub);
                for (icmd, cmd) in sub.commands.iter().enumerate() {
                    if ui.collapsing_header(
                        format!("[{}] Submission {}, Command {}", cmd.eid, isub, icmd),
                        TreeNodeFlags::empty(),
                    ) {
                        let mut cmdctx = CommandContext { d: self, dbg: &mut dbg, modules: &mut modules, cmd };
                        command_params_ui(&mut cmdctx, ui, st);
                    }
                }
            }
        });
    }

    pub unsafe fn render_gui(&self, rd: &RenderData, fd: &FrameData) {
        let mut st = self.gui.lock();

        with_imgui_context(|ctx| {
            let io = ctx.io_mut();
            io.config_flags |= imgui::ConfigFlags::NAV_ENABLE_KEYBOARD | imgui::ConfigFlags::DOCKING_ENABLE;
            io.display_size = [rd.width as f32, rd.height as f32];
            io.delta_time = 1.0 / 60.0;
            let ui = ctx.frame();
            self.main_gui(ui, &mut st);
            self.overlay.render(self, fd, rd, ctx);
        });

        /*ui.window("Debug Layer")
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
                    let _id = ui.new_id(0);
                    if let Some(_cid) = ui.tree_node_config(format!("command {icmd}")).push() {
                        ui.next_column();
                        ui.next_column();
                        if let Some(pipeline_data) = self.get_private_data_ref(cmd.key.pipeline) {
                            if let Some(vertex) = pipeline_data.vertex {
                                ui.text("vertex");
                                ui.next_column();
                                self.entry_point_ui(ui, &vertex, &mut st);
                            }
                            if let Some(mesh) = pipeline_data.mesh {
                                ui.text("mesh");
                                ui.next_column();
                                self.entry_point_ui(ui, &mesh, &mut st);
                            }
                            if let Some(task) = pipeline_data.task {
                                ui.text("task");
                                ui.next_column();
                                self.entry_point_ui(ui, &task, &mut st);
                            }
                            if let Some(fragment) = pipeline_data.fragment {
                                ui.text("fragment");
                                ui.next_column();
                                self.entry_point_ui(ui, &fragment, &mut st);
                            }
                            if let Some(compute) = pipeline_data.compute {
                                ui.text("compute");
                                ui.next_column();
                                self.entry_point_ui(ui, &compute, &mut st);
                            }
                        }
                    }
                }
            }
            ui.columns(1, "param_columns_end", false);

        });*/
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
