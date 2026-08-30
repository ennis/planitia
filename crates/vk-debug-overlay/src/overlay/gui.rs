use crate::debugger::{CaptureKind, Debugger, LoadChain};
use crate::event::EId;
use crate::overlay::renderer::{FrameData, RenderData};
use crate::spirv::{Module, ScalarType, StructType, TypeId, TypeInfo, pretty_print_type, type_byte_size};
use crate::state_tracker::command::Command;
use crate::state_tracker::pipeline::ShaderStageInfo;
use crate::{Device, ModuleId, ModuleMap, SubmissionState};
use ash::vk;
use ash::vk::Handle;
use color_print::cwrite;
use imgui::Condition::Always;
use imgui::{StyleVar, TreeNodeFlags, Ui};
use slotmap::Key;
use spirv::StorageClass;
use std::cell::RefCell;
use std::fmt::Write;
use std::hash::{Hash, Hasher};
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

static TABLE_FLAGS: imgui::TableFlags = imgui::TableFlags::from_bits(
    imgui::TableFlags::BORDERS.bits() | imgui::TableFlags::ROW_BG.bits() | imgui::TableFlags::RESIZABLE.bits(),
)
.unwrap();

static STATUS_BAR_HEIGHT: i32 = 20;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum GuiMode {
    Normal,
    PickSizeHint,
}

pub struct GuiState {
    selected: usize,
    lens_x: i32,
    lens_y: i32,

    show_raw_watches: bool,
    show_size_hints: bool,
    show_commands: bool,
    show_memory_map: bool,

    fuse_single_field_structs: bool,
    interpret_vec2u_as_descriptor_handle: bool,
    pinned_watches: Vec<PinnedWatch>,
    status: String,
    size_hints: Vec<ArraySizeHint>,

    mode: GuiMode,

    // state when choosing array size hint
    array_size_hint_source: Option<VarInfo>,
    array_size_hint_len: Option<VarInfo>,
}

impl GuiState {
    pub fn new() -> GuiState {
        GuiState {
            selected: 0,
            lens_x: 0,
            lens_y: 0,
            show_raw_watches: false,
            show_size_hints: false,
            show_commands: true,
            show_memory_map: false,
            fuse_single_field_structs: true,
            interpret_vec2u_as_descriptor_handle: false,
            pinned_watches: vec![],
            status: "".to_string(),
            size_hints: vec![],
            mode: GuiMode::Normal,
            array_size_hint_source: None,
            array_size_hint_len: None,
        }
    }
}

#[derive(Clone)]
struct ArraySizeHint {
    size_hint_var: VarInfo,
    array_var_path: String,
}

#[derive(Clone, Eq, PartialEq, Hash)]
struct PinnedWatch {
    eid: EId,
    var: VarInfo,
}

/// Represents a shader parameter variable, relative to a particular command.
#[derive(Clone, Eq, PartialEq, Hash)]
struct VarInfo {
    mid: ModuleId,
    load_chain: LoadChain,
    byte_size: usize,
    path: String,
    offset: usize,
    ty: TypeInfo,
}

// Hierarchy of contexts during GUI:
//
// RootContext (device, dbg, modules): root windows
// +- CommandContext (+cmd): command browser window
//    +- ParamWalk (+stage, +module, +load_chain, +data): command param tree

struct RootContext<'a> {
    d: &'a Device,
    rd: &'a RenderData,
    sbs: &'a mut SubmissionState,
    dbg: &'a mut Debugger,
    modules: &'a mut ModuleMap,
}

struct CommandContext<'a> {
    d: &'a Device,
    dbg: &'a mut Debugger,
    modules: &'a mut ModuleMap,
    cmd: &'a Command,
}

struct ParamDataCtx {
    load_chain: LoadChain,
    byte_size: usize,
    data: Option<Vec<u8>>, // data
}

struct ParamWalkCtx<'a> {
    d: &'a Device,
    eid: EId,
    dbg: &'a mut Debugger,
    m: &'a Module,
    mid: ModuleId,
}

struct ParamWalk<'a, 'ctx, 'data, 'ty> {
    ctx: &'a mut ParamWalkCtx<'ctx>,
    data: &'data ParamDataCtx, // data context (load chain & data)
    offset: usize,             // item offset in data
    path: String,              // item path
    ty: &'ty TypeInfo,         // item TypeInfo
}

impl<'a, 'ctx, 'data, 'ty> ParamWalk<'a, 'ctx, 'data, 'ty> {
    fn child<'b, 'ty2>(&'b mut self, offset: usize, ty: &'ty2 TypeInfo, name: &str) -> ParamWalk<'b, 'ctx, 'data, 'ty2>
    where
        'a: 'b,
    {
        ParamWalk {
            ctx: self.ctx,
            data: self.data,
            offset: self.offset + offset,
            path: format!("{}.{}", self.path, name),
            ty,
        }
    }

    /// Returns the VarInfo for the current variable.
    fn var_info(&self) -> VarInfo {
        VarInfo {
            mid: self.ctx.mid,
            load_chain: self.data.load_chain.clone(),
            byte_size: self.data.byte_size,
            path: self.path.clone(),
            offset: self.offset,
            ty: self.ty.clone(),
        }
    }
}

//fn add_array_size_hint(walk: &mut ParamWalk,

#[derive(Copy, Clone, Debug)]
enum Value {
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
}

impl Value {
    fn to_usize(&self) -> Option<usize> {
        match *self {
            Value::U8(v) => Some(v as usize),
            Value::U16(v) => Some(v as usize),
            Value::U32(v) => Some(v as usize),
            Value::U64(v) => Some(v as usize),
            //_ => None
        }
    }
}

fn fetch_value(ctx: &mut ParamWalkCtx, var_info: &VarInfo) -> Option<Value> {
    eprintln!(
        "request value {} load chain {:?} ty {}",
        var_info.path,
        var_info.load_chain,
        pretty_print_type(ctx.m, &var_info.ty)
    );
    let data = ctx.dbg.capture_load_chain(ctx.eid, &var_info.load_chain, var_info.byte_size);
    if let Some(data) = data {
        let data = &data[var_info.offset..];
        match var_info.ty {
            TypeInfo::Scalar(s) => match s {
                ScalarType::U8 => Some(Value::U8(data[0])),
                ScalarType::U16 => Some(Value::U16(u16::from_le_bytes(data[0..2].try_into().unwrap()))),
                ScalarType::U32 => Some(Value::U32(u32::from_le_bytes(data[0..4].try_into().unwrap()))),
                ScalarType::U64 => Some(Value::U64(u64::from_le_bytes(data[0..8].try_into().unwrap()))),
                _ => None,
            },
            _ => None,
        }
    } else {
        None
    }
}

fn raw_watches_window(ctx: &RootContext, ui: &Ui, st: &mut GuiState) {
    ui.window("Raw Watches")
        .size([400.0, 300.0], imgui::Condition::FirstUseEver)
        .opened(&mut st.show_raw_watches)
        .build(|| {
            if let Some(_t) = ui.begin_table_with_flags("watch_table", 6, TABLE_FLAGS) {
                ui.table_setup_column("ID");
                ui.table_setup_column("Command");
                ui.table_setup_column("Type");

                ui.table_setup_column("Size");
                ui.table_setup_column("Load Chain");
                ui.table_setup_column("Image");

                ui.table_headers_row();

                for (id, watch) in ctx.dbg.watches.iter() {
                    ui.table_next_column();
                    ui.text(format!("{}", id.data().as_ffi() & 0xFFFF_FFFF));
                    ui.table_next_column();
                    ui.text(format!("{}", watch.eid));
                    match watch.capture {
                        CaptureKind::Buffer(ref cap) => {
                            ui.table_next_column();
                            ui.text("BUFFER");
                            ui.table_next_column();
                            ui.text(format!("{}", cap.size));
                            ui.table_next_column();
                            ui.text(format!("{:?}", cap.load_chain));
                            ui.table_next_row();
                        }
                        CaptureKind::Image(ref cap) => {
                            let image_info = unsafe { ctx.d.get_private_data_ref(cap.image).unwrap() };
                            ui.table_next_column();
                            ui.text("IMAGE");
                            ui.table_next_column(); // skip size
                            ui.table_next_column(); // skip load chain
                            ui.text(format!("{:?} (VkImage {:016x})", image_info.name, cap.image.as_raw()));
                        }
                    }
                }
            }
        });
}

fn memory_map_window(ctx: &RootContext, ui: &Ui, st: &mut GuiState) {
    let mut mmap = ctx.d.addrmap.lock();

    ui.window("Memory Map").size([400.0, 300.0], imgui::Condition::FirstUseEver).opened(&mut st.show_memory_map).build(
        || {
            if let Some(_t) = ui.begin_table_with_flags("memory_map", 3, TABLE_FLAGS) {
                ui.table_setup_column("Range");
                ui.table_setup_column("Size");
                ui.table_setup_column("Buffer");
                ui.table_headers_row();
                for range in mmap.ranges.iter() {
                    ui.table_next_column();
                    ui.text(format!("{:016x} -- {:016x}", range.base, range.base + range.size));
                    ui.table_next_column();
                    ui.text(format!("{}", range.size));
                    ui.table_next_column();
                    unsafe {
                        if let Some(buf) = ctx.d.get_private_data_ref(range.handle) {
                            ui.text(format!("{} (VkBuffer 0x{:016x})", buf.name, range.handle.as_raw()));
                        } else {
                            ui.text(format!("VkBuffer 0x{:016x}", range.handle.as_raw()));
                        }
                    }
                }
            }
        },
    );
}

fn size_hints_window(ctx: &RootContext, ui: &Ui, st: &mut GuiState) {
    ui.window("Size Hints").size([400.0, 300.0], imgui::Condition::FirstUseEver).opened(&mut st.show_size_hints).build(
        || {
            if let Some(_t) = ui.begin_table_with_flags("size_hints", 3, TABLE_FLAGS) {
                ui.table_setup_column("Array Expr");
                ui.table_setup_column("Size Hint");
                ui.table_setup_column("Load Chain");

                ui.table_headers_row();

                for sh in st.size_hints.iter() {
                    ui.table_next_column();
                    ui.text(format!("{}", sh.array_var_path));
                    ui.table_next_column();
                    ui.text(format!("{}", sh.size_hint_var.path));
                    ui.table_next_column();
                    ui.text(format!("{:?}", sh.size_hint_var.load_chain));
                }
            }
        },
    );
}

fn pinned_watch_window(ctx: &mut RootContext, ui: &Ui, st: &mut GuiState, watch: &PinnedWatch) {
    // restore the ParamWalkContext
    let data = ctx.dbg.capture_load_chain(watch.eid, &watch.var.load_chain, watch.var.byte_size);
    let module = &ctx.modules[watch.var.mid];

    let mut ctx = ParamWalkCtx { d: ctx.d, eid: watch.eid, dbg: ctx.dbg, m: module, mid: watch.var.mid };
    let datactx = ParamDataCtx { load_chain: watch.var.load_chain.clone(), byte_size: watch.var.byte_size, data };
    let mut walk = ParamWalk {
        ctx: &mut ctx,
        data: &datactx,
        offset: watch.var.offset,
        path: watch.var.path.clone(),
        ty: &watch.var.ty,
    };

    if let Some(_t) = ui.begin_table_with_flags("param_table", 3, TABLE_FLAGS) {
        ui.table_setup_column("Name");
        ui.table_setup_column("Type");
        ui.table_setup_column("Value");
        ui.table_headers_row();
        param_ui(&mut walk, ui, st, &watch.var.path);
    }
}

fn pinned_watch_windows(ctx: &mut RootContext, ui: &Ui, st: &mut GuiState) {
    let watches = mem::take(&mut st.pinned_watches);
    for watch in watches.iter() {
        ui.window(&format!("{}:{}", watch.eid, watch.var.path))
            .size([400.0, 300.0], imgui::Condition::FirstUseEver)
            .build(|| pinned_watch_window(ctx, ui, st, watch));
    }
    st.pinned_watches.extend(watches);
}

// Returns whether the specified type can be expanded to child rows in the ui.
fn type_has_child_rows(st: &GuiState, ty: &TypeInfo) -> bool {
    match ty {
        TypeInfo::Void | TypeInfo::Bool | TypeInfo::Scalar(_) => false,
        TypeInfo::Vector(ScalarType::U32, 2) if st.interpret_vec2u_as_descriptor_handle => false,
        TypeInfo::Struct(sty) if sty.fields.len() == 0 => false,
        _ => true,
    }
}

fn format_value(st: &GuiState, ty: &TypeInfo, data: &[u8]) -> String {
    match *ty {
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
            if st.interpret_vec2u_as_descriptor_handle && scalar == ScalarType::U32 && count == 2 {
                let handle = u64::from_le_bytes(data[0..8].try_into().unwrap());
                return format!("desc({handle})");
            }

            let mut str = String::new();
            let elem_size = scalar.byte_size();
            for i in 0..count as usize {
                if i > 0 {
                    str.push_str(", ");
                }
                let value = format_value(st, &TypeInfo::Scalar(scalar), &data[i * elem_size..]);
                str.push_str(&value);
            }
            str
        }
        TypeInfo::Matrix { scalar, rows, cols, stride } => {
            let mut str = String::new();
            let elem_size = scalar.byte_size();
            let col_stride = stride.map(|s| s as usize).unwrap_or(elem_size * rows as usize);
            for c in 0..cols as usize {
                if c > 0 {
                    str.push_str(", ");
                }
                let mut col_str = String::new();
                for r in 0..rows as usize {
                    if r > 0 {
                        col_str.push_str(", ");
                    }
                    let value = format_value(st, &TypeInfo::Scalar(scalar), &data[c * col_stride + r * elem_size..]);
                    col_str.push_str(&value);
                }
                str.push_str(&format!("col[{}]({})", c, col_str));
            }
            str
        }
        TypeInfo::Pointer(_pointee) => {
            let device_address = u64::from_le_bytes(data[0..8].try_into().unwrap());
            format!("0x{device_address:x}")
        }
        _ => "--".to_string(),
    }
}

fn add_pinned_watch(walk: &mut ParamWalk, ui: &Ui, st: &mut GuiState) {
    st.pinned_watches.push(PinnedWatch { eid: walk.ctx.eid, var: walk.var_info() });
}

fn param_value_summary_ui(walk: &mut ParamWalk, ui: &Ui, st: &mut GuiState) {
    let Some(ref data) = walk.data.data else {
        ui.text_disabled("--");
        return;
    };
    let data = &data[walk.offset..];
    ui.text(format_value(st, walk.ty, data));
}

/*
impl<'a> ParamWalkContext<'a> {
    fn walk_pointer<R>(
        &mut self,
        offset: usize,
        byte_size: usize,
        inner: impl FnOnce(&mut ParamWalkContext) -> R,
    ) -> R {

    }
}*/

fn pointer_param_child_rows_ui(
    walk: &mut ParamWalk,
    ui: &Ui,
    st: &mut GuiState,
    pointee_type_id: TypeId,
    size_hint: Option<usize>,
) {
    let count = size_hint.unwrap_or(1);

    // Load data at pointer.
    let pointee_type = &walk.ctx.m[pointee_type_id];
    let pointee_size = type_byte_size(walk.ctx.m, pointee_type).unwrap_or(0);
    let byte_size = count * pointee_size;

    let load_chain = walk.data.load_chain.with_deref(walk.offset);
    let data = walk.ctx.dbg.capture_load_chain(walk.ctx.eid, &load_chain, byte_size);

    let child_data = ParamDataCtx { load_chain, byte_size, data };
    let mut child_walk =
        ParamWalk { ctx: walk.ctx, data: &child_data, offset: 0, path: format!("{}.$", walk.path), ty: pointee_type };

    if size_hint.is_none() {
        // no size hint, this is a pointer to a single element, show it inline
        param_child_rows_ui(&mut child_walk, ui, st, None);
    } else {
        // assume this is a pointer to an array
        array_param_child_rows_ui(&mut child_walk, ui, st, pointee_type, pointee_size, count);
    }
}

fn array_param_child_rows_ui(
    walk: &mut ParamWalk,
    ui: &Ui,
    st: &mut GuiState,
    element_type: &TypeInfo,
    stride: usize,
    count: usize,
) {
    for i in 0..count {
        let name = format!("[{i}]");
        param_ui(&mut walk.child(i * stride, element_type, &name), ui, st, &name);
    }
}

fn vector_param_child_rows_ui(walk: &mut ParamWalk, ui: &Ui, st: &mut GuiState, scalar_type: ScalarType, count: u8) {
    assert!(count <= 4);
    let elem_size = scalar_type.byte_size();
    let component_names = ["x", "y", "z", "w"];
    for i in 0..count as usize {
        param_ui(
            &mut walk.child(i * elem_size, &TypeInfo::Scalar(scalar_type), component_names[i]),
            ui,
            st,
            component_names[i],
        );
    }
}

fn matrix_param_child_rows_ui(
    walk: &mut ParamWalk,
    ui: &Ui,
    st: &mut GuiState,
    scalar_type: ScalarType,
    rows: u8,
    cols: u8,
) {
    let elem_size = scalar_type.byte_size();
    for c in 0..cols as usize {
        param_ui(
            &mut walk.child(
                c * rows as usize * elem_size,
                &TypeInfo::Vector(scalar_type, rows),
                &format!("col[{}]", c),
            ),
            ui,
            st,
            &format!("col[{}]", c),
        );
    }
}

fn struct_param_child_rows_ui(walk: &mut ParamWalk, ui: &Ui, st: &mut GuiState, ty: &StructType) {
    for field in ty.fields.iter() {
        param_ui(&mut walk.child(field.offset as usize, &walk.ctx.m[field.ty], &field.name), ui, st, &field.name);
    }
}

fn param_child_rows_ui(walk: &mut ParamWalk, ui: &Ui, st: &mut GuiState, size_hint: Option<usize>) {
    match *walk.ty {
        TypeInfo::Vector(scalar_type, count) => {
            vector_param_child_rows_ui(walk, ui, st, scalar_type, count);
        }
        TypeInfo::Matrix { scalar, rows, cols, .. } => {
            matrix_param_child_rows_ui(walk, ui, st, scalar, rows, cols);
        }
        TypeInfo::Struct(ref sty) => {
            struct_param_child_rows_ui(walk, ui, st, sty);
        }
        TypeInfo::Array { element, stride, len } => {
            let elem_ty = &walk.ctx.m[element];
            let elem_ty_size = type_byte_size(walk.ctx.m, elem_ty).unwrap();
            let stride = stride.unwrap_or(elem_ty_size as u32) as usize;
            let len = walk.ctx.m[len].as_usize().unwrap();
            array_param_child_rows_ui(walk, ui, st, elem_ty, stride, len);
        }
        TypeInfo::Pointer(pointee_type) => {
            if let Some(pointee) = pointee_type.pointee {
                pointer_param_child_rows_ui(walk, ui, st, pointee, size_hint);
            }
        }
        _ => {}
    }
}

fn begin_set_size_hint(walk: &mut ParamWalk, ui: &Ui, st: &mut GuiState) {
    st.array_size_hint_source = Some(walk.var_info());
    st.status = format!("Click on a variable to select a size hint for {}", walk.path);
    st.mode = GuiMode::PickSizeHint;
}

fn finish_set_size_hint(walk: &mut ParamWalk, ui: &Ui, st: &mut GuiState) {
    st.mode = GuiMode::Normal;

    let TypeInfo::Scalar(len_ty @ ScalarType::U16 | len_ty @ ScalarType::U32 | len_ty @ ScalarType::U64) = walk.ty
    else {
        st.status = format!("Invalid size type {}, expected u16, u32 or u64", pretty_print_type(walk.ctx.m, walk.ty));
        return;
    };

    // should be in the same command and module
    let Some(array_var) = st.array_size_hint_source.take() else {
        return;
    };
    if array_var.mid != walk.ctx.mid {
        st.status = "Size hint should be in the same shader stage".to_string();
        return;
    }

    st.status = format!("Added size hint for {} -> {}", array_var.path, walk.path);
    st.size_hints.push(ArraySizeHint { size_hint_var: walk.var_info(), array_var_path: array_var.path })
}

fn param_ui(walk: &mut ParamWalk, ui: &Ui, st: &mut GuiState, name: &str) {
    // Special case for structs with one field: display directly struct.field
    if let TypeInfo::Struct(sty) = walk.ty
        && sty.fields.len() == 1
        && st.fuse_single_field_structs
    {
        let name = format!("{}.{}", name, sty.fields[0].name);
        let ty = &walk.ctx.m[sty.fields[0].ty];
        param_ui(&mut walk.child(sty.fields[0].offset as usize, ty, &sty.fields[0].name), ui, st, &name);
        return;
    }

    // Find an applicable size hint for the variable
    let size_hint = st
        .size_hints
        .iter()
        .find(|hint| {
            // same module, and match the access path by string
            hint.size_hint_var.mid == walk.ctx.mid && hint.array_var_path == walk.path
        })
        .cloned()
        .and_then(|hint| fetch_value(walk.ctx, &hint.size_hint_var))
        .and_then(|val| val.to_usize());

    let has_child_rows = type_has_child_rows(st, walk.ty);
    ui.table_next_column();
    let _id = ui.tree_node_config(name).flags(TreeNodeFlags::DEFAULT_OPEN).leaf(!has_child_rows).push();
    if let Some(_token) = ui.begin_popup_context_item() {
        if ui.menu_item("Pin watch") {
            add_pinned_watch(walk, ui, st);
        }
        if ui.menu_item("Set size hint...") {
            begin_set_size_hint(walk, ui, st);
        }
    }
    if ui.is_item_clicked() && st.mode == GuiMode::PickSizeHint {
        finish_set_size_hint(walk, ui, st);
    }
    if ui.is_item_hovered() {
        ui.tooltip(|| {
            ui.text(format!("{}", walk.path));
            if let Some(size_hint) = size_hint {
                ui.text(format!("Size hint: {:?}", size_hint));
            }
        });
    }
    ui.table_next_column();
    ui.text(format!("{}", pretty_print_type(walk.ctx.m, walk.ty)));
    if ui.is_item_hovered() {
        ui.tooltip(|| match *walk.ty {
            TypeInfo::Array { stride, .. } => ui.text(format!("Array Stride: {stride:?}")),
            _ => {}
        });
    }
    ui.table_next_column();
    param_value_summary_ui(walk, ui, st);

    if _id.is_some() && has_child_rows {
        param_child_rows_ui(walk, ui, st, size_hint);
    }
}

// command > shader stage > root param
fn entry_point_param_ui(walk: &mut ParamWalk, ui: &Ui, st: &mut GuiState, name: &str) {
    param_ui(walk, ui, st, name);
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
            let param_info = &module[param];
            if param_info.sc != StorageClass::PushConstant {
                continue;
            }

            // this should be a pointer to push constants
            let TypeInfo::Pointer(pointee) = &module[param_info.ty] else {
                continue;
            };
            let pointee = pointee.pointee.unwrap();

            let mut wctx = ParamWalkCtx { d: ctx.d, eid: ctx.cmd.eid, dbg: ctx.dbg, m: module, mid: ep.module };
            let data_ctx = ParamDataCtx { load_chain: Default::default(), byte_size: 0, data: None };
            let mut walk = ParamWalk {
                ctx: &mut wctx,
                data: &data_ctx,
                offset: 0,
                path: format!("[{}]", entry_point.name),
                ty: &module[pointee],
            };
            entry_point_param_ui(&mut walk, ui, st, &param_info.name);
        }
    }
}

fn command_params_ui(ctx: &mut CommandContext, ui: &Ui, st: &mut GuiState) {
    if let Some(_t) = ui.begin_table_with_flags("param_table", 3, TABLE_FLAGS) {
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

fn main_menu(ui: &Ui, st: &mut GuiState) {
    ui.main_menu_bar(|| {
        ui.menu("File", || {
            ui.menu_item("Exit");
        });
        ui.menu("View", || {
            if ui.menu_item_config("Commands").selected(st.show_commands).build() {
                st.show_commands = !st.show_commands;
            }
            if ui.menu_item_config("Raw Watches").selected(st.show_raw_watches).build() {
                st.show_raw_watches = !st.show_raw_watches;
            }
            if ui.menu_item_config("Size Hints").selected(st.show_size_hints).build() {
                st.show_size_hints = !st.show_size_hints;
            }
            if ui.menu_item_config("Memory Map").selected(st.show_memory_map).build() {
                st.show_memory_map = !st.show_memory_map;
            }
        });
        ui.menu("Options", || {
            if ui.menu_item_config("Fuse single-field structs").selected(st.fuse_single_field_structs).build() {
                st.fuse_single_field_structs = !st.fuse_single_field_structs;
            }
            if ui
                .menu_item_config("Interpret vec2u as descriptor handle")
                .selected(st.interpret_vec2u_as_descriptor_handle)
                .build()
            {
                st.interpret_vec2u_as_descriptor_handle = !st.interpret_vec2u_as_descriptor_handle;
            }
        });
    });
}

fn command_browser_window(ctx: &mut RootContext, ui: &Ui, st: &mut GuiState) {
    let mut opened = st.show_commands;
    ui.window("Commands").bg_alpha(0.5).opened(&mut opened).build(|| {
        ui.text("Debug layer active");
        ui.text(format!("dear imgui version: {}", imgui::dear_imgui_version()));
        ui.text(format!("Total submissions: {}", ctx.sbs.submission_count));

        for (i_sub, sub) in ctx.sbs.subs.iter().enumerate() {
            let _guard = ui.push_id_usize(i_sub);
            for (i_cmd, cmd) in sub.commands.iter().enumerate() {
                if ui.collapsing_header(
                    format!("[{}] Submission {}, Command {}", cmd.eid, i_sub, i_cmd),
                    TreeNodeFlags::empty(),
                ) {
                    let mut cmd_ctx = CommandContext { d: ctx.d, dbg: ctx.dbg, modules: ctx.modules, cmd };
                    command_params_ui(&mut cmd_ctx, ui, st);
                }
            }
        }
    });
    st.show_commands = opened;
}

fn status_bar_ui(ctx: &mut RootContext, ui: &Ui, st: &mut GuiState) {
    let y = (ctx.rd.height - STATUS_BAR_HEIGHT) as f32;
    let w = ctx.rd.width as f32;
    let h = STATUS_BAR_HEIGHT as f32;

    {
        let _token = ui.push_style_var(StyleVar::WindowPadding([3.0, 3.0]));
        ui.window("Status Bar")
            .bg_alpha(0.5)
            .title_bar(false)
            .position([0.0, y], Always)
            .resizable(false)
            .movable(false)
            .size([w, h], Always)
            .build(|| {
                ui.text(&st.status);
            });
    }
}

impl Device {
    fn main_gui(&self, rd: &RenderData, ui: &Ui, st: &mut GuiState) {
        let mut sbs = self.submissions.lock();
        let mut dbg = self.debugger.lock();
        let mut modules = self.modules.lock();

        let mut root_ctx = RootContext { d: self, rd, dbg: &mut dbg, sbs: &mut sbs, modules: &mut modules };

        ui.dockspace_over_main_viewport();
        main_menu(ui, st);
        if st.show_raw_watches {
            raw_watches_window(&mut root_ctx, ui, st);
        }
        if st.show_size_hints {
            size_hints_window(&mut root_ctx, ui, st);
        }
        if st.show_commands {
            command_browser_window(&mut root_ctx, ui, st);
        }
        if st.show_memory_map {
            memory_map_window(&mut root_ctx, ui, st);
        }
        pinned_watch_windows(&mut root_ctx, ui, st);
        status_bar_ui(&mut root_ctx, ui, st);
    }

    pub unsafe fn render_gui(&self, rd: &RenderData, fd: &FrameData) {
        let mut st = self.gui.lock();

        with_imgui_context(|ctx| {
            let io = ctx.io_mut();
            io.config_flags |= imgui::ConfigFlags::NAV_ENABLE_KEYBOARD | imgui::ConfigFlags::DOCKING_ENABLE;
            io.display_size = [rd.width as f32, rd.height as f32];
            io.delta_time = 1.0 / 60.0;
            let ui = ctx.frame();
            self.main_gui(rd, ui, &mut st);
            self.overlay.render(self, fd, rd, ctx);
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

/*
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
}*/

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
