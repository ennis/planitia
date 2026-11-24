use crate::project::{ImageResource, OptimizationLevel, ShaderOptions};
use crate::shader::{CompilationState, EntryPoint, GraphicsPipeline, Pipeline, Shader};
use crate::{APP, Game};
use gamelib::imgui;
use gamelib::imgui::Condition::Always;
use gamelib::imgui::{StyleVar, TreeNodeFlags, Ui};
use gpu::GraphicsState;
use gpu::vulkan::*;
use serde::{Deserialize, Serialize};
use spirv_reflect::{ScalarType, StructType, TypeId, TypeInfo, pretty_print_type, type_byte_size};

const TABLE_FLAGS: imgui::TableFlags = imgui::TableFlags::from_bits(
    imgui::TableFlags::BORDERS.bits() | imgui::TableFlags::ROW_BG.bits() | imgui::TableFlags::RESIZABLE.bits(),
)
.unwrap();

#[derive(Serialize, Deserialize, Default)]
pub(crate) struct Gui {
    show_shaders_window: bool,
    show_images_window: bool,
    #[serde(skip)]
    status: String,
    fuse_single_field_structs: bool,
}

impl Gui {
    pub(crate) fn set_status(&mut self, status: String) {
        self.status = status;
    }

    pub(crate) fn report_error(&mut self, status: impl Into<String>, err: anyhow::Error) {
        // Set status bar message and print details to the console
        let status = status.into();
        error!("{}: {:?}", status, err);
        self.set_status(status);
    }
}

fn string_combo(ui: &Ui, label: &str, current: &mut String, options: &[&str]) {
    let mut current_index = options.iter().position(|&s| s == current.as_str()).unwrap_or(0);
    if ui.combo_simple_string(label, &mut current_index, options) {
        *current = options[current_index].to_string();
    }
}

fn enum_combo<T: Copy + Eq>(ui: &Ui, label: &str, current: &mut T, options: &[(T, &str)]) {
    let mut current_index = options.iter().position(|&(v, _)| v == *current).unwrap_or(0);
    if ui.combo_simple_string(label, &mut current_index, &options.iter().map(|&(_, s)| s).collect::<Vec<_>>()) {
        *current = options[current_index].0;
    }
}

// command > shader stages [i]
fn entry_point_ui(m: &spirv_reflect::Module, ep: &EntryPoint, ui: &Ui, st: &mut Gui) {
    let Some(ep) = m.find_entry_point(&ep.name) else { return };
    let ep = &m[ep];
    ui.table_next_column();
    let id = ui.tree_node(&ep.name);
    ui.table_next_column();
    ui.text_disabled("entry point");
    ui.table_next_column();
    if id.is_some() {
        for &param in ep.params.iter() {
            // filter out non push-constants
            let param_info = &m[param];
            if param_info.sc != spirv_reflect::StorageClass::PushConstant {
                continue;
            }
            // this should be a pointer to push constants
            let TypeInfo::Pointer(pointee) = &m[param_info.ty] else {
                continue;
            };
            let pointee = pointee.pointee.unwrap();
            let mut walk = ParamWalk { m, offset: 0, path: format!("[{}]", ep.name), ty: &m[pointee] };
            shader_param_ui(&mut walk, ui, st, &param_info.name);
        }
    }
}

fn stencil_state_ui(ui: &Ui, st: &mut Gui, state: &mut gpu::StencilOpState) {
    enum_combo(
        ui,
        "Compare Op",
        &mut state.compare_op,
        &[
            (VK_COMPARE_OP_NEVER, "Never"),
            (VK_COMPARE_OP_LESS, "Less"),
            (VK_COMPARE_OP_EQUAL, "Equal"),
            (VK_COMPARE_OP_LESS_OR_EQUAL, "LessOrEqual"),
            (VK_COMPARE_OP_GREATER, "Greater"),
            (VK_COMPARE_OP_NOT_EQUAL, "NotEqual"),
            (VK_COMPARE_OP_GREATER_OR_EQUAL, "GreaterOrEqual"),
            (VK_COMPARE_OP_ALWAYS, "Always"),
        ],
    );
    enum_combo(
        ui,
        "Fail Op",
        &mut state.fail_op,
        &[
            (VK_STENCIL_OP_KEEP, "Keep"),
            (VK_STENCIL_OP_ZERO, "Zero"),
            (VK_STENCIL_OP_REPLACE, "Replace"),
            (VK_STENCIL_OP_INCREMENT_AND_CLAMP, "IncrementAndClamp"),
            (VK_STENCIL_OP_DECREMENT_AND_CLAMP, "DecrementAndClamp"),
            (VK_STENCIL_OP_INVERT, "Invert"),
            (VK_STENCIL_OP_INCREMENT_AND_WRAP, "IncrementAndWrap"),
            (VK_STENCIL_OP_DECREMENT_AND_WRAP, "DecrementAndWrap"),
        ],
    );
    enum_combo(
        ui,
        "Depth Fail Op",
        &mut state.depth_fail_op,
        &[
            (VK_STENCIL_OP_KEEP, "Keep"),
            (VK_STENCIL_OP_ZERO, "Zero"),
            (VK_STENCIL_OP_REPLACE, "Replace"),
            (VK_STENCIL_OP_INCREMENT_AND_CLAMP, "IncrementAndClamp"),
            (VK_STENCIL_OP_DECREMENT_AND_CLAMP, "DecrementAndClamp"),
            (VK_STENCIL_OP_INVERT, "Invert"),
            (VK_STENCIL_OP_INCREMENT_AND_WRAP, "IncrementAndWrap"),
            (VK_STENCIL_OP_DECREMENT_AND_WRAP, "DecrementAndWrap"),
        ],
    );
    enum_combo(
        ui,
        "Pass Op",
        &mut state.pass_op,
        &[
            (VK_STENCIL_OP_KEEP, "Keep"),
            (VK_STENCIL_OP_ZERO, "Zero"),
            (VK_STENCIL_OP_REPLACE, "Replace"),
            (VK_STENCIL_OP_INCREMENT_AND_CLAMP, "IncrementAndClamp"),
            (VK_STENCIL_OP_DECREMENT_AND_CLAMP, "DecrementAndClamp"),
            (VK_STENCIL_OP_INVERT, "Invert"),
            (VK_STENCIL_OP_INCREMENT_AND_WRAP, "IncrementAndWrap"),
            (VK_STENCIL_OP_DECREMENT_AND_WRAP, "DecrementAndWrap"),
        ],
    );
}

fn blend_factor_combo(ui: &Ui, label: &str, current: &mut VkBlendFactor) {
    enum_combo(
        ui,
        label,
        current,
        &[
            (VK_BLEND_FACTOR_ZERO, "Zero"),
            (VK_BLEND_FACTOR_ONE, "One"),
            (VK_BLEND_FACTOR_SRC_COLOR, "SrcColor"),
            (VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR, "OneMinusSrcColor"),
            (VK_BLEND_FACTOR_DST_COLOR, "DstColor"),
            (VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR, "OneMinusDstColor"),
            (VK_BLEND_FACTOR_SRC_ALPHA, "SrcAlpha"),
            (VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA, "OneMinusSrcAlpha"),
            (VK_BLEND_FACTOR_DST_ALPHA, "DstAlpha"),
            (VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA, "OneMinusDstAlpha"),
            (VK_BLEND_FACTOR_CONSTANT_COLOR, "ConstantColor"),
            (VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR, "OneMinusConstantColor"),
            (VK_BLEND_FACTOR_CONSTANT_ALPHA, "ConstantAlpha"),
            (VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA, "OneMinusConstantAlpha"),
            (VK_BLEND_FACTOR_SRC_ALPHA_SATURATE, "SrcAlphaSaturate"),
        ],
    );
}

fn blend_op_combo(ui: &Ui, label: &str, current: &mut VkBlendOp) {
    enum_combo(
        ui,
        label,
        current,
        &[
            (VK_BLEND_OP_ADD, "Add"),
            (VK_BLEND_OP_SUBTRACT, "Subtract"),
            (VK_BLEND_OP_REVERSE_SUBTRACT, "ReverseSubtract"),
            (VK_BLEND_OP_MIN, "Min"),
            (VK_BLEND_OP_MAX, "Max"),
        ],
    );
}

fn color_target_ui(ui: &Ui, st: &mut Gui, color_target: &mut gpu::ColorTargetState) {
    let mut blend_enable = color_target.blend.is_some();
    ui.checkbox("Blend Enable", &mut blend_enable);
    ui.disabled(!blend_enable, || {
        if let Some(ref mut blend) = color_target.blend {
            if let Some(_token) = ui.begin_table_with_flags("blend_state_table", 2, TABLE_FLAGS) {
                ui.table_setup_column("Property");
                ui.table_setup_column("Value");
                ui.table_next_row();
                ui.table_next_column();
                ui.text("Src Color");
                ui.table_next_column();
                blend_factor_combo(ui, "##src_color", &mut blend.src_color_blend_factor);
                ui.table_next_column();
                ui.text("Dst Color");
                ui.table_next_column();
                blend_factor_combo(ui, "##dst_color", &mut blend.dst_color_blend_factor);
                ui.table_next_column();
                ui.text("Color Blend Op");
                ui.table_next_column();
                blend_op_combo(ui, "##color_op", &mut blend.color_blend_op);
                ui.table_next_column();
                ui.text("Src Alpha");
                ui.table_next_column();
                blend_factor_combo(ui, "##src_alpha", &mut blend.src_alpha_blend_factor);
                ui.table_next_column();
                ui.text("Dst Alpha");
                ui.table_next_column();
                blend_factor_combo(ui, "##dst_alpha", &mut blend.dst_alpha_blend_factor);
                ui.table_next_column();
                ui.text("Alpha Blend Op");
                ui.table_next_column();
                blend_op_combo(ui, "##alpha_op", &mut blend.alpha_blend_op);
            }
        }
    });
    if blend_enable && color_target.blend.is_none() {
        color_target.blend = Some(gpu::BlendState::default());
    }
}

fn graphics_state_ui(ui: &Ui, st: &mut Gui, state: &mut GraphicsPipeline) {
    enum_combo(
        ui,
        "Polygon Mode",
        &mut state.rasterization.polygon_mode,
        &[(VK_POLYGON_MODE_FILL, "Fill"), (VK_POLYGON_MODE_LINE, "Line"), (VK_POLYGON_MODE_POINT, "Point")],
    );
    enum_combo(
        ui,
        "Cull Mode",
        &mut state.rasterization.cull_mode,
        &[(VK_CULL_MODE_NONE, "None"), (VK_CULL_MODE_FRONT_BIT, "Front"), (VK_CULL_MODE_BACK_BIT, "Back")],
    );
    enum_combo(
        ui,
        "Front Face",
        &mut state.rasterization.front_face,
        &[(VK_FRONT_FACE_COUNTER_CLOCKWISE, "CCW"), (VK_FRONT_FACE_CLOCKWISE, "CW")],
    );
    enum_combo(
        ui,
        "Depth Compare Op",
        &mut state.depth_stencil.depth_compare_op,
        &[
            (VK_COMPARE_OP_NEVER, "Never"),
            (VK_COMPARE_OP_LESS, "Less"),
            (VK_COMPARE_OP_EQUAL, "Equal"),
            (VK_COMPARE_OP_LESS_OR_EQUAL, "LessOrEqual"),
            (VK_COMPARE_OP_GREATER, "Greater"),
            (VK_COMPARE_OP_NOT_EQUAL, "NotEqual"),
            (VK_COMPARE_OP_GREATER_OR_EQUAL, "GreaterOrEqual"),
            (VK_COMPARE_OP_ALWAYS, "Always"),
        ],
    );
    ui.checkbox("Depth Write", &mut state.depth_stencil.depth_write_enable);
    enum_combo(
        ui,
        "Depth Format",
        &mut state.depth_stencil.format,
        &[
            (VK_FORMAT_D32_SFLOAT_S8_UINT, "D32_SFLOAT_S8_UINT"),
            (VK_FORMAT_D32_SFLOAT, "D32_SFLOAT"),
            (VK_FORMAT_D24_UNORM_S8_UINT, "D24_UNORM_S8_UINT"),
            (VK_FORMAT_D16_UNORM, "D16_UNORM"),
        ],
    );
    ui.checkbox("Depth Test", &mut state.depth_stencil.depth_test_enable);

    {
        let mut to_remove = None;
        for (i, c) in state.color_targets.iter_mut().enumerate() {
            color_target_ui(ui, st, c);
            if ui.button(format!("Remove Color Target##{i}")) {
                to_remove = Some(i);
            }
        }
        if let Some(i) = to_remove {
            state.color_targets.remove(i);
        }
        if ui.button("Add Color Target") {
            state.color_targets.push(gpu::ColorTargetState::default());
        }
    }
}

fn shader_ui(ui: &Ui, st: &mut Gui, shader: &mut Shader) {
    let _token = ui.push_id(shader.path.display().to_string());
    let shader_options = &mut shader.options;
    string_combo(ui, "Profile", &mut shader_options.profile, &["glsl_460"]);
    ui.checkbox("Debug Info", &mut shader_options.debug);
    enum_combo(
        ui,
        "Optimization Level",
        &mut shader_options.optimization_level,
        &[
            (OptimizationLevel::None, "None"),
            (OptimizationLevel::Default, "Default"),
            (OptimizationLevel::High, "High"),
            (OptimizationLevel::Maximal, "Maximal"),
        ],
    );
    ui.text("Defines");
    let mut remove_index = None;
    if let Some(_table) = ui.begin_table_with_flags("shader_defines_table", 3, TABLE_FLAGS) {
        ui.table_setup_column("Name");
        ui.table_setup_column("Value");
        ui.table_setup_column("");
        ui.table_headers_row();
        for (index, (name, value)) in shader_options.defines.iter_mut().enumerate() {
            ui.table_next_row();
            ui.table_next_column();
            ui.set_next_item_width(-1.0);
            ui.input_text(format!("##define_name_{index}"), name).build();
            ui.table_next_column();
            ui.set_next_item_width(-1.0);
            ui.input_text(format!("##define_value_{index}"), value).build();
            ui.table_next_column();
            if ui.button(format!("Remove##{index}")) {
                remove_index = Some(index);
            }
        }
    }
    if let Some(index) = remove_index {
        shader_options.defines.remove(index);
    }
    if ui.button("Add Define") {
        shader_options.defines.push((String::new(), String::new()));
    }

    if let CompilationState::Compiled { ref reflect, .. } = shader.compilation_state {
        if let Some(_t) = ui.begin_table_with_flags("shader_param_table", 3, TABLE_FLAGS) {
            ui.table_setup_column("Name");
            ui.table_setup_column("Type");
            ui.table_setup_column("Value");
            ui.table_headers_row();
            unsafe {
                for (ep_name, ep) in shader.entry_points.iter() {
                    entry_point_ui(reflect, ep, ui, st);
                }
            }
        }
    }

    for (name, pipe) in shader.pipelines.iter_mut() {
        match pipe {
            Pipeline::Primitive(gpipe) => {
                graphics_state_ui(ui, st, gpipe);
            }
            Pipeline::Mesh(gpipe) => {
                graphics_state_ui(ui, st, gpipe);
            }
            Pipeline::Compute(_) => {}
        }
    }
}

fn image_format_combo(ui: &Ui, label: &str, current: &mut gpu::Format) {
    enum_combo(
        ui,
        label,
        current,
        &[
            (VK_FORMAT_R8_UNORM, "R8_UNORM"),
            (VK_FORMAT_R8G8_UNORM, "R8G8_UNORM"),
            (VK_FORMAT_R8G8B8A8_UNORM, "R8G8B8A8_UNORM"),
            (VK_FORMAT_R8G8B8A8_SRGB, "R8G8B8A8_SRGB"),
            (VK_FORMAT_B8G8R8A8_UNORM, "B8G8R8A8_UNORM"),
            (VK_FORMAT_B8G8R8A8_SRGB, "B8G8R8A8_SRGB"),
            (VK_FORMAT_R16_SFLOAT, "R16_SFLOAT"),
            (VK_FORMAT_R16G16B16A16_SFLOAT, "R16G16B16A16_SFLOAT"),
            (VK_FORMAT_R32_SFLOAT, "R32_SFLOAT"),
            (VK_FORMAT_R32G32B32A32_SFLOAT, "R32G32B32A32_SFLOAT"),
        ],
    );
}

fn image_resource_ui(ui: &Ui, _st: &mut Gui, image: &mut ImageResource) {
    let _token = ui.push_id(&image.name);
    ui.set_next_item_width(-1.0);
    ui.input_text("Name", &mut image.name).build();
    enum_combo(
        ui,
        "Type",
        &mut image.ty,
        &[
            (gpu::ImageType::Image1D, "1D"),
            (gpu::ImageType::Image2D, "2D"),
            (gpu::ImageType::Image3D, "3D"),
        ],
    );
    image_format_combo(ui, "Format", &mut image.format);
    {
        let mut width = image.width as i32;
        if ui.input_int("Width", &mut width).build() {
            image.width = width.max(1) as u32;
        }
    }
    if image.ty != gpu::ImageType::Image1D {
        let mut height = image.height as i32;
        if ui.input_int("Height", &mut height).build() {
            image.height = height.max(1) as u32;
        }
    }
    if image.ty == gpu::ImageType::Image3D {
        let mut depth = image.depth as i32;
        if ui.input_int("Depth", &mut depth).build() {
            image.depth = depth.max(1) as u32;
        }
    }
    {
        let mut mip_levels = image.mip_levels as i32;
        if ui.input_int("Mip Levels", &mut mip_levels).build() {
            image.mip_levels = mip_levels.max(1) as u32;
        }
    }
    {
        let mut array_layers = image.array_layers as i32;
        if ui.input_int("Array Layers", &mut array_layers).build() {
            image.array_layers = array_layers.max(1) as u32;
        }
    }
    ui.text_disabled(if image.image.is_some() { "Status: loaded" } else { "Status: not loaded" });
}

// Returns whether the specified type can be expanded to child rows in the ui.
fn type_has_child_rows(st: &Gui, ty: &TypeInfo) -> bool {
    match ty {
        TypeInfo::Void | TypeInfo::Bool | TypeInfo::Scalar(_) => false,
        //TypeInfo::Vector(ScalarType::U32, 2) if st.interpret_vec2u_as_descriptor_handle => false,
        TypeInfo::Struct(sty) if sty.fields.len() == 0 => false,
        _ => true,
    }
}

struct ParamWalk<'ctx, 'ty> {
    //ctx: &'a mut ParamWalkCtx<'ctx>,
    m: &'ctx spirv_reflect::Module,
    offset: usize,     // item offset in data
    path: String,      // item path
    ty: &'ty TypeInfo, // item TypeInfo
}

impl<'ctx, 'ty> ParamWalk<'ctx, 'ty> {
    fn child<'ty2>(&mut self, offset: usize, ty: &'ty2 TypeInfo, name: &str) -> ParamWalk<'ctx, 'ty2> {
        ParamWalk { m: self.m, offset: self.offset + offset, path: format!("{}.{}", self.path, name), ty }
    }
}

fn shader_param_ui(walk: &mut ParamWalk, ui: &Ui, st: &mut Gui, name: &str) {
    // Special case for structs with one field: display directly struct.field
    if let TypeInfo::Struct(sty) = walk.ty
        && sty.fields.len() == 1
        && st.fuse_single_field_structs
    {
        let name = format!("{}.{}", name, sty.fields[0].name);
        let ty = &walk.m[sty.fields[0].ty];
        shader_param_ui(&mut walk.child(sty.fields[0].offset as usize, ty, &sty.fields[0].name), ui, st, &name);
        return;
    }

    let has_child_rows = type_has_child_rows(st, walk.ty);
    ui.table_next_column();
    let id = ui.tree_node_config(name).flags(TreeNodeFlags::DEFAULT_OPEN).leaf(!has_child_rows).push();
    if ui.is_item_hovered() {
        ui.tooltip(|| {
            ui.text(format!("{}", walk.path));
        });
    }
    ui.table_next_column();
    ui.text(format!("{}", pretty_print_type(walk.m, walk.ty)));
    if ui.is_item_hovered() {
        ui.tooltip(|| match *walk.ty {
            TypeInfo::Array { stride, .. } => ui.text(format!("Array Stride: {stride:?}")),
            _ => {}
        });
    }
    ui.table_next_column();
    //param_value_summary_ui(walk, ui, st);

    if id.is_some() && has_child_rows {
        shader_param_child_rows_ui(walk, ui, st);
    }
}

fn pointer_param_child_rows_ui(walk: &mut ParamWalk, ui: &Ui, st: &mut Gui, pointee_type_id: TypeId) {
    let pointee_type = &walk.m[pointee_type_id];
    let mut child_walk = ParamWalk { m: walk.m, offset: 0, path: format!("{}.$", walk.path), ty: pointee_type };
    shader_param_child_rows_ui(&mut child_walk, ui, st);
}

fn struct_param_child_rows_ui(walk: &mut ParamWalk, ui: &Ui, st: &mut Gui, ty: &StructType) {
    for field in ty.fields.iter() {
        shader_param_ui(&mut walk.child(field.offset as usize, &walk.m[field.ty], &field.name), ui, st, &field.name);
    }
}

fn shader_param_child_rows_ui(walk: &mut ParamWalk, ui: &Ui, st: &mut Gui) {
    match *walk.ty {
        TypeInfo::Struct(ref sty) => {
            struct_param_child_rows_ui(walk, ui, st, sty);
        }
        TypeInfo::Pointer(pointee_type) => {
            if let Some(pointee) = pointee_type.pointee {
                pointer_param_child_rows_ui(walk, ui, st, pointee);
            }
        }
        _ => {}
    }
}

impl Game {
    fn main_menu(&mut self, ui: &Ui) {
        ui.main_menu_bar(|| {
            ui.menu("File", || {
                if ui.menu_item("Open Project...") {
                    self.open_project();
                }
                if ui.menu_item("Save Project") {
                    self.save_project(false);
                }
                if ui.menu_item("Save Project As...") {
                    self.save_project(true);
                }
                ui.menu_item("Exit");
            });
            ui.menu("Edit", || {
                if ui.menu_item("Add Geometry...") {
                    self.add_geometry();
                }
                if ui.menu_item("Add Shader...") {
                    self.add_shader();
                }
            });
            ui.menu("View", || {
                if ui.menu_item("Shaders") {
                    self.gui.show_shaders_window = true;
                }
                if ui.menu_item("Images") {
                    self.gui.show_images_window = true;
                }
            });
            ui.menu("Options", || {});
        });
    }

    fn shaders_window(&mut self, ui: &Ui) {
        if !self.gui.show_shaders_window {
            return;
        }
        let mut opened = self.gui.show_shaders_window;
        ui.window("Shaders").opened(&mut opened).size([400.0, 300.0], imgui::Condition::FirstUseEver).build(|| {
            let mut to_remove = None;
            for (index, shader) in self.project.shaders.iter_mut().enumerate() {
                if ui.collapsing_header(format!("Shader: {}", shader.path.display()), TreeNodeFlags::empty()) {
                    shader_ui(ui, &mut self.gui, shader);
                    if ui.button(format!("Remove Shader##{index}")) {
                        to_remove = Some(index);
                    }
                }
            }
            if let Some(index) = to_remove {
                self.project.shaders.remove(index);
            }
            if ui.button("Add Shader...") {
                self.add_shader();
            }
        });
        self.gui.show_shaders_window = opened;
    }

    fn images_window(&mut self, ui: &Ui) {
        if !self.gui.show_images_window {
            return;
        }
        let mut opened = self.gui.show_images_window;
        ui.window("Images").opened(&mut opened).size([400.0, 300.0], imgui::Condition::FirstUseEver).build(|| {
            let mut to_remove = None;
            for (index, image) in self.project.images.iter_mut().enumerate() {
                let header_name = if image.name.is_empty() { format!("Image #{index}") } else { image.name.clone() };
                if ui.collapsing_header(format!("{header_name}##{index}"), TreeNodeFlags::empty()) {
                    image_resource_ui(ui, &mut self.gui, image);
                    if ui.button(format!("Remove Image##{index}")) {
                        to_remove = Some(index);
                    }
                }
            }
            if let Some(index) = to_remove {
                self.project.images.remove(index);
            }
            if ui.button("Add Image") {
                self.add_image();
            }
        });
        self.gui.show_images_window = opened;
    }

    fn status_bar(&mut self, ui: &Ui) {
        const STATUS_BAR_HEIGHT: f32 = 20.0;
        let w = ui.io().display_size[0];
        let h = ui.io().display_size[1];
        let y = h - STATUS_BAR_HEIGHT;
        let _token = ui.push_style_var(StyleVar::WindowPadding([3.0, 3.0]));
        ui.window("Status Bar")
            .bg_alpha(0.5)
            .title_bar(false)
            .position([0.0, y], Always)
            .resizable(false)
            .movable(false)
            .size([w, STATUS_BAR_HEIGHT], Always)
            .build(|| {
                ui.text(&self.gui.status);
            });
    }

    pub(crate) fn main_gui(&mut self, ui: &Ui) {
        ui.dockspace_over_main_viewport();
        self.main_menu(ui);
        self.status_bar(ui);
        self.shaders_window(ui);
        self.images_window(ui);
    }
}
