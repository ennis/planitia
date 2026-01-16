use crate::manifest::Error::{InvalidType, MissingField};
use anyhow::{Context, anyhow};
use log::error;
use shader_archive::{gpu, ColorBlendEquationData};
use shader_archive::gpu::vk;
use shader_archive::gpu::vk::PolygonMode;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use toml::Value as TomlValue;

/// The maximum number of color targets in graphics states.
pub const MAX_COLOR_TARGETS: usize = 8;

pub const DEFAULT_SHADER_PROFILE: &str = "glsl_460";

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("missing field: {0}")]
    MissingField(&'static str),
    #[error("invalid type for field {0}")]
    InvalidType(&'static str),
    #[error("{0}")]
    Other(&'static str),
}

fn get_image_usage(usage_str: &str) -> Result<gpu::ImageUsage, Error> {
    match usage_str {
        "color_attachment" => Ok(gpu::ImageUsage::COLOR_ATTACHMENT),
        "depth_stencil_attachment" => Ok(gpu::ImageUsage::DEPTH_STENCIL_ATTACHMENT),
        "sampled" => Ok(gpu::ImageUsage::SAMPLED),
        "storage" => Ok(gpu::ImageUsage::STORAGE),
        "transfer_src" => Ok(gpu::ImageUsage::TRANSFER_SRC),
        "transfer_dst" => Ok(gpu::ImageUsage::TRANSFER_DST),
        _ => {
            error!("Unknown image usage: {}", usage_str);
            Err(InvalidType("usage"))
        }
    }
}

fn get_image_usages(usages: &TomlValue) -> Result<gpu::ImageUsage, Error> {
    if let Some(array) = usages.as_array() {
        let mut usage_flags = gpu::ImageUsage::empty();
        for item in array {
            let usage_str = item.as_str().ok_or(InvalidType("usages array element"))?;
            let usage = get_image_usage(usage_str)?;
            usage_flags |= usage;
        }
        Ok(usage_flags)
    } else if let Some(usage_str) = usages.as_str() {
        get_image_usage(usage_str)
    } else {
        Err(InvalidType("usage").into())
    }
}

#[derive(Clone, Default)]
pub struct Resource {
    pub format: vk::Format,
    pub length: Option<u32>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub usage: Option<gpu::ImageUsage>,
}

impl Resource {
    fn from_toml(toml: &TomlValue) -> anyhow::Result<Self> {
        let mut resource = Resource::default();
        if let Some(format_str) = toml.get_optional_str("format")? {
            resource.format = get_format(format_str)?;
        }

        if let Some(length) = toml.get("length") {
            if let Some(value) = length.as_integer() {
                resource.length = Some(value.try_into()?);
            } else if let Some(str) = length.as_str() {
                // parse special strings like "dynamic"
                match str {
                    "dynamic" => resource.length = None,
                    _ => return Err(InvalidType("length").into()),
                }
            } else {
                return Err(InvalidType("length").into());
            }
        }

        if let Some(usage_toml) = toml.get("usage") {
            resource.usage = Some(get_image_usages(usage_toml)?);
        }

        if let Some(width) = toml.get_optional_integer("width")? {
            resource.width = Some(width.try_into()?);
        }

        if let Some(height) = toml.get_optional_integer("height")? {
            resource.height = Some(height.try_into()?);
        }

        Ok(resource)
    }
}

#[derive(Clone)]
pub struct ColorAttachment {
    pub resource: Option<String>,
    pub clear_color: Option<[f32; 4]>,
}

impl ColorAttachment {
    pub fn from_toml(toml: &TomlValue) -> anyhow::Result<Self> {
        let clear_color = if let Some(array) = toml.get("clear_color") {
            let arr = array.as_array().ok_or(InvalidType("clear_color"))?;
            if arr.len() != 4 {
                return Err(InvalidType("clear_color").into());
            }
            let mut color = [0.0f32; 4];
            for (i, v) in arr.iter().enumerate() {
                color[i] = v.as_float().ok_or(InvalidType("clear_color array element"))? as f32;
            }
            Some(color)
        } else {
            None
        };

        let resource = toml.get_optional_str("resource")?.map(|s| s.to_string());

        Ok(ColorAttachment { resource, clear_color })
    }
}

#[derive(Clone)]
pub struct DepthStencilAttachment {
    pub resource: Option<String>,
    pub clear_depth: Option<f32>,
    pub clear_stencil: Option<u32>,
}

impl DepthStencilAttachment {
    pub fn from_toml(toml: &TomlValue) -> anyhow::Result<Self> {
        let clear_depth = toml.get_optional_float("clear_depth")?.map(|v| v as f32);
        let clear_stencil = toml.get_optional_integer("clear_stencil")?.map(|v| v as u32);
        let resource = toml.get_optional_str("resource")?.map(|s| s.to_string());

        Ok(DepthStencilAttachment {
            resource,
            clear_depth,
            clear_stencil,
        })
    }
}

#[derive(Clone)]
pub struct Pass {
    // Original raw table, contains render state overrides
    pub raw: TomlValue,
    pub color_attachments: Vec<ColorAttachment>,
    pub depth_stencil_attachment: Option<DepthStencilAttachment>,
}

impl Pass {
    pub fn from_toml(toml: &TomlValue) -> anyhow::Result<Self> {
        let mut color_attachments = vec![];
        if let Some(array) = toml.get_optional_array("color_attachments")? {
            for item in array {
                color_attachments.push(ColorAttachment::from_toml(item)?);
            }
        }

        let depth_attachment = if let Some(depth_toml) = toml.get_optional_table("depth_stencil_attachment")? {
            Some(DepthStencilAttachment::from_toml(depth_toml)?)
        } else {
            None
        };

        Ok(Pass {
            raw: toml.clone(),
            color_attachments,
            depth_stencil_attachment: depth_attachment,
        })
    }
}

#[derive(Clone)]
pub struct BuildManifest {
    pub input_files: Vec<String>,
    pub manifest_path: PathBuf,
    pub include_paths: Vec<String>,
    pub output_file: String,
    pub default: GraphicsState,
    pub shader_profile: String,
    pub compiler: CompilerOptions,
    pub pass: BTreeMap<String, Pass>,
    //pub override_: toml::Table,
    pub resources: BTreeMap<String, Resource>,
}

impl BuildManifest {
    pub(crate) fn load(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        fn load_inner(path: &Path) -> anyhow::Result<BuildManifest> {
            let manifest_str = std::fs::read_to_string(&path)?;
            let manifest_toml: TomlValue = toml::from_str(&manifest_str).context("invalid TOML")?;
            BuildManifest::from_toml(&manifest_toml, path.to_path_buf()).context("failed to parse manifest")
        }
        load_inner(path.as_ref())
    }

    pub fn from_toml(toml: &TomlValue, manifest_path: PathBuf) -> anyhow::Result<Self> {
        // input_files = ["file1.slang", "file2.slang", "../*.slang", ...]
        let input_files = {
            let input_files_toml = toml.get("input_files").ok_or(MissingField("input_files"))?;
            if let Some(array) = input_files_toml.as_array() {
                array
                    .iter()
                    .map(|v| {
                        v.as_str()
                            .ok_or(InvalidType("input_files array element"))
                            .map(|s| s.to_string())
                    })
                    .collect::<Result<Vec<String>, Error>>()?
            } else if let Some(s) = input_files_toml.as_str() {
                vec![s.to_string()]
            } else {
                return Err(InvalidType("input_files").into());
            }
        };

        // include_paths = ["path1", "path2", ...] (optional)
        let include_paths = toml
            .get_optional_array("include_paths")?
            .unwrap_or(&vec![])
            .iter()
            .map(|v| {
                v.as_str()
                    .ok_or(InvalidType("include_paths array element"))
                    .map(|s| s.to_string())
            })
            .collect::<Result<Vec<String>, Error>>()?;

        // output file
        let output_file = toml
            .get_optional_str("output_file")?
            .ok_or(MissingField("output_file"))?
            .to_string();

        // default graphics state
        let mut default = GraphicsState::default();
        default.read(toml.get("default").ok_or(MissingField("default"))?)?;

        // shader profile
        let shader_profile = toml
            .get_optional_str("shader_profile")?
            .unwrap_or(DEFAULT_SHADER_PROFILE)
            .to_string();

        // passes
        let mut pass = BTreeMap::new();
        if let Some(pass_toml) = toml.get_optional_table("pass")? {
            for (name, pass_toml) in pass_toml.as_table().unwrap().iter() {
                pass.insert(name.clone(), Pass::from_toml(pass_toml)?);
            }
        }

        let compiler = {
            let mut compiler = CompilerOptions::default();
            if let Some(compiler_toml) = toml.get_optional_table("compiler")? {
                compiler = CompilerOptions::from_toml(compiler_toml)?;
            }
            compiler
        };

        // resource table
        let mut resources = BTreeMap::new();
        if let Some(resources_toml) = toml.get_optional_table("resources")? {
            for (name, res_toml) in resources_toml.as_table().unwrap().iter() {
                resources.insert(name.clone(), Resource::from_toml(res_toml)?);
            }
        }

        Ok(BuildManifest {
            input_files,
            shader_profile,
            manifest_path,
            include_paths,
            output_file,
            default,
            pass,
            compiler,
            resources,
        })
    }
}

/// Shader compilation options.
#[derive(Clone, Default)]
pub struct CompilerOptions {
    /// Preprocessor definitions
    pub defines: BTreeMap<String, String>,
    /// Shader profile
    pub profile: String,
    /// Enable optimizations
    pub optimize: bool,
    /// Enable debug information
    pub debug: bool,
}

impl CompilerOptions {
    fn from_toml(toml: &TomlValue) -> Result<Self, Error> {
        let mut options = CompilerOptions {
            defines: BTreeMap::new(),
            profile: DEFAULT_SHADER_PROFILE.to_string(),
            optimize: false,
            debug: false,
        };

        if let Some(defines_array) = toml.get_optional_array("defines")? {
            for define_value in defines_array {
                let define_str = define_value.as_str().ok_or(InvalidType("defines array element"))?;
                let parts: Vec<&str> = define_str.splitn(2, '=').collect();
                if parts.len() == 2 {
                    // DEFINE=VALUE
                    options.defines.insert(parts[0].to_string(), parts[1].to_string());
                } else {
                    // DEFINE
                    options.defines.insert(parts[0].to_string(), String::new());
                }
            }
        }

        if let Some(profile_str) = toml.get_optional_str("profile")? {
            options.profile = profile_str.to_string();
        }

        if let Some(optimize) = toml.get_optional_bool("optimize")? {
            options.optimize = optimize;
        }

        if let Some(debug) = toml.get_optional_bool("debug")? {
            options.debug = debug;
        }

        Ok(options)
    }
}

/// Graphics state configuration for a graphics pipeline.
#[derive(Clone)]
pub struct GraphicsState {
    pub rasterizer: shader_archive::RasterizerStateData,
    pub depth_stencil: shader_archive::DepthStencilStateData,
    pub color_targets: Vec<shader_archive::ColorTarget>,
}

impl Default for GraphicsState {
    fn default() -> Self {
        Self {
            rasterizer: shader_archive::RasterizerStateData::default(),
            depth_stencil: shader_archive::DepthStencilStateData::default(),
            color_targets: vec![],
        }
    }
}

impl GraphicsState {
    fn read(&mut self, toml: &TomlValue) -> anyhow::Result<()> {
        if let Some(rasterizer_obj) = toml.get_optional_table("rasterizer").context("in rasterizer")? {
            read_rasterizer_state(rasterizer_obj, &mut self.rasterizer)?;
        }
        if let Some(depth_stencil_obj) = toml.get_optional_table("depth_stencil").context("in depth_stencil")? {
            read_depth_stencil_state(depth_stencil_obj, &mut self.depth_stencil)?;
        }

        // The color targets array is mandatory: the "default" would be an empty array and this
        // is too error-prone.
        let color_targets = toml
            .get_optional_table_or_array("color_targets")?
            .ok_or(MissingField("color_targets"))?;
        read_color_targets(color_targets, &mut self.color_targets)?;

        Ok(())
    }

    pub fn apply_overrides(&mut self, overrides: &TomlValue) -> anyhow::Result<()> {
        if let Some(rasterizer_obj) = overrides.get_optional_table("rasterizer")? {
            read_rasterizer_state(rasterizer_obj, &mut self.rasterizer)?;
        }
        if let Some(depth_stencil_obj) = overrides.get_optional_table("depth_stencil")? {
            read_depth_stencil_state(depth_stencil_obj, &mut self.depth_stencil)?;
        }
        if let Some(color_targets) = overrides.get_optional_table_or_array("color_targets")? {
            read_color_targets(color_targets, &mut self.color_targets)?;
        }

        Ok(())
    }
}

////////////////////////////////////////////////////////////////////////////////////////////

trait TomlExt {
    /// Retrieves an optional string field from a TOML value.
    ///
    /// Returns `Ok(None)` if the field is not present.
    /// Returns `Err(Error::InvalidType)` if the field is present but not a string.
    fn get_optional_str(&self, field: &'static str) -> Result<Option<&str>, Error>;
    /// Retrieves an optional boolean field from a TOML value.
    ///
    /// Returns `Ok(None)` if the field is not present.
    /// Returns `Err(Error::InvalidType)` if the field is present but not a boolean
    fn get_optional_bool(&self, field: &'static str) -> Result<Option<bool>, Error>;
    fn get_optional_integer(&self, field: &'static str) -> Result<Option<i64>, Error>;
    fn get_optional_float(&self, field: &'static str) -> Result<Option<f64>, Error>;
    /// Retrieves an optional table field from a TOML value.
    ///
    /// Returns `Ok(None)` if the field is not present.
    /// Returns `Err(Error::InvalidType)` if the field is present but not a table
    fn get_optional_table(&self, field: &'static str) -> Result<Option<&TomlValue>, Error>;
    /// Retrieves an optional array field from a TOML value.
    ///
    /// Returns `Ok(None)` if the field is not present.
    /// Returns `Err(Error::InvalidType)` if the field is present but not an array
    fn get_optional_array(&self, field: &'static str) -> Result<Option<&Vec<TomlValue>>, Error>;
    /// Retrieves an optional field that is either a table or an array from a TOML value.
    ///
    /// Returns `Ok(None)` if the field is not present.
    /// Returns `Err(Error::InvalidType)` if the field is present but neither a table nor an array.
    fn get_optional_table_or_array(&self, field: &'static str) -> Result<Option<&TomlValue>, Error>;
}

impl TomlExt for toml::Value {
    fn get_optional_str(&self, field: &'static str) -> Result<Option<&str>, Error> {
        match self.get(field) {
            None => Ok(None),
            Some(value) => value.as_str().ok_or(InvalidType(field)).map(Some),
        }
    }

    fn get_optional_bool(&self, field: &'static str) -> Result<Option<bool>, Error> {
        match self.get(field) {
            None => Ok(None),
            Some(value) => value.as_bool().ok_or(InvalidType(field)).map(Some),
        }
    }

    fn get_optional_integer(&self, field: &'static str) -> Result<Option<i64>, Error> {
        match self.get(field) {
            None => Ok(None),
            Some(value) => value.as_integer().ok_or(InvalidType(field)).map(Some),
        }
    }

    fn get_optional_float(&self, field: &'static str) -> Result<Option<f64>, Error> {
        match self.get(field) {
            None => Ok(None),
            Some(value) => value.as_float().ok_or(InvalidType(field)).map(Some),
        }
    }

    fn get_optional_table(&self, field: &'static str) -> Result<Option<&TomlValue>, Error> {
        match self.get(field) {
            None => Ok(None),
            Some(value) => value.as_table().ok_or(InvalidType(field)).map(|_| Some(value)),
        }
    }

    fn get_optional_array(&self, field: &'static str) -> Result<Option<&Vec<TomlValue>>, Error> {
        match self.get(field) {
            None => Ok(None),
            Some(value) => value.as_array().ok_or(InvalidType(field)).map(|arr| Some(arr)),
        }
    }

    fn get_optional_table_or_array(&self, field: &'static str) -> Result<Option<&TomlValue>, Error> {
        match self.get(field) {
            None => Ok(None),
            Some(value) => {
                if value.is_table() || value.is_array() {
                    Ok(Some(value))
                } else {
                    Err(InvalidType(field))
                }
            }
        }
    }
}

fn read_rasterizer_state(toml: &TomlValue, out: &mut shader_archive::RasterizerStateData) -> Result<(), Error> {
    //let cull_mode = read_str(json, "cull_mode", Some("back"))?;
    if let Some(polygon_mode) = toml.get_optional_str("polygon_mode")? {
        out.polygon_mode = match polygon_mode {
            "fill" => PolygonMode::FILL,
            "line" => PolygonMode::LINE,
            "point" => PolygonMode::POINT,
            _ => {
                error!("Unknown polygon mode: {}", polygon_mode);
                PolygonMode::FILL
            }
        };
    }

    if let Some(cull_mode) = toml.get_optional_str("cull_mode")? {
        out.cull_mode = match cull_mode {
            "none" => vk::CullModeFlags::NONE,
            "front" => vk::CullModeFlags::FRONT,
            "back" => vk::CullModeFlags::BACK,
            "front_and_back" => vk::CullModeFlags::FRONT_AND_BACK,
            _ => {
                error!("Unknown cull mode: {}", cull_mode);
                vk::CullModeFlags::BACK
            }
        };
    }

    Ok(())
}

fn get_format(fmtstr: &str) -> Result<vk::Format, Error> {
    match fmtstr {
        "RGBA8" => Ok(vk::Format::R8G8B8A8_UNORM),
        "RGBA8UI" => Ok(vk::Format::R8G8B8A8_UINT),
        "RGBA16UI" => Ok(vk::Format::R16G16B16A16_UINT),
        "RGB10_A2" => Ok(vk::Format::A2B10G10R10_UNORM_PACK32),
        "R32F" => Ok(vk::Format::R32_SFLOAT),
        "RG32F" => Ok(vk::Format::R32G32_SFLOAT),
        "D32F" => Ok(vk::Format::D32_SFLOAT),
        "D32F_S8UI" => Ok(vk::Format::D32_SFLOAT_S8_UINT),
        _ => {
            error!("Unknown format: {}", fmtstr);
            Err(Error::InvalidType("format"))
        }
    }
}

fn get_blend_factor(factor_str: &str) -> Result<vk::BlendFactor, Error> {
    match factor_str {
        "zero" => Ok(vk::BlendFactor::ZERO),
        "one" => Ok(vk::BlendFactor::ONE),
        "src_alpha" => Ok(vk::BlendFactor::SRC_ALPHA),
        "one_minus_src_alpha" => Ok(vk::BlendFactor::ONE_MINUS_SRC_ALPHA),
        _ => {
            error!("Unknown blend factor: {}", factor_str);
            Err(Error::InvalidType("blend_factor"))
        }
    }
}

fn get_blend_op(op_str: &str) -> Result<vk::BlendOp, Error> {
    match op_str {
        "add" => Ok(vk::BlendOp::ADD),
        "subtract" => Ok(vk::BlendOp::SUBTRACT),
        "reverse_subtract" => Ok(vk::BlendOp::REVERSE_SUBTRACT),
        _ => {
            error!("Unknown blend op: {}", op_str);
            Err(Error::InvalidType("blend_op"))
        }
    }
}

fn read_depth_stencil_state(toml: &TomlValue, out: &mut shader_archive::DepthStencilStateData) -> anyhow::Result<()> {
    // any depth-stencil field automatically enables depth testing
    if let Some(format_str) = toml.get_optional_str("format")? {
        out.format = get_format(format_str).context("in depth_stencil")?;
        out.enable = true;
    }
    if let Some(depth_compare_op) = toml.get_optional_str("compare_op")? {
        out.depth_compare_op = match depth_compare_op {
            "always" => vk::CompareOp::ALWAYS,
            "less" => vk::CompareOp::LESS,
            "lequal" => vk::CompareOp::LESS_OR_EQUAL,
            _ => {
                error!("Unknown depth compare op: {}", depth_compare_op);
                vk::CompareOp::ALWAYS
            }
        };
        out.enable = true;
    }
    if let Some(depth_write_enable) = toml.get_optional_bool("write_enable")? {
        out.depth_write_enable = depth_write_enable;
        out.enable = true;
    }
    // ... but if "enable" is explicitly set, it overrides everything
    if let Some(enable) = toml.get_optional_bool("enable")? {
        out.enable = enable;
    }
    Ok(())
}

fn read_blend(toml: &TomlValue) -> anyhow::Result<Option<ColorBlendEquationData>> {
    if let Some(str) = toml.as_str() {
        match str {
            "disabled" => Ok(None),
            "over" => {
                Ok(Some(ColorBlendEquationData {
                    src_color_blend_factor: vk::BlendFactor::SRC_ALPHA,
                    dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
                    color_blend_op: vk::BlendOp::ADD,
                    src_alpha_blend_factor: vk::BlendFactor::ONE,
                    dst_alpha_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
                    alpha_blend_op: vk::BlendOp::ADD,
                }))
            }
            "over_premultiplied" => {
                Ok(Some(ColorBlendEquationData {
                    src_color_blend_factor: vk::BlendFactor::ONE,
                    dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
                    color_blend_op: vk::BlendOp::ADD,
                    src_alpha_blend_factor: vk::BlendFactor::ONE,
                    dst_alpha_blend_factor: vk::BlendFactor::ZERO,
                    alpha_blend_op: vk::BlendOp::ADD,
                }))
            }
            _ => Err(anyhow!("unknown predefined blend mode").context("in blend")),
        }
    } else {
        let mut blend = ColorBlendEquationData::default();
        if let Some(src_color_blend_factor) = toml.get_optional_str("src_color")? {
            blend.src_color_blend_factor = get_blend_factor(src_color_blend_factor)?;
        }
        if let Some(dst_color_blend_factor) = toml.get_optional_str("dst_color")? {
            blend.dst_color_blend_factor = get_blend_factor(dst_color_blend_factor)?;
        }
        if let Some(color_blend_op) = toml.get_optional_str("color_op")? {
            blend.color_blend_op = get_blend_op(color_blend_op)?;
        }
        if let Some(src_alpha_blend_factor) = toml.get_optional_str("src_alpha")? {
            blend.src_alpha_blend_factor = get_blend_factor(src_alpha_blend_factor)?;
        }
        if let Some(dst_alpha_blend_factor) = toml.get_optional_str("dst_alpha")? {
            blend.dst_alpha_blend_factor = get_blend_factor(dst_alpha_blend_factor)?;
        }
        if let Some(alpha_blend_op) = toml.get_optional_str("alpha_op")? {
            blend.alpha_blend_op = get_blend_op(alpha_blend_op)?;
        }
        Ok(Some(blend))
    }
}

fn read_color_target(toml: &TomlValue, out: &mut shader_archive::ColorTarget) -> anyhow::Result<()> {
    if let Some(format_str) = toml.get_optional_str("format")? {
        out.format = get_format(format_str)?;
    }
    if let Some(blend_toml) = toml.get("blend") {
        out.blend = read_blend(blend_toml)?;
    }
    Ok(())
}

fn read_color_targets(toml: &TomlValue, out: &mut Vec<shader_archive::ColorTarget>) -> anyhow::Result<()> {
    if let Some(array) = toml.as_array() {
        out.clear();
        for item in array {
            let mut color_target = shader_archive::ColorTarget::default();
            read_color_target(item, &mut color_target).context("in color_targets")?;
            out.push(color_target);
        }
        Ok(())
    } else if let Some(object) = toml.as_table() {
        // parse overrides like:
        //
        //      {
        //          "0": { ... },
        //          "2": { ... }
        //      }
        for (key, value) in object {
            if let Ok(index) = key.parse::<usize>() {
                // sanity check index and resize if needed
                if index >= MAX_COLOR_TARGETS {
                    return Err(anyhow!("color target index out of range").context("in color_targets"));
                }
                if index >= out.len() {
                    out.resize(index + 1, shader_archive::ColorTarget::default());
                }
                read_color_target(value, &mut out[index]).context("in color_targets")?;
            }
        }
        Ok(())
    } else {
        return Err(InvalidType("color_targets").into());
    }
}
