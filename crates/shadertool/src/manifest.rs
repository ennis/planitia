use crate::get_file_mtime;
use crate::manifest::Error::{InvalidType, MissingField};
use anyhow::{Context, anyhow};
use log::error;
use sharc::ColorBlendEquation;
use sharc::gpu_types::vk;
use sharc::gpu_types::vk::PolygonMode;
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
    #[error("invalid type for field {0}, expected {1}")]
    InvalidType(&'static str, &'static str),
    #[error("invalid field")]
    InvalidField,
    #[error("{0}")]
    Other(&'static str),
    #[error("`{0}`: invalid enum value `{1}`, expected `{2}`")]
    InvalidEnumValue(&'static str, String, &'static str),
}

#[derive(thiserror::Error, Debug)]
#[error("invalid enum value")]
pub struct InvalidEnumValue;

fn validate_keys(toml_value: &TomlValue, mandatory: &[&str], optional: &[&str]) -> anyhow::Result<()> {
    let mut has_errors = false;
    if let Some(table) = toml_value.as_table() {
        for key in table.keys() {
            if !mandatory.contains(&key.as_str()) && !optional.contains(&key.as_str()) {
                error!("unknown field: {}", key);
                has_errors = true;
            }
        }

        for mandatory_key in mandatory {
            if !table.contains_key(*mandatory_key) {
                error!("missing mandatory field: {}", mandatory_key);
                has_errors = true;
            }
        }

        if has_errors { Err(Error::InvalidField.into()) } else { Ok(()) }
    } else {
        error!("expected a table");
        Err(Error::Other("expected a table").into())
    }
}

/*
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
}*/

/*
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
}*/

#[derive(Clone)]
pub struct ColorAttachment {
    pub resource: Option<String>,
    pub clear_color: Option<[f32; 4]>,
}

impl ColorAttachment {
    pub fn from_toml(toml: &TomlValue) -> anyhow::Result<Self> {
        let clear_color = if let Some(arr) = toml.get_optional_array("clear_color")? {
            if arr.len() != 4 {
                return Err(InvalidType("clear_color", "array of 4 floats").into());
            }
            let mut color = [0.0f32; 4];
            for (i, v) in arr.iter().enumerate() {
                color[i] = v.as_float().ok_or(InvalidType("clear_color", "array of 4 floats"))? as f32;
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

        Ok(DepthStencilAttachment { resource, clear_depth, clear_stencil })
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

        Ok(Pass { raw: toml.clone(), color_attachments, depth_stencil_attachment: depth_attachment })
    }
}

#[derive(Clone)]
pub struct BuildManifest {
    //pub input_files: Vec<String>,
    pub manifest_path: PathBuf,
    pub canonical_manifest_path: PathBuf,
    pub mtime: u64,
    pub include_paths: Vec<String>,
    pub default: GraphicsState,
    pub shader_profile: String,
    pub compiler: CompilerOptions,
    // pass_name -> pass overrides
    pub pass: BTreeMap<String, Pass>,
}

impl Default for BuildManifest {
    fn default() -> Self {
        Self {
            manifest_path: Default::default(),
            canonical_manifest_path: Default::default(),
            mtime: 0,
            include_paths: vec![],
            //output_directory: None,
            default: GraphicsState::default(),
            shader_profile: DEFAULT_SHADER_PROFILE.to_string(),
            compiler: CompilerOptions::default(),
            pass: BTreeMap::new(),
        }
    }
}

impl BuildManifest {
    pub(crate) fn load(&mut self, path: impl AsRef<Path>) -> anyhow::Result<()> {
        let path = path.as_ref();
        let manifest_str = std::fs::read_to_string(&path)?;
        let manifest_toml: TomlValue = toml::from_str(&manifest_str).context("invalid TOML")?;
        self.load_from_toml(&manifest_toml, path.to_path_buf()).context("failed to parse manifest")?;
        Ok(())
    }

    pub fn load_from_toml(&mut self, toml: &TomlValue, manifest_path: PathBuf) -> anyhow::Result<()> {
        let (canonical_manifest_path, mtime) = get_file_mtime(&manifest_path)?;

        self.manifest_path = manifest_path.clone();
        self.canonical_manifest_path = canonical_manifest_path;
        self.mtime = mtime;

        // Load inherited manifests.
        // inherit = ["other_manifest.toml", ...] (optional)
        if let Some(inherits) = toml.get_optional_str_or_array("inherit")? {
            for inherit in inherits {
                let inherit_path = manifest_path.parent().unwrap_or(Path::new(".")).join(inherit);
                //debug!("loading inherited manifest: {}", inherit_path.display());
                let manifest_str = std::fs::read_to_string(&inherit_path)?;
                let manifest_toml: TomlValue = toml::from_str(&manifest_str).context("invalid TOML")?;
                self.load_from_toml(&manifest_toml, inherit_path.clone())
                    .with_context(|| format!("failed to load inherited manifest `{}`", inherit_path.display()))?;
            }
        }

        /*
        // input_files = ["file1.slang", "file2.slang", "..slang", ...]
        let input_files = {
            let input_files_toml = toml.get("input_files").ok_or(MissingField("input_files"))?;
            if let Some(array) = input_files_toml.as_array() {
                array
                    .iter()
                    .map(|v| v.as_str().ok_or(InvalidType("input_files array element")).map(|s| s.to_string()))
                    .collect::<Result<Vec<String>, Error>>()?
            } else if let Some(s) = input_files_toml.as_str() {
                vec![s.to_string()]
            } else {
                return Err(InvalidType("input_files").into());
            }
        };
        */

        // Slang include paths.
        // include_paths = ["path1", "path2", ...] (optional)
        self.include_paths.extend(
            toml.get_optional_array("include_paths")?
                .unwrap_or(&vec![])
                .iter()
                .map(|v| v.as_str().ok_or(InvalidType("include_paths", "array of strings")).map(|s| s.to_string()))
                .collect::<Result<Vec<String>, Error>>()?,
        );

        // default graphics state
        self.default.read(toml.get("default").ok_or(MissingField("default"))?)?;

        // shader profile
        self.shader_profile = toml.get_optional_str("shader_profile")?.unwrap_or(DEFAULT_SHADER_PROFILE).to_string();

        // passes
        if let Some(toml) = toml.get_optional_table("pass")? {
            for (name, toml) in toml.as_table().unwrap().iter() {
                self.pass.insert(name.clone(), Pass::from_toml(toml)?);
            }
        }

        // compiler options
        if let Some(compiler_toml) = toml.get_optional_table("compiler")? {
            self.compiler.load_from_toml(compiler_toml)?;
        }

        // resource table
        //let mut resources = BTreeMap::new();
        //if let Some(resources_toml) = toml.get_optional_table("resources")? {
        //    for (name, res_toml) in resources_toml.as_table().unwrap().iter() {
        //        resources.insert(name.clone(), Resource::from_toml(res_toml)?);
        //    }
        //}

        Ok(())
    }
}

/// Shader compilation options.
#[derive(Clone)]
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

impl Default for CompilerOptions {
    fn default() -> Self {
        Self { defines: BTreeMap::new(), profile: DEFAULT_SHADER_PROFILE.to_string(), optimize: false, debug: false }
    }
}

impl CompilerOptions {
    fn load_from_toml(&mut self, toml: &TomlValue) -> Result<(), Error> {
        if let Some(defines_array) = toml.get_optional_array("defines")? {
            for define_value in defines_array {
                let define_str = define_value.as_str().ok_or(InvalidType("defines", "array of strings"))?;
                let parts: Vec<&str> = define_str.splitn(2, '=').collect();
                if parts.len() == 2 {
                    // DEFINE=VALUE
                    self.defines.insert(parts[0].to_string(), parts[1].to_string());
                } else {
                    // DEFINE
                    self.defines.insert(parts[0].to_string(), String::new());
                }
            }
        }

        if let Some(profile_str) = toml.get_optional_str("profile")? {
            self.profile = profile_str.to_string();
        }

        if let Some(optimize) = toml.get_optional_bool("optimize")? {
            self.optimize = optimize;
        }

        if let Some(debug) = toml.get_optional_bool("debug")? {
            self.debug = debug;
        }

        Ok(())
    }
}

/// Graphics state configuration for a graphics pipeline.
#[derive(Clone)]
pub struct GraphicsState {
    pub rasterizer: sharc::RasterizationState,
    pub depth_stencil: sharc::DepthStencilState,
    pub color_targets: Vec<sharc::ColorTarget>,
}

impl Default for GraphicsState {
    fn default() -> Self {
        Self {
            rasterizer: sharc::RasterizationState::default(),
            depth_stencil: sharc::DepthStencilState::default(),
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
        let color_targets = toml.get_optional_table_or_array("color_targets")?.ok_or(MissingField("color_targets"))?;
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

    /// Retrieves an optional string field from a TOML value as an owned string.
    ///
    /// Returns `Ok(None)` if the field is not present.
    /// Returns `Err(Error::InvalidType)` if the field is present but not a string.
    fn get_optional_string(&self, field: &'static str) -> Result<Option<String>, Error> {
        match self.get_optional_str(field)? {
            Some(s) => Ok(Some(s.to_string())),
            None => Ok(None),
        }
    }

    /// Retrieves an optional field that is either a string or an array of strings from a TOML value.
    ///
    /// Returns `Ok(None)` if the field is not present.
    /// Returns `Err(Error::InvalidType)` if the field is present but neither a string nor an array of strings.
    fn get_optional_str_or_array(&self, field: &'static str) -> Result<Option<Vec<&str>>, Error>;

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

    /// Retrieves a field array value.
    fn get_array(&self, field: &'static str) -> Result<&Vec<TomlValue>, Error>;

    /// Retrieves an optional enum value.
    fn get_optional_enum<T: Copy>(
        &self,
        field: &'static str,
        values: &[(&str, T)],
        expected: &'static str,
    ) -> Result<Option<T>, Error> {
        match self.get_optional_str(field)? {
            Some(s) => values
                .iter()
                .find(|(name, _)| *name == s)
                .map(|(_, value)| *value)
                .ok_or(Error::InvalidEnumValue(field, s.to_string(), expected))
                .map(Some),
            None => Ok(None),
        }
    }
}

static POLYGON_MODES: &[(&str, PolygonMode)] =
    &[("fill", PolygonMode::FILL), ("line", PolygonMode::LINE), ("point", PolygonMode::POINT)];

static CULL_MODES: &[(&str, vk::CullModeFlags)] = &[
    ("none", vk::CullModeFlags::NONE),
    ("front", vk::CullModeFlags::FRONT),
    ("back", vk::CullModeFlags::BACK),
    ("front_and_back", vk::CullModeFlags::FRONT_AND_BACK),
];

static FORMATS: &[(&str, vk::Format)] = &[
    ("RGBA8", vk::Format::R8G8B8A8_UNORM),
    ("RGBA8UI", vk::Format::R8G8B8A8_UINT),
    ("RGBA16UI", vk::Format::R16G16B16A16_UINT),
    ("RGB10_A2", vk::Format::A2B10G10R10_UNORM_PACK32),
    ("R32F", vk::Format::R32_SFLOAT),
    ("RG32F", vk::Format::R32G32_SFLOAT),
    ("RGBA32F", vk::Format::R32G32B32A32_SFLOAT),
    ("D32F", vk::Format::D32_SFLOAT),
    ("D32F_S8UI", vk::Format::D32_SFLOAT_S8_UINT),
];

static COMPARE_OPS: &[(&str, vk::CompareOp)] =
    &[("always", vk::CompareOp::ALWAYS), ("less", vk::CompareOp::LESS), ("lequal", vk::CompareOp::LESS_OR_EQUAL)];

static BLEND_FACTORS: &[(&str, vk::BlendFactor)] = &[
    ("zero", vk::BlendFactor::ZERO),
    ("one", vk::BlendFactor::ONE),
    ("src_alpha", vk::BlendFactor::SRC_ALPHA),
    ("one_minus_src_alpha", vk::BlendFactor::ONE_MINUS_SRC_ALPHA),
];

static BLEND_OPS: &[(&str, vk::BlendOp)] = &[
    ("add", vk::BlendOp::ADD),
    ("subtract", vk::BlendOp::SUBTRACT),
    ("reverse_subtract", vk::BlendOp::REVERSE_SUBTRACT),
];

impl TomlExt for toml::Value {
    fn get_array(&self, field: &'static str) -> Result<&Vec<TomlValue>, Error> {
        self.as_array().ok_or(InvalidType(field, "array"))
    }

    fn get_optional_str(&self, field: &'static str) -> Result<Option<&str>, Error> {
        match self.get(field) {
            None => Ok(None),
            Some(value) => value.as_str().ok_or(InvalidType(field, "string")).map(Some),
        }
    }

    fn get_optional_str_or_array(&self, field: &'static str) -> Result<Option<Vec<&str>>, Error> {
        match self.get(field) {
            None => Ok(None),
            Some(value) => {
                if let Some(s) = value.as_str() {
                    Ok(Some(vec![s]))
                } else if let Some(array) = value.as_array() {
                    let mut result = Vec::new();
                    for item in array {
                        if let Some(s) = item.as_str() {
                            result.push(s);
                        } else {
                            return Err(InvalidType(field, "string or array of strings"));
                        }
                    }
                    Ok(Some(result))
                } else {
                    Err(InvalidType(field, "string or array of strings"))
                }
            }
        }
    }

    fn get_optional_bool(&self, field: &'static str) -> Result<Option<bool>, Error> {
        match self.get(field) {
            None => Ok(None),
            Some(value) => value.as_bool().ok_or(InvalidType(field, "boolean")).map(Some),
        }
    }

    fn get_optional_integer(&self, field: &'static str) -> Result<Option<i64>, Error> {
        match self.get(field) {
            None => Ok(None),
            Some(value) => value.as_integer().ok_or(InvalidType(field, "integer")).map(Some),
        }
    }

    fn get_optional_float(&self, field: &'static str) -> Result<Option<f64>, Error> {
        match self.get(field) {
            None => Ok(None),
            Some(value) => value.as_float().ok_or(InvalidType(field, "float")).map(Some),
        }
    }

    fn get_optional_table(&self, field: &'static str) -> Result<Option<&TomlValue>, Error> {
        match self.get(field) {
            None => Ok(None),
            Some(value) => value.as_table().ok_or(InvalidType(field, "table")).map(|_| Some(value)),
        }
    }

    fn get_optional_array(&self, field: &'static str) -> Result<Option<&Vec<TomlValue>>, Error> {
        match self.get(field) {
            None => Ok(None),
            Some(value) => value.as_array().ok_or(InvalidType(field, "array")).map(|arr| Some(arr)),
        }
    }

    fn get_optional_table_or_array(&self, field: &'static str) -> Result<Option<&TomlValue>, Error> {
        match self.get(field) {
            None => Ok(None),
            Some(value) => {
                if value.is_table() || value.is_array() {
                    Ok(Some(value))
                } else {
                    Err(InvalidType(field, "table or array"))
                }
            }
        }
    }
}

fn read_rasterizer_state(toml: &TomlValue, out: &mut sharc::RasterizationState) -> Result<(), Error> {
    //let cull_mode = read_str(json, "cull_mode", Some("back"))?;

    let polygon_mode = toml.get_optional_enum("polygon_mode", POLYGON_MODES, "<polygon mode>")?;
    let cull_mode = toml.get_optional_enum("cull_mode", CULL_MODES, "<cull mode>")?;

    if let Some(polygon_mode) = polygon_mode {
        out.polygon_mode = polygon_mode;
    }
    if let Some(cull_mode) = cull_mode {
        out.cull_mode = cull_mode;
    }

    Ok(())
}
/*
fn get_format(fmtstr: &str) -> Option<vk::Format> {
    match fmtstr {
        "RGBA8" => Some(vk::Format::R8G8B8A8_UNORM),
        "RGBA8UI" => Some(vk::Format::R8G8B8A8_UINT),
        "RGBA16UI" => Some(vk::Format::R16G16B16A16_UINT),
        "RGB10_A2" => Some(vk::Format::A2B10G10R10_UNORM_PACK32),
        "R32F" => Some(vk::Format::R32_SFLOAT),
        "RG32F" => Some(vk::Format::R32G32_SFLOAT),
        "RGBA32F" => Some(vk::Format::R32G32B32A32_SFLOAT),
        "D32F" => Some(vk::Format::D32_SFLOAT),
        "D32F_S8UI" => Some(vk::Format::D32_SFLOAT_S8_UINT),
        _ => {
            None
        }
    }
}

fn get_blend_factor(factor_str: &str) -> Option<vk::BlendFactor> {
    match factor_str {
        "zero" => Some(vk::BlendFactor::ZERO),
        "one" => Some(vk::BlendFactor::ONE),
        "src_alpha" => Some(vk::BlendFactor::SRC_ALPHA),
        "one_minus_src_alpha" => Some(vk::BlendFactor::ONE_MINUS_SRC_ALPHA),
        _ => {
            None
        }
    }
}

fn get_blend_op(op_str: &str) -> Option<vk::BlendOp> {
    match op_str {
        "add" => Some(vk::BlendOp::ADD),
        "subtract" => Some(vk::BlendOp::SUBTRACT),
        "reverse_subtract" => Some(vk::BlendOp::REVERSE_SUBTRACT),
        _ => {
            None
        }
    }
}*/

fn read_depth_stencil_state(toml: &TomlValue, out: &mut sharc::DepthStencilState) -> anyhow::Result<()> {
    // any depth-stencil field automatically enables depth testing
    if let Some(format) = toml.get_optional_enum("format", FORMATS, "<image format>")? {
        out.format = format;
        out.enable = true;
    }
    if let Some(depth_compare_op) = toml.get_optional_enum("compare_op", COMPARE_OPS, "<compare op>")? {
        out.depth_compare_op = depth_compare_op;
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

fn read_blend(toml: &TomlValue) -> anyhow::Result<Option<ColorBlendEquation>> {
    if let Some(str) = toml.as_str() {
        match str {
            "disabled" => Ok(None),
            "over" => Ok(Some(ColorBlendEquation {
                src_color_blend_factor: vk::BlendFactor::SRC_ALPHA,
                dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
                color_blend_op: vk::BlendOp::ADD,
                src_alpha_blend_factor: vk::BlendFactor::ONE,
                dst_alpha_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
                alpha_blend_op: vk::BlendOp::ADD,
            })),
            "over_premultiplied" => Ok(Some(ColorBlendEquation {
                src_color_blend_factor: vk::BlendFactor::ONE,
                dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
                color_blend_op: vk::BlendOp::ADD,
                src_alpha_blend_factor: vk::BlendFactor::ONE,
                dst_alpha_blend_factor: vk::BlendFactor::ZERO,
                alpha_blend_op: vk::BlendOp::ADD,
            })),
            _ => Err(anyhow!("unknown predefined blend mode").context("in blend")),
        }
    } else {
        validate_keys(toml, &[], &["src_color", "dst_color", "color_op", "src_alpha", "dst_alpha", "alpha_op"])
            .context("in blend")?;

        let mut blend = ColorBlendEquation::default();
        if let Some(src_color_blend_factor) = toml.get_optional_enum("src_color", BLEND_FACTORS, "<blend factor>")? {
            blend.src_color_blend_factor = src_color_blend_factor;
        }
        if let Some(dst_color_blend_factor) = toml.get_optional_enum("dst_color", BLEND_FACTORS, "<blend factor>")? {
            blend.dst_color_blend_factor = dst_color_blend_factor;
        }
        if let Some(color_blend_op) = toml.get_optional_enum("color_op", BLEND_OPS, "<blend op>")? {
            blend.color_blend_op = color_blend_op;
        }
        if let Some(src_alpha_blend_factor) = toml.get_optional_enum("src_alpha", BLEND_FACTORS, "<blend factor>")? {
            blend.src_alpha_blend_factor = src_alpha_blend_factor;
        }
        if let Some(dst_alpha_blend_factor) = toml.get_optional_enum("dst_alpha", BLEND_FACTORS, "<blend factor>")? {
            blend.dst_alpha_blend_factor = dst_alpha_blend_factor;
        }
        if let Some(alpha_blend_op) = toml.get_optional_enum("alpha_op", BLEND_OPS, "<blend op>")? {
            blend.alpha_blend_op = alpha_blend_op;
        }
        Ok(Some(blend))
    }
}

fn read_color_target(toml: &TomlValue, out: &mut sharc::ColorTarget) -> anyhow::Result<()> {
    validate_keys(toml, &[], &["format", "blend"]).context("in color target")?;
    if let Some(format) = toml.get_optional_enum("format", FORMATS, "<image format>")? {
        out.format = format;
    }
    if let Some(blend_toml) = toml.get("blend") {
        out.blend = read_blend(blend_toml)?;
    }
    Ok(())
}

fn read_color_targets(toml: &TomlValue, out: &mut Vec<sharc::ColorTarget>) -> anyhow::Result<()> {
    if let Some(array) = toml.as_array() {
        out.clear();
        for item in array {
            let mut color_target = sharc::ColorTarget::default();
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
                    out.resize(index + 1, sharc::ColorTarget::default());
                }
                read_color_target(value, &mut out[index]).context("in color_targets")?;
            }
        }
        Ok(())
    } else {
        return Err(InvalidType("color_targets", "array of color target descriptions").into());
    }
}
