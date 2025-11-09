//! Pipeline build manifest
use crate::Error::MissingField;
use anyhow::{Context, anyhow};
use log::error;
use pipeline_archive::ColorBlendEquationData;
use pipeline_archive::gpu::vk;
use pipeline_archive::gpu::vk::PolygonMode;
use serde_json::Value as Json;
use std::collections::{BTreeMap};
use std::path::{Path, PathBuf};

const MAX_COLOR_TARGETS: usize = 8;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("missing field: {0}")]
    MissingField(&'static str),
    #[error("invalid field type for field {0}")]
    InvalidType(&'static str),
    #[error("{0}")]
    Other(&'static str),
    #[error("{0}")]
    CompilationError(String),
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum PipelineType {
    Graphics,
    Compute,
}

#[derive(Clone, Debug, Eq, PartialOrd, PartialEq, Ord)]
pub struct Tag(pub String);

impl Tag {
    pub fn set(&self) -> &str {
        self.0.split_once('=').map(|(key, _value)| key).unwrap_or(&self.0)
    }

    pub fn value(&self) -> Option<&str> {
        self.0.split_once('=').map(|(_key, value)| value)
    }
}

#[derive(Clone, Debug)]
pub struct Variant {
    /// TODO remove variant tags, instead put keywords in the name of the pipeline directly
    pub tag: Tag,
    pub overrides: Json,
}

impl Default for Variant {
    fn default() -> Self {
        // dummy variant for convenience when computing permutations
        Variant {
            tag: Tag(String::new()),
            overrides: Json::Object(serde_json::Map::new()),
        }
    }
}

#[derive(Clone)]
pub struct Configuration {
    pub defines: BTreeMap<String, String>,
    pub rasterization_state: pipeline_archive::RasterizerStateData,
    pub depth_stencil_state: pipeline_archive::DepthStencilStateData,
    pub color_targets: Vec<pipeline_archive::ColorTarget>,
}

#[derive(Clone)]
pub struct Input {
    pub file_path: String,
    pub name: String,
    pub vertex_entry_point: String,
    pub fragment_entry_point: String,
    pub compute_entry_point: String,
    pub task_entry_point: String,
    pub mesh_entry_point: String,
    pub overrides: Option<Json>,
}

#[derive(Clone)]
pub struct BuildManifest {
    pub type_: PipelineType,
    pub manifest_path: PathBuf,
    pub inputs: Vec<Input>,
    /// Output archive path.
    pub output: String,
    pub base_configuration: Configuration,
}

//////////////////////////////////////////////////////////////////////////////////////////////////
fn read_str<'a>(json: &'a Json, field: &'static str) -> Result<Option<&'a str>, Error> {
    match json.get(field) {
        None => Ok(None),
        Some(value) => value.as_str().ok_or(Error::InvalidType(field)).map(Some),
    }
}

fn read_f32(json: &Json, field: &'static str) -> Result<Option<f32>, Error> {
    match json.get(field) {
        None => Ok(None),
        Some(value) => value.as_f64().ok_or(Error::InvalidType(field)).map(|v| Some(v as f32)),
    }
}

fn read_bool(json: &Json, field: &'static str) -> Result<Option<bool>, Error> {
    match json.get(field) {
        None => Ok(None),
        Some(value) => value.as_bool().ok_or(Error::InvalidType(field)).map(Some),
    }
}

fn read_object<'a>(json: &'a Json, field: &'static str) -> Result<Option<&'a Json>, Error> {
    match json.get(field) {
        None => Ok(None),
        Some(value) => value.as_object().ok_or(Error::InvalidType(field)).map(|_| Some(value)),
    }
}

fn read_array<'a>(json: &'a Json, field: &'static str) -> Result<Option<&'a Vec<Json>>, Error> {
    match json.get(field) {
        None => Ok(None),
        Some(value) => value.as_array().ok_or(Error::InvalidType(field)).map(|arr| Some(arr)),
    }
}

fn read_rasterizer_state(json: &Json, out: &mut pipeline_archive::RasterizerStateData) -> Result<(), Error> {
    //let cull_mode = read_str(json, "cull_mode", Some("back"))?;

    if let Some(polygon_mode) = read_str(json, "polygon_mode")? {
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

    if let Some(cull_mode) = read_str(json, "cull_mode")? {
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

fn read_format(fmtstr: &str) -> Result<vk::Format, Error> {
    match fmtstr {
        "RGBA8" => Ok(vk::Format::R8G8B8A8_UNORM),
        "D32F" => Ok(vk::Format::D32_SFLOAT),
        "D32F_S8UI" => Ok(vk::Format::D32_SFLOAT_S8_UINT),
        _ => {
            error!("Unknown format: {}", fmtstr);
            Err(Error::InvalidType("format"))
        }
    }
}

fn read_blend_factor(factor_str: &str) -> Result<vk::BlendFactor, Error> {
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

fn read_blend_op(op_str: &str) -> Result<vk::BlendOp, Error> {
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

fn read_depth_stencil_state(json: &Json, out: &mut pipeline_archive::DepthStencilStateData) -> Result<(), Error> {
    if let Some(format_str) = read_str(json, "format")? {
        out.format = read_format(format_str)?;
    }
    if let Some(depth_compare_op) = read_str(json, "depth_compare_op")? {
        out.depth_compare_op = match depth_compare_op {
            "always" => vk::CompareOp::ALWAYS,
            "less" => vk::CompareOp::LESS,
            "lequal" => vk::CompareOp::LESS_OR_EQUAL,
            _ => {
                error!("Unknown depth compare op: {}", depth_compare_op);
                vk::CompareOp::ALWAYS
            }
        };
    }
    if let Some(depth_write_enable) = read_bool(json, "depth_write_enable")? {
        out.depth_write_enable = depth_write_enable;
    }
    out.depth_test_enable = true;
    Ok(())
}


fn read_color_target(json: &Json, out: &mut pipeline_archive::ColorTarget) -> Result<(), Error> {
    if let Some(format_str) = read_str(json, "format")? {
        out.format = read_format(format_str)?;
    }
    if let Some(src_color_blend_factor) = read_str(json, "src_color_blend_factor")? {
        out.blend.src_color_blend_factor = read_blend_factor(src_color_blend_factor)?;
    }
    if let Some(dst_color_blend_factor) = read_str(json, "dst_color_blend_factor")? {
        out.blend.dst_color_blend_factor = read_blend_factor(dst_color_blend_factor)?;
    }
    if let Some(color_blend_op) = read_str(json, "color_blend_op")? {
        out.blend.color_blend_op = read_blend_op(color_blend_op)?;
    }
    if let Some(src_alpha_blend_factor) = read_str(json, "src_alpha_blend_factor")? {
        out.blend.src_alpha_blend_factor = read_blend_factor(src_alpha_blend_factor)?;
    }
    if let Some(dst_alpha_blend_factor) = read_str(json, "dst_alpha_blend_factor")? {
        out.blend.dst_alpha_blend_factor = read_blend_factor(dst_alpha_blend_factor)?;
    }
    if let Some(alpha_blend_op) = read_str(json, "alpha_blend_op")? {
        out.blend.alpha_blend_op = read_blend_op(alpha_blend_op)?;
    }

    Ok(())
}

fn read_color_targets(json: &Json, out: &mut Vec<pipeline_archive::ColorTarget>) -> Result<(), Error> {
    if let Some(array) = json.as_array() {
        out.clear();
        for item in array {
            let mut color_target = pipeline_archive::ColorTarget::default();
            read_color_target(item, &mut color_target)?;
            out.push(color_target);
        }
        Ok(())
    } else if let Some(object) = json.as_object() {
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
                    return Err(Error::Other("color target index out of range"));
                }
                if index >= out.len() {
                    out.resize(index + 1, pipeline_archive::ColorTarget::default());
                }
                read_color_target(value, &mut out[index])?;
            }
        }
        Ok(())
    } else {
        return Err(Error::InvalidType("color_targets"));
    }
}

impl Configuration {
    pub fn apply_overrides(&mut self, overrides: &Json) -> Result<(), Error> {
        if let Some(rasterizer_obj) = read_object(overrides, "rasterizer_state")? {
            read_rasterizer_state(rasterizer_obj, &mut self.rasterization_state)?;
        }
        if let Some(depth_stencil_obj) = read_object(overrides, "depth_stencil_state")? {
            read_depth_stencil_state(depth_stencil_obj, &mut self.depth_stencil_state)?;
        }
        if let Some(color_targets_json) = read_object(overrides, "color_targets")? {
            read_color_targets(color_targets_json, &mut self.color_targets)?;
        }

        Ok(())
    }
}

impl Input {
    fn from_json(json: &Json, type_: PipelineType) -> anyhow::Result<Input> {
        let file_path = read_str(json, "file_path")?
            .ok_or(MissingField("file_path"))?
            .to_string();
        let name = read_str(json, "name")?.ok_or(MissingField("name"))?.to_string();
        let vertex_entry_point = read_str(json, "vertex_entry_point")?.unwrap_or("").to_string();
        let fragment_entry_point = read_str(json, "fragment_entry_point")?.unwrap_or("").to_string();
        let compute_entry_point = read_str(json, "compute_entry_point")?.unwrap_or("").to_string();
        let task_entry_point = read_str(json, "task_entry_point")?.unwrap_or("").to_string();
        let mesh_entry_point = read_str(json, "mesh_entry_point")?.unwrap_or("").to_string();

        // check that a valid combination of entry points is specified
        let vertex = !vertex_entry_point.is_empty();
        let fragment = !fragment_entry_point.is_empty();
        let compute = !compute_entry_point.is_empty();
        let task = !task_entry_point.is_empty();
        let mesh = !mesh_entry_point.is_empty();
        match (vertex, task, mesh, fragment, compute) {
            (true, false, false, true, false) => {
                if type_ != PipelineType::Graphics {
                    return Err(anyhow!("mismatch between pipeline type and entry points"));
                }
            }
            (false, _, true, true, false) => {
                if type_ != PipelineType::Graphics {
                    return Err(anyhow!("mismatch between pipeline type and entry points"));
                }
            }
            (false, false, false, false, true) => {
                if type_ != PipelineType::Compute {
                    return Err(anyhow!("mismatch between pipeline type and entry points"));
                }
            }
            _ => {
                return Err(anyhow!("invalid combination of entry points specified"));
            }
        }

        let overrides = read_object(json, "overrides")?.cloned();

        Ok(Input {
            file_path,
            name,
            vertex_entry_point,
            fragment_entry_point,
            compute_entry_point,
            task_entry_point,
            mesh_entry_point,
            overrides,
        })
    }
}

impl BuildManifest {


    pub(crate) fn load(path: impl AsRef<Path>) -> Result<BuildManifest, anyhow::Error> {
        fn load_inner(path: &Path) -> Result<BuildManifest, anyhow::Error> {
            let manifest_str = std::fs::read_to_string(&path)?;
            let manifest_json: Json = serde_json::from_str(&manifest_str).context("invalid json")?;
            BuildManifest::from_json(path, &manifest_json).context("failed to parse manifest")
        }
        load_inner(path.as_ref())
    }

    fn from_json(manifest_path: &Path, json: &Json) -> Result<BuildManifest, anyhow::Error> {
        let type_str = read_str(json, "type")?.ok_or(MissingField("type"))?;
        let type_ = match type_str {
            "graphics" => PipelineType::Graphics,
            "compute" => PipelineType::Compute,
            _ => {
                return Err(anyhow!("Unknown pipeline type: {}", type_str));
            }
        };
        let inputs = if let Some(inputs_array) = read_array(json, "inputs")? {
            let mut inputs = Vec::new();
            for (i, input_json) in inputs_array.iter().enumerate() {
                let input = Input::from_json(input_json, type_).with_context(|| format!("in inputs[{i}]"))?;
                inputs.push(input);
            }
            inputs
        } else {
            return Err(MissingField("inputs").into());
        };

        let output = read_str(json, "output")?.ok_or(MissingField("output"))?.to_string();

        let rasterizer_state =
            if let Some(rasterizer_obj) = read_object(json, "rasterizer_state").context("in rasterizer_state")? {
                let mut rasterization_state = pipeline_archive::RasterizerStateData::default();
                read_rasterizer_state(rasterizer_obj, &mut rasterization_state)?;
                rasterization_state
            } else {
                pipeline_archive::RasterizerStateData::default()
            };
        let depth_stencil_state = if let Some(depth_stencil_obj) =
            read_object(json, "depth_stencil_state").context("in depth_stencil_state")?
        {
            let mut depth_stencil_state = pipeline_archive::DepthStencilStateData::default();
            read_depth_stencil_state(depth_stencil_obj, &mut depth_stencil_state)?;
            depth_stencil_state
        } else {
            pipeline_archive::DepthStencilStateData::default()
        };

        let mut color_targets = Vec::new();
        // color targets array is mandatory
        let color_targets_json = json.get("color_targets").ok_or(MissingField("color_targets"))?;
        read_color_targets(color_targets_json, &mut color_targets)?;


        Ok(BuildManifest {
            type_,
            output,
            manifest_path: manifest_path.to_path_buf(),
            inputs,
            base_configuration: Configuration {
                defines: Default::default(),
                rasterization_state: rasterizer_state,
                depth_stencil_state,
                color_targets,
            },
        })
    }
}
