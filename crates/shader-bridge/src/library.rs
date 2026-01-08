use crate::error::Error;
use crate::session::{create_session, SessionOptions};
use crate::SHADER_PROFILE;
use gpu::vk;
use slang::reflection::{TypeLayout};
use slang::Downcast;
use std::path::{Path, PathBuf};

/// Additional options for loading a `ShaderLibrary`.
#[derive(Default)]
pub struct ShaderLibraryLoadOptions {
    /// Additional search paths for module dependencies.
    pub module_search_paths: Vec<String>,
    /// List of macro definitions to apply during compilation.
    pub macro_definitions: Vec<(String, String)>,
    /// Shader profile to use when compiling the shader library.
    ///
    /// Defaults to "glsl_460" if unspecified.
    pub shader_profile: Option<String>,
    /// Emits debug information during shader compilation.
    pub debug: bool,
}

/// Information about a root parameter.
pub struct RootParamInfo {
    /// Name of the root parameter.
    pub name: String,
    pub offset: u32,
    pub size: u32,
    /// Vulkan format of the root parameter.
    pub format: vk::Format,
    /// Optional render world binding associated with the root parameter.
    pub render_world_binding: Option<String>,
}

/// Describes a shader entry point.
pub struct EntryPoint {
    /// Name of the entry point function.
    pub name: String,
    /// Shader stage.
    pub stage: gpu::ShaderStage,
    /// Pass attribute
    pub pass: Option<String>,
}

/// Represents information about a compiled shader entry point.
pub struct CompiledEntryPoint {
    /// Name of the entry point function.
    pub name: String,
    /// SPIR-V bytecode blob as a vector of `u32`.
    pub spirv: Vec<u32>,
    /// Size of push constants used by the entry point, in bytes.
    pub push_constants_size: usize,
    /// Shader stage.
    pub stage: gpu::ShaderStage,
    /// Work group size for compute shaders.
    pub work_group_size: [u32; 3],
    /// Source file path of the shader.
    pub path: Option<String>,
    /// Reflection information.
    program: slang::ComponentType,
}

impl CompiledEntryPoint {
    pub fn as_gpu_entry_point(&self) -> gpu::ShaderEntryPoint<'_> {
        gpu::ShaderEntryPoint {
            stage: self.stage,
            code: &self.spirv,
            entry_point: &self.name,
            push_constants_size: self.push_constants_size,
            source_path: self.path.as_deref(),
            workgroup_size: self.work_group_size,
        }
    }
}

/// Represents a loaded and parsed shader source file (in slang language).
///
/// This can be used to retrieve compiled SPIR-V code for entry points defined in the shader library.
pub struct ShaderLibrary {
    path: Option<String>,
    _options: ShaderLibraryLoadOptions,
    session: slang::Session,
    module: slang::Module,
}

impl ShaderLibrary {
    /// Loads a shader library from the specified path.
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self, Error> {
        Self::new_inner(path.as_ref(), ShaderLibraryLoadOptions::default())
    }

    /// Loads a shader library from the specified path, specifying search paths & macro definitions.
    pub fn with_options<P: AsRef<Path>>(path: P, options: ShaderLibraryLoadOptions) -> Result<Self, Error> {
        Self::new_inner(path.as_ref(), options)
    }

    fn new_inner(path: &Path, options: ShaderLibraryLoadOptions) -> Result<Self, Error> {
        let path = path
            .canonicalize()?
            .to_str()
            .expect("path is not valid UTF-8")
            .to_string();
        let profile = options.shader_profile.as_deref().unwrap_or(SHADER_PROFILE);
        let session = create_session(&SessionOptions {
            profile_id: profile,
            module_search_paths: &options.module_search_paths,
            macro_definitions: &options.macro_definitions,
            debug: options.debug,
        });
        let module = session.load_module(&path)?;

        Ok(Self {
            path: Some(path),
            _options: options,
            session,
            module,
        })
    }

    /// Returns the list of entry points defined in this shader library.
    pub fn entry_points(&self) -> Vec<EntryPoint> {
        let layout = self.module.downcast().layout(0).expect("failed to get layout");
        let count = layout.entry_point_count();
        let mut entry_points = Vec::with_capacity(count as usize);
        for i in 0..count {
            let ep = layout.entry_point_by_index(i).unwrap();

            // `[pass("...")]` attribute
            let mut pass = None;
            for attr in ep.function().user_attributes() {
                if attr.name() == "pass" {
                    pass = attr.argument_value_string(0).map(String::from);
                }
            }

            entry_points.push(EntryPoint {
                name: ep.name().to_string(),
                stage: slang_stage_to_gpu_stage(ep.stage()),
                pass,
            });
        }
        entry_points
    }

    /// Returns compiled SPIR-V code for the specified entry point in the shader library.
    pub fn get_compiled_entry_point(&self, entry_point_name: &str) -> Result<CompiledEntryPoint, Error> {
        let entry_point = self
            .module
            .find_entry_point_by_name(entry_point_name)
            .ok_or_else(|| Error::EntryPointNotFound(entry_point_name.to_string()))?;

        let program = self
            .session
            .create_composite_component_type(&[self.module.downcast().clone(), entry_point.downcast().clone()])?;
        let program = program.link()?;

        let blob = {
            let blob = program.entry_point_code(0, 0)?;
            convert_spirv_u8_to_u32(blob.as_slice())
        };

        let reflection = program.layout(0).expect("failed to get reflection");
        let ep_refl = reflection.entry_point_by_index(0).unwrap();

        Ok(CompiledEntryPoint {
            spirv: blob,
            push_constants_size: get_push_constants_size(&ep_refl),
            stage: slang_stage_to_gpu_stage(ep_refl.stage()),
            work_group_size: {
                let s = ep_refl.compute_thread_group_size();
                [s[0] as u32, s[1] as u32, s[2] as u32]
            },
            name: entry_point_name.to_string(),
            path: self.path.clone(),
            program,
        })
    }

    /// Returns the list of all source files that were included in this shader library.
    pub fn source_paths(&self) -> Vec<PathBuf> {
        // TODO: this currently only returns the main source file, need to find a way to get
        //       referenced modules as well
        if let Some(path) = &self.path {
            vec![PathBuf::from(path)]
        } else {
            vec![]
        }
    }
}

pub(crate) fn convert_spirv_u8_to_u32(bytes: &[u8]) -> Vec<u32> {
    assert_eq!(bytes.len() % 4, 0, "invalid SPIR-V code length");
    bytes
        .chunks_exact(4)
        .map(|chunk| {
            let bytes: [u8; 4] = chunk.try_into().unwrap();
            u32::from_ne_bytes(bytes)
        })
        .collect::<Vec<u32>>()
}

/// Returns the size of push constants used by the specified entry point.
pub(crate) fn get_push_constants_size(entry_point: &slang::reflection::EntryPoint) -> usize {
    // push constant are entry point function parameters with kind uniform
    let mut size = 0;
    for p in entry_point.parameters() {
        // There's a PushConstantBuffer category, but it doesn't seem to be used
        if p.category() == slang::ParameterCategory::Uniform {
            size += p.type_layout().size(slang::ParameterCategory::Uniform);
        }
    }
    size
}

fn slang_stage_to_gpu_stage(stage: slang::Stage) -> gpu::ShaderStage {
    match stage {
        slang::Stage::Vertex => gpu::ShaderStage::Vertex,
        slang::Stage::Hull => gpu::ShaderStage::TessControl,
        slang::Stage::Domain => gpu::ShaderStage::TessEvaluation,
        slang::Stage::Geometry => gpu::ShaderStage::Geometry,
        slang::Stage::Fragment => gpu::ShaderStage::Fragment,
        slang::Stage::Compute => gpu::ShaderStage::Compute,
        _ => panic!("unsupported shader stage: {:?}", stage),
    }
}
