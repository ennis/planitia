use crate::error::Error;
use crate::session::create_session;
use crate::SHADER_PROFILE;
use slang::Downcast;
use std::path::Path;

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
}

/// Represents information about a compiled shader entry point.
pub struct ShaderEntryPointInfo {
    /// SPIR-V bytecode blob as a vector of `u32`.
    pub spirv: Vec<u32>,
    /// Size of push constants used by the entry point, in bytes.
    pub push_constants_size: usize,
    pub stage: gpu::ShaderStage,
    pub work_group_size: [u32; 3],
    pub entry_point_name: String,
    pub path: Option<String>,
}

impl ShaderEntryPointInfo {
    pub fn as_gpu_entry_point(&self) -> gpu::ShaderEntryPoint<'_> {
        gpu::ShaderEntryPoint {
            stage: self.stage,
            code: &self.spirv,
            entry_point: &self.entry_point_name,
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
        let session = create_session(profile, &options.module_search_paths, &options.macro_definitions);
        let module = session.load_module(&path)?;

        Ok(Self {
            path: Some(path),
            _options: options,
            session,
            module,
        })
    }

    /// Returns compiled SPIR-V code for the specified entry point in the shader library.
    pub fn get_compiled_entry_point(&self, entry_point_name: &str) -> Result<ShaderEntryPointInfo, Error> {
        let entry_point = self
            .module
            .find_entry_point_by_name(entry_point_name)
            .ok_or_else(|| Error::EntryPointNotFound(entry_point_name.to_string()))?;

        let program = self
            .session
            .create_composite_component_type(&[self.module.downcast().clone(), entry_point.downcast().clone()])?;
        let program = program.link()?;
        let blob = program.entry_point_code(0, 0)?;

        // dump spirv for debugging
        /*use std::fs::File;
        use std::io::Write;
        let mut f = File::create(format!("shader_{entry_point_name}.spv")).unwrap();
        f.write_all(blob.as_slice()).unwrap();*/

        let blob = convert_spirv_u8_to_u32(blob.as_slice());

        let reflection = program.layout(0).expect("failed to get reflection");
        let ep_refl = reflection.entry_point_by_index(0).unwrap();



        Ok(ShaderEntryPointInfo {
            spirv: blob,
            push_constants_size: get_push_constants_size(&ep_refl),
            stage: slang_stage_to_gpu_stage(ep_refl.stage()),
            work_group_size: {
                let s = ep_refl.compute_thread_group_size();
                [s[0] as u32, s[1] as u32, s[2] as u32]
            },
            entry_point_name: entry_point_name.to_string(),
            path: self.path.clone(),
        })
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
