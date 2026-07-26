#![feature(default_field_values)]
mod build;
//mod dump;
mod archive_writer;
mod dump2;
mod header;
mod manifest;
mod reflection;

use anyhow::anyhow;
use color_print::{ceprintln, cprintln};
use log::warn;
pub use manifest::*;
use scoped_tls::scoped_thread_local;
use std::path::{Path, PathBuf};
use std::time::SystemTime;
use std::{fs, io};
use thiserror::Error;

use crate::archive_writer::build_and_write_archive;
use crate::build::{compile_slang_module, create_slang_session};
use crate::reflection::Param;
pub use dump2::dump_archive_file;
use sharc::gpu_types::vk;
//--------------------------------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct LogOptions {
    /// Don't print logs to stdout.
    pub quiet: bool = false,
    /// Verbosity level
    pub verbosity: u8 = 0,
}

scoped_thread_local!(pub(crate) static LOG_OPTIONS: LogOptions);

pub(crate) fn get_log_options() -> LogOptions {
    if LOG_OPTIONS.is_set() {
        LOG_OPTIONS.with(|options| options.clone())
    } else {
        LogOptions { quiet: true, verbosity: 0 }
    }
}

/// Generic error type wrapper.
#[derive(Error, Debug)]
#[error(transparent)]
pub enum Error {
    /// Errors from the slang compiler.
    #[error("{0}")]
    CompilerErrors(String),
    /// Other errors.
    Other(#[from] anyhow::Error),
}

impl From<io::Error> for Error {
    fn from(err: io::Error) -> Self {
        Error::Other(anyhow!(err))
    }
}

impl Error {
    pub fn print_cargo_error(&self) {
        let fmt = format!("{:#}", self);
        for line in fmt.lines() {
            println!("cargo::error={line}");
        }
    }
}

//--------------------------------------------------------------------------------------------------

pub type EntryPointIndex = usize;

/// Represents a shader pipeline.
pub struct Pipeline {
    pub name: String,
    /// Indices in [`Module::entry_points`].
    pub stages: Vec<EntryPointIndex>,
    pub graphics_state: GraphicsState,
    pub workgroup_size: [u32; 3],
    pub push_constants_size: usize,
}

#[derive(Clone, Debug)]
pub struct ModuleDependency {
    pub path: PathBuf,
    pub mtime: u64
}

/// A compiled Slang shader module with all its entry points.
pub struct Module {
    // Keep the session alive for the lifetime of the module.
    _session: slang::Session,
    pub name: String,
    pub module: slang::Module,
    pub program: slang::ComponentType,
    pub file_path: PathBuf,
    pub file_mtime: u64,
    pub spirv: Vec<u32>,
    //pub reflection: Vec<Param>,
    pub entry_points: Vec<EntryPoint>,
    pub pipelines: Vec<Pipeline>,
    /// List of all slang module dependencies (including transitive dependencies).
    pub dependencies: Vec<ModuleDependency>,
}

/// A single shader entry point extracted from a [`Module`].
pub struct EntryPoint {
    pub name: String,
    //pub params: Vec<Param>,
    /// The pipeline that this entry point belongs to, either from a `[pipeline("...")]` attribute or inferred
    /// from the entry point name by stripping the stage suffix.
    pub pipeline_name: Option<String>,
    pub stage: vk::ShaderStageFlags,
    pub push_constants_size: usize,
    pub workgroup_size: [u32; 3],
}

//--------------------------------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct BuildOptions {
    /// Emit cargo dependency information.
    pub emit_cargo_deps: bool,
    /// Emit shader debug information.
    pub emit_debug_information: bool,
    /// Dumps SPIR-V binaries to disk.
    pub emit_spirv_binaries: bool,
    pub emit_reflection: bool,
    pub include_paths: Vec<PathBuf>,
    /// Output directory
    pub output_directory: Option<PathBuf> = None,
}

static KNOWN_METADATA: &[&str] = &["Manifest"];

fn check_metadata(metadata: &std::collections::BTreeMap<String, String>) -> Result<(), anyhow::Error> {
    for key in metadata.keys() {
        if !KNOWN_METADATA.contains(&key.as_str()) {
            ceprintln!("<y,bold>warning</>: unknown metadata key: `{}`", key);
        }
    }
    Ok(())
}

/// Loads the companion manifest file for a shader source.
///
/// The function first looks for a `Manifest` metadata header in the shader source. If found, it uses the specified path to load the manifest.
/// Otherwise, it looks for a manifest file with the same name as the shader file but with a `.toml` extension in the same directory.
///
/// # Return value
///
/// If no manifest file is found, an empty manifest is returned (this is not an error condition).
/// This function returns an error if the manifest file is specified but does not exist,
/// or if the manifest is malformed or cannot be loaded for any other reason.
fn load_manifest_for_source(shader_file_path: &Path, shader_source: &str) -> Result<BuildManifest, Error> {
    if get_log_options().verbosity >= 2 {
        ceprintln!("parsing metadata header: {}", shader_file_path.display());
    }

    let metadata = header::parse_metadata_header(&shader_source);
    check_metadata(&metadata)?;

    if get_log_options().verbosity >= 2 {
        if metadata.is_empty() {
            ceprintln!("no metadata found");
        } else {
            for (key, value) in &metadata {
                ceprintln!("metadata: `{}: {}`", key, value);
            }
        }
    }

    let mut manifest_path = None;

    if let Some(path) = metadata.get("Manifest") {
        let path = shader_file_path.parent().unwrap().join(path);
        if !path.exists() {
            return Err(anyhow!(
                "manifest file {} specified in {} does not exist",
                path.display(),
                shader_file_path.display()
            )
            .into());
        }
        manifest_path = Some(path.to_path_buf());
    } else {
        // Try to find a manifest file in the same directory as the shader file.
        let path = shader_file_path.with_extension(".toml");
        if path.exists() {
            manifest_path = Some(path);
        }
    }

    let mut manifest = BuildManifest::default();
    if let Some(manifest_path) = manifest_path {
        if get_log_options().verbosity >= 1 {
            cprintln!("load manifest file: {}", manifest_path.display());
        }
        manifest.load(&manifest_path)?;
    }
    Ok(manifest)
}

/// Compiles slang shader files to shader archives (.sharc).
///
/// # Arguments
/// * `shader_files` - path to the shader files. Glob patterns are supported, e.g. `shaders/**/*.slang`.
pub fn build<'a, I>(glob_patterns: I, options: &BuildOptions, log_options: &LogOptions) -> Result<(), Error>
where
    I: IntoIterator<Item = &'a str>,
{
    let glob_patterns: Vec<&str> = glob_patterns.into_iter().collect();
    LOG_OPTIONS.set(log_options, || build_inner(&glob_patterns, options))
}

fn build_inner(glob_patterns: &[&str], options: &BuildOptions) -> Result<(), Error> {
    // resolve glob patterns
    let mut files = Vec::new();
    for pattern in glob_patterns {
        let entries = glob::glob(pattern).map_err(|err| anyhow!("failed to parse glob pattern '{pattern}': {err}"))?;
        for entry in entries {
            match entry {
                Ok(path) => {
                    files.push(path);
                }
                Err(err) => {
                    warn!("failed to resolve glob pattern: {err}");
                }
            }
        }
    }

    if files.is_empty() {
        return Err(anyhow!("no input files matching patterns: {:?}", glob_patterns).into());
    }

    let mut has_errors = false;

    for file in files {
        if get_log_options().verbosity >= 1 {
            cprintln!("load source file: {}", file.display());
        }

        // Load source file.
        let source = match fs::read_to_string(&file) {
            Ok(source) => source,
            Err(err) => {
                ceprintln!("<r,bold>error</>: failed to read source file {}: {}", file.display(), err);
                has_errors = true;
                continue;
            }
        };

        // Read manifest.
        let manifest = match load_manifest_for_source(&file, &source) {
            Ok(manifest) => manifest,
            Err(err) => {
                ceprintln!("<r,bold>error</>: failed to load manifest for source file {}: {}", file.display(), err);
                has_errors = true;
                continue;
            }
        };

        if let Err(err) = build_and_write_archive(&file, &source, &manifest, options) {
            ceprintln!("<r,bold>error</>: failed to build from manifest for source file {}: {}", file.display(), err);
            has_errors = true;
            continue;
        }
    }

    if has_errors {
        return Err(anyhow!("failed to build one or more shader files").into());
    }

    Ok(())
}

fn get_file_mtime(path: &Path) -> anyhow::Result<(PathBuf, u64)> {
    let canonical_path = path.canonicalize()?;
    let metadata = fs::metadata(path)?;
    let modified_time = metadata.modified()?;
    let mtime = match modified_time.duration_since(SystemTime::UNIX_EPOCH) {
        Ok(duration) => duration.as_secs(),
        Err(_) => {
            warn!("invalid mtime for {} (before UNIX_EPOCH)", canonical_path.display());
            0
        }
    };
    Ok((canonical_path, mtime))
}

//--------------------------------------------------------------------------------------------------

pub fn compile<P: AsRef<Path>>(file: P, options: &BuildOptions) -> Result<Module, Error> {
    let file = file.as_ref();
    let source = fs::read_to_string(&file)?;
    let manifest = load_manifest_for_source(&file, &source)?;
    let compiler_session = create_slang_session(&options.include_paths, &manifest, options)?;
    compile_slang_module(&compiler_session, file, &source, &manifest, options)
}
