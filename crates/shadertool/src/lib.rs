#![feature(default_field_values)]
mod build;
mod manifest;
mod reflection;

use std::fs;
use anyhow::Context;
pub use manifest::*;
use std::path::{Path, PathBuf};
use std::time::SystemTime;
use log::warn;
use thiserror::Error;

#[derive(Debug, Clone)]
pub struct BuildOptions {
    /// Don't print logs to stdout.
    pub quiet: bool,
    /// Emit cargo dependency information.
    pub emit_cargo_deps: bool,
    /// Emit shader debug information.
    pub emit_debug_information: bool,
    /// Dumps SPIR-V binaries to disk alongside the archive.
    pub emit_spirv_binaries: bool,
    /// Verbosity level
    pub verbosity: u8 = 0,
}

#[derive(Error, Debug)]
#[error(transparent)]
pub struct Error(#[from] anyhow::Error);

impl Error {
    pub fn print_cargo_error(&self) {
        let fmt = format!("{:#}", self.0);
        for line in fmt.lines() {
            println!("cargo::error={line}");
        }
    }
}

/// Build all pipelines defined in the manifest at the given path.
///
/// # Arguments
/// * `manifest_path` - Path to the pipeline manifest file (JSON).
pub fn build_pipeline(manifest_path: impl AsRef<Path>, options: &BuildOptions) -> Result<(), Error> {
    fn build_pipeline_inner(manifest_path: &Path, options: &BuildOptions) -> anyhow::Result<()> {
        let manifest = match BuildManifest::load(manifest_path) {
            Ok(manifest) => manifest,
            Err(err) => {
                return Err(err).with_context(|| format!("failed to load manifest from {}", manifest_path.display()));
            }
        };

        eprintln!("loaded manifest from {}", manifest_path.display());

        manifest.build(options)?;
        //.with_context(|| format!("building from manifest {}", manifest_path.display()))?;

        Ok(())
    }

    build_pipeline_inner(manifest_path.as_ref(), options).map_err(Error)
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
