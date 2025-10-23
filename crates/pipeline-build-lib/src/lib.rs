mod build;
mod manifest;

use anyhow::Context;
pub use manifest::*;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct BuildOptions {
    /// Don't print anything to stdout.
    pub quiet: bool,
}

/// Build all pipelines defined in the manifest at the given path.
///
/// # Arguments
/// * `manifest_path` - Path to the pipeline manifest file (JSON).
pub fn build_pipeline(manifest_path: impl AsRef<Path>, options: &BuildOptions) -> anyhow::Result<()> {
    fn build_pipeline_inner(manifest_path: &Path, options: &BuildOptions) -> anyhow::Result<()> {
        let manifest = BuildManifest::load(manifest_path)?;

        manifest
            .build_all(options)
            .with_context(|| format!("Failed to build pipelines defined in {}", manifest_path.display()))?;

        Ok(())
    }

    build_pipeline_inner(manifest_path.as_ref(), options)
}
