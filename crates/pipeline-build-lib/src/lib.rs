mod build;
mod manifest;

use anyhow::Context;
use color_print::cprintln;
pub use manifest::*;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct BuildOptions {
    /// Don't print logs to stdout.
    pub quiet: bool,
    /// Emit cargo dependency information.
    pub emit_cargo_deps: bool,
}

/// Build all pipelines defined in the manifest at the given path.
///
/// # Arguments
/// * `manifest_path` - Path to the pipeline manifest file (JSON).
pub fn build_pipeline(manifest_path: impl AsRef<Path>, options: &BuildOptions) -> anyhow::Result<()> {
    fn build_pipeline_inner(manifest_path: &Path, options: &BuildOptions) -> anyhow::Result<()> {

        let manifest = match BuildManifest::load(manifest_path) {
            Ok(manifest) => manifest,
            Err(err) => {
                if !options.quiet {
                    cprintln!(
                        "<r,bold>error:</> failed to load manifest from {}: {:#}",
                        manifest_path.display(),
                        err
                    );
                }
                return Err(err).with_context(|| format!("failed to load manifest from {}", manifest_path.display()));
            }
        };

        // emit cargo dependency
        if options.emit_cargo_deps {
            manifest.print_cargo_dependencies();
        }

        manifest
            .build_all(options)
            .with_context(|| format!("Failed to build pipelines defined in {}", manifest_path.display()))?;

        Ok(())
    }

    build_pipeline_inner(manifest_path.as_ref(), options)
}
