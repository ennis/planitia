use std::{fs, io};
use std::time::SystemTime;
use log::{debug, warn};
use sharc::ShaderArchive;
use crate::asset::{AssetCache, DefaultLoader, Handle, VfsPath};

pub mod pipeline_cache;
mod render_world;
mod reflection;
mod util;
//mod technique;

//pub use technique::{Technique, TechniqueInstance};

pub use pipeline_cache::GraphicsPipeline;
//pub use pipeline_cache::ComputePipeline;
//pub use util::allocate_render_target_for_pipeline;
pub use util::RenderTarget;

//-----------------------------------------------------------------

fn unix_mtime(last_modified: SystemTime) -> u64 {
    if last_modified > SystemTime::now() {
        warn!("last modification time is in the future: {:?}", last_modified);
    }

    match last_modified.duration_since(SystemTime::UNIX_EPOCH) {
        Ok(duration) => duration.as_secs(),
        Err(_) => {
            warn!("invalid modification time (before UNIX_EPOCH)");
            0
        }
    }
}

/// Loads a pipeline archive file or retrieves it from the asset cache.
fn load_shader_archive(path: impl AsRef<VfsPath>) -> Handle<ShaderArchive> {
    let cache = AssetCache::instance();
    cache.load(path.as_ref(), |path, metadata, provider, deps| {
        debug!("loading pipeline archive: {}", path.as_str());
        let data = provider.load(path)?;
        let a = ShaderArchive::from_bytes(&*data).unwrap();

        // hot reloading support
        #[cfg(feature = "hot_reload")]
        {
            fn should_rebuild_archive(archive: &ShaderArchive) -> bool {
                fn inner(archive: &ShaderArchive) -> io::Result<bool> {
                    let manifest_path = &archive[archive.manifest_file().path];
                    let manifest_mtime = unix_mtime(fs::metadata(manifest_path)?.modified()?);

                    if manifest_mtime > archive.manifest_file().mtime {
                        debug!(
                            "shader manifest modified: {} (last:{:?}, archive:{:?})",
                            manifest_path,
                            manifest_mtime,
                            archive.manifest_file().mtime
                        );
                        return Ok(true);
                    }

                    for source in archive.shader_sources() {
                        let path = &archive[source.path];
                        let source_metadata = fs::metadata(path)?;
                        if unix_mtime(source_metadata.modified()?) > source.mtime {
                            debug!(
                                "shader archive dependency modified: {} (last:{:?}, archive:{:?})",
                                path,
                                source_metadata.modified()?,
                                source.mtime
                            );
                            return Ok(true);
                        }
                    }
                    Ok(false)
                }

                inner(archive).unwrap_or(false)
            }

            // we should hot-reload when a shader source file changes, so add a dependency on them
            for source in a.shader_sources() {
                deps.add_local_file(&a[source.path]);
            }
            // also hot-reload if manifest used to build the archive has changed
            deps.add_local_file(&a[a.manifest_file().path]);

            if should_rebuild_archive(&a) {
                // Either the manifest or shader sources have changed, rebuild the archive.
                //
                // At this point we've already loaded the archive, but rebuilding the archive file
                // will immediately trigger another reload.
                // This is a bit wasteful as we're effectively reloading the archive twice:
                // first when the manifest or shaders change, so that we can run shadertool again,
                // and a second time after shadertool has rebuilt the archive.
                //
                // This could be simplified if we had a way to know which dependencies have triggered
                // the reload: with that we could trigger an archive rebuild when the manifest
                // or shader sources change, without needing to load the archive first
                // and check the modification times ourselves.
                shadertool::build_pipeline(
                    &a[a.manifest_file().path],
                    &shadertool::BuildOptions {
                        quiet: false,
                        emit_cargo_deps: false,
                        emit_debug_information: true, // TODO
                        emit_spirv_binaries: true,
                        ..
                    },
                )?;
            }
        }

        Ok(a)
    })
}

