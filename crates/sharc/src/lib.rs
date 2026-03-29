//! Shader archive files.
//!
//! Shader files hold a collection of pipelines (which are variants of the same pipeline).
//! It contains fixed function state, SPIR-V shader modules, tags for pipeline variants,
//! specialization constants, useful reflection data (e.g. push constant sizes),
//! and possibly cached pipeline binary blobs.
//!
//! Information in a pipeline file is sufficient to create a complete pipeline object.
//!
//! Pipeline files can be directly mapped in memory and read without any copy or parsing step.
//!
//! TODO: add documentation for resource entries

#![feature(default_field_values)]

pub mod reflection;

use gpu::vk;
use gpu::vk::{CullModeFlags, PolygonMode};
use std::borrow::Cow;
use std::ops::Deref;
use std::path::Path;
use std::time::SystemTime;
use std::{fs, io};
use utils::archive::{ArchiveError, ArchiveReader, ArchiveReaderOwned, ArchiveRoot, Offset};
use utils::zstring::ZString;

// reexport gpu types
pub use gpu;
use log::{debug, warn};
pub use utils::{archive, zstring};

/// Root struct of the shader archive file.
#[repr(C)]
#[derive(Copy, Clone)]
// NoPadding
pub struct ShaderArchiveRoot {
    /// The manifest that was used to generate this archive.
    pub manifest: FileDependency,
    /// All passes (graphics or compute) in this archive.
    pub passes: Offset<[Pass]>,
    pub images: Offset<[ImageResourceDesc]>,
}


/// Describes a sequence of passes to run in order.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct Technique {
    /// Name of the technique.
    pub name: Offset<str>,
    /// The sequence of passes to run, in order.
    pub passes: Offset<[Offset<Pass>]>,
    // TODO: table of resources that need to be allocated
    // images
    // buffers
    // TODO: table of variables to set
}

impl ArchiveRoot for ShaderArchiveRoot {
    const SIGNATURE: [u8; 4] = *b"PARC";
    const VERSION: u32 = 3;
}

/// How the size of an image resource is determined.
#[repr(C)]
#[derive(Copy, Clone)]
pub enum ImageResourceSize {
    /// Size is determined at runtime.
    Dynamic,
    /// The image is a screen-sized render target.
    RenderTarget,
    /// Fixed size.
    Fixed { width: u32, height: u32 },
}

/// Describes an image resource.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct ImageResourceDesc {
    pub name: ZString<32>,
    pub format: vk::Format,
    pub usage: gpu::ImageUsage,
    pub size: ImageResourceSize,
}

/// Describes a file that was used during compilation of the pipeline archive.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct FileDependency {
    /// Path to the file.
    pub path: Offset<str> = Offset::INVALID,
    /// Modification time as UNIX timestamp (seconds since epoch).
    pub mtime: u64 = 0,
}

impl Default for FileDependency {
    fn default() -> Self {
        Self { .. }
    }
}

/*
impl FileDependency {
    pub fn modification_time(&self) -> SystemTime {
        SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(self.mtime)
    }
}*/

/// Layout of the root parameter struct.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RootParamLayout {
    /// Size of the root parameter struct in bytes.
    pub byte_size: u32,
    pub parameters: Offset<[RootParamInfo]>,
}

/// Information about a root parameter.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RootParamInfo {
    /// Name of the root parameter in the struct (for debugging purposes).
    pub name: ZString<32>,
    /// The ID of the render-world value or resource that this root parameter should be bound to.
    pub render_world_binding: ZString<32>,
    /// Offset of the parameter in the root parameter struct.
    pub offset: u32,
    /// Size of the parameter in bytes.
    pub size: u32,
    /// Type of the root parameter.
    ///
    /// For array types, this is the type of array elements.
    /// For structs (i.e. non-scalar, non-vector, non-array types), this is UNDEFINED.
    pub format: vk::Format,
}

/// Describes a shader pipeline (graphics or compute).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Pass {
    /// Name of the pipeline.
    pub name: ZString<64>,
    /// The kind of pipeline (graphics or compute) and its associated data.
    pub kind: PassKind,
    pub root_params: RootParamLayout,
    /// List of source files.
    pub sources: Offset<[FileDependency]>,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub enum PassKind {
    Graphics(GraphicsPass),
    Compute(ComputePass),
}

impl PassKind {
    pub fn as_graphics(&self) -> Option<&GraphicsPass> {
        match self {
            PassKind::Graphics(data) => Some(data),
            _ => None,
        }
    }

    pub fn as_compute(&self) -> Option<&ComputePass> {
        match self {
            PassKind::Compute(data) => Some(data),
            _ => None,
        }
    }
}

/// Represents a compiled shader entry point.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Shader {
    pub stage: vk::ShaderStageFlags,
    pub entry_point: ZString<64>,
    pub spirv: Offset<[u32]>,
}

/// Describes a color attachment of a graphics pipeline.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ColorAttachment {
    pub resource_name: ZString<32>,
    pub clear_color: Option<[f32; 4]>,
}

/// Describes a depth/stencil attachment of a graphics pipeline.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct DepthStencilAttachment {
    pub resource_name: ZString<32>,
    pub clear_depth: Option<f32>,
    pub clear_stencil: Option<u32>,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct GraphicsPass {
    /// Size of push constant block in bytes.
    pub push_constants_size: u16,
    /// Rasterization data.
    pub rasterization: RasterizationState,
    /// Depth/stencil data.
    pub depth_stencil: DepthStencilState,
    /// Color targets
    pub color_targets: Offset<[Offset<ColorTarget>]>,
    pub color_attachments: Offset<[ColorAttachment]>,
    pub depth_stencil_attachment: Option<DepthStencilAttachment>,
    pub shaders: Offset<[Shader]>,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ComputePass {
    /// Size of push constant block in bytes.
    pub push_constants_size: u16,
    pub compute_shader: Shader,
    pub workgroup_size: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct RasterizationState {
    pub polygon_mode: PolygonMode,
    pub cull_mode: CullModeFlags,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct DepthStencilState {
    pub enable: bool,
    pub format: vk::Format,
    pub depth_compare_op: vk::CompareOp,
    pub depth_write_enable: bool,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct ColorBlendEquation {
    pub src_color_blend_factor: vk::BlendFactor,
    pub dst_color_blend_factor: vk::BlendFactor,
    pub color_blend_op: vk::BlendOp,
    pub src_alpha_blend_factor: vk::BlendFactor,
    pub dst_alpha_blend_factor: vk::BlendFactor,
    pub alpha_blend_op: vk::BlendOp,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct ColorTarget {
    pub format: vk::Format,
    pub blend: Option<ColorBlendEquation>,
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Pipeline archive file.
pub struct ShaderArchive(Cow<'static, ArchiveReader<ShaderArchiveRoot>>);

impl ShaderArchive {
    /// Loads a shader archive from a file.
    pub fn load(file_path: impl AsRef<Path>) -> Result<Self, ArchiveError> {
        let archive = ArchiveReaderOwned::load(file_path)?;
        let this = Self(Cow::Owned(archive));
        Ok(this)
    }

    /// Loads a shader archive from a byte slice.
    pub fn from_bytes(data: &[u8]) -> Result<Self, ArchiveError> {
        let archive = ArchiveReaderOwned::from_bytes(data)?;
        let this = Self(Cow::Owned(archive));
        Ok(this)
    }

    /// Loads a shader archive from a static byte slice.
    pub fn from_bytes_static(data: &'static [u8]) -> Result<Self, ArchiveError> {
        let archive = ArchiveReader::new(data)?;
        Ok(Self(Cow::Borrowed(archive)))
    }

    /// Returns an iterator over all file dependencies of the archive.
    pub fn dependencies(&self) -> impl Iterator<Item = &FileDependency> {
        let root = self.root();
        let source_files = {
            let entries = &self[root.passes];
            entries.iter().flat_map(move |entry| {
                let sources = &self[entry.sources];
                sources.iter()
            })
        };
        std::iter::once(&root.manifest).chain(source_files)
    }

    /// Checks whether any dependency of the archive has changed compared to the recorded modification times.
    pub fn dependencies_changed(&self) -> io::Result<bool> {
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

        let root = self.root();
        let manifest_path = &self[root.manifest.path];
        let manifest_mtime = unix_mtime(fs::metadata(manifest_path)?.modified()?);

        if manifest_mtime > root.manifest.mtime {
            debug!(
                "shader manifest modified: {} (last:{:?}, archive:{:?})",
                manifest_path, manifest_mtime, root.manifest.mtime
            );
            return Ok(true);
        }

        let source_files = {
            let entries = &self[root.passes];
            entries.iter().flat_map(move |entry| {
                let sources = &self[entry.sources];
                sources.iter()
            })
        };

        for source in source_files {
            let path = &self[source.path];
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

    fn data(&self) -> &ShaderArchiveRoot {
        self.0.root()
    }

    /// Finds a graphics pipeline by name.
    pub fn find_graphics_pipeline(&self, name: &str) -> Option<&GraphicsPass> {
        let data = self.data();
        let entries = &self.0[data.passes];
        for entry in entries {
            if entry.name.as_str() == name {
                match &entry.kind {
                    PassKind::Graphics(p) => return Some(p),
                    _ => continue,
                }
            }
        }
        None
    }

    /// Finds a compute pipeline by name.
    pub fn find_compute_pipeline(&self, name: &str) -> Option<&ComputePass> {
        let data = self.data();
        let entries = &self.0[data.passes];
        for entry in entries {
            if entry.name.as_str() == name {
                match &entry.kind {
                    PassKind::Compute(p) => return Some(p),
                    _ => continue,
                }
            }
        }
        None
    }

    /// Returns an iterator over all shader source files referenced by the pipelines in this archive.
    pub fn shader_sources<'a>(&'a self) -> impl Iterator<Item = &'a FileDependency> + 'a {
        let data = self.data();
        let entries = &self.0[data.passes];
        entries.iter().flat_map(move |entry| {
            let sources = &self.0[entry.sources];
            sources.iter()
        })
    }

    /// Returns the manifest path used to generate this pipeline archive.
    pub fn manifest_file(&self) -> &FileDependency {
        let data = self.data();
        &data.manifest
    }
}

// deref to ArchiveReader
impl Deref for ShaderArchive {
    type Target = ArchiveReader<ShaderArchiveRoot>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use utils::archive::ArchiveWriter;
    use utils::zstring::ZString64;

    #[test]
    fn generate_archive() {
        let mut writer = ArchiveWriter::new();
        // start with the header since it contains the signature

        let color_targets = {
            let color_targets = &[writer.write(&ColorTarget {
                format: vk::Format::R8G8B8A8_UNORM,
                blend: Some(ColorBlendEquation {
                    src_color_blend_factor: vk::BlendFactor::SRC_ALPHA,
                    dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
                    color_blend_op: vk::BlendOp::ADD,
                    src_alpha_blend_factor: vk::BlendFactor::ONE,
                    dst_alpha_blend_factor: vk::BlendFactor::ZERO,
                    alpha_blend_op: vk::BlendOp::ADD,
                }),
            })];
            writer.write_slice(color_targets)
        };
        let entries = writer.write_iter(
            1,
            [Pass {
                name: ZString64::new("example_pipeline"),
                kind: PassKind::Graphics(GraphicsPass {
                    push_constants_size: 128,
                    // just some example state
                    rasterization: RasterizationState {
                        polygon_mode: PolygonMode::FILL,
                        cull_mode: CullModeFlags::BACK,
                    },
                    depth_stencil: DepthStencilState {
                        format: vk::Format::D32_SFLOAT,
                        depth_compare_op: vk::CompareOp::ALWAYS,
                        enable: true,
                        depth_write_enable: true,
                    },
                    color_targets,
                    color_attachments: Offset::INVALID,
                    depth_stencil_attachment: None,
                    shaders: Offset::INVALID,
                }),
                root_params: RootParamLayout {
                    byte_size: 0,
                    parameters: Offset::INVALID,
                },
                sources: Offset::INVALID,
            }],
        );
        writer.write_root(&ShaderArchiveRoot {
            manifest: FileDependency {
                path: Offset::INVALID,
                mtime: 0,
            },
            passes: entries,
            images: Offset::INVALID,
        });

        // dump to disk
        fs::write("src/pipeline_archive/example_archive.parc", writer.as_slice()).unwrap();
    }
}
