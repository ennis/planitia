//! Pipeline archive files.
//!
//! Pipeline files hold a collection of pipelines (which are variants of the same pipeline).
//! It contains fixed function state, SPIR-V shader modules, tags for pipeline variants,
//! specialization constants, useful reflection data (e.g. push constant sizes),
//! and possibly cached pipeline binary blobs.
//!
//! Information in a pipeline file is sufficient to create a complete pipeline object.
//!
//! Pipeline files can be directly mapped in memory and read without any copy or parsing step.

#![feature(default_field_values)]

use gpu::vk;
use gpu::vk::{CullModeFlags, PolygonMode};
use std::borrow::Cow;
use std::io;
use std::ops::Deref;
use std::path::Path;
use utils::archive::{ArchiveReader, ArchiveReaderOwned, Offset};
use utils::zstring::ZString;

pub use gpu;
pub use utils::{archive, zstring};

pub const PIPELINE_ARCHIVE_MAGIC: [u8; 4] = *b"PARC";

#[repr(C)]
#[derive(Copy, Clone)]
// NoPadding
pub struct PipelineArchiveData {
    /// "PARC"
    pub magic: [u8; 4],
    /// 1
    pub version: u32,
    pub manifest_file: FileDependency,
    pub entries: Offset<[PipelineEntryData]>,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct FileDependency {
    pub path: Offset<str> = Offset::INVALID,
    /// Modification time as UNIX timestamp (seconds since epoch).
    pub mtime: u64 = 0,
}

/*
impl FileDependency {
    pub fn modification_time(&self) -> SystemTime {
        SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(self.mtime)
    }
}*/

impl Default for FileDependency {
    fn default() -> Self {
        Self { .. }
    }
}

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
    /// The ID of the render-world value that this root parameter should be bound to.
    ///
    /// It can be empty, in which case the parameter is initialized to zero.
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

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PipelineEntryData {
    /// Name of the pipeline.
    pub name: ZString<64>,
    pub kind: PipelineKind,
    pub root_params: RootParamLayout,
    /// List of source files.
    pub sources: Offset<[FileDependency]>,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub enum PipelineKind {
    Graphics(GraphicsPipelineData),
    Compute(ComputePipelineData),
}

impl PipelineKind {
    pub fn as_graphics(&self) -> Option<&GraphicsPipelineData> {
        match self {
            PipelineKind::Graphics(data) => Some(data),
            _ => None,
        }
    }

    pub fn as_compute(&self) -> Option<&ComputePipelineData> {
        match self {
            PipelineKind::Compute(data) => Some(data),
            _ => None,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ShaderData {
    pub stage: vk::ShaderStageFlags,
    pub entry_point: ZString<32>,
    pub spirv: Offset<[u32]>,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct GraphicsPipelineData {
    /// Size of push constant block in bytes.
    pub push_constants_size: u16,
    /// Rasterization data.
    pub rasterization: RasterizerStateData,
    /// Depth/stencil data.
    pub depth_stencil: DepthStencilStateData,
    /// Color targets
    pub color_targets: Offset<[Offset<ColorTarget>]>,
    pub shaders: Offset<[ShaderData]>,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ComputePipelineData {
    /// Size of push constant block in bytes.
    pub push_constants_size: u16,
    pub compute_shader: ShaderData,
    pub workgroup_size: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct RasterizerStateData {
    pub polygon_mode: PolygonMode,
    pub cull_mode: CullModeFlags,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct DepthStencilStateData {
    pub enable: bool,
    pub format: vk::Format,
    pub depth_compare_op: vk::CompareOp,
    pub depth_write_enable: bool,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct ColorBlendEquationData {
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
    pub blend: ColorBlendEquationData,
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Pipeline archive file.
pub struct PipelineArchive(Cow<'static, ArchiveReader>);

impl PipelineArchive {
    pub fn load(file_path: impl AsRef<Path>) -> io::Result<Self> {
        let archive = ArchiveReaderOwned::load(file_path)?;
        let this = Self(Cow::Owned(archive));
        this.check_header()?;
        Ok(this)
    }

    pub fn from_bytes(data: &[u8]) -> io::Result<Self> {
        let archive = ArchiveReaderOwned::from_bytes(data);
        let this = Self(Cow::Owned(archive));
        this.check_header()?;
        Ok(this)
    }

    pub fn from_bytes_static(data: &'static [u8]) -> io::Result<Self> {
        let archive = ArchiveReader::new(data);
        Ok(Self(Cow::Borrowed(archive)))
    }

    fn check_header(&self) -> io::Result<()> {
        let header: &PipelineArchiveData = self.0.header().unwrap();
        if header.magic != PIPELINE_ARCHIVE_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid pipeline archive magic",
            ));
        }
        if header.version != 1 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Unsupported pipeline archive version",
            ));
        }
        Ok(())
    }

    fn data(&self) -> &PipelineArchiveData {
        let header: &PipelineArchiveData = self.0.header().unwrap();
        header
    }

    pub fn find_graphics_pipeline(&self, name: &str) -> Option<&GraphicsPipelineData> {
        let data = self.data();
        let entries = &self.0[data.entries];
        for entry in entries {
            if entry.name.as_str() == name {
                match &entry.kind {
                    PipelineKind::Graphics(p) => return Some(p),
                    _ => continue,
                }
            }
        }
        None
    }

    pub fn find_compute_pipeline(&self, name: &str) -> Option<&ComputePipelineData> {
        let data = self.data();
        let entries = &self.0[data.entries];
        for entry in entries {
            if entry.name.as_str() == name {
                match &entry.kind {
                    PipelineKind::Compute(p) => return Some(p),
                    _ => continue,
                }
            }
        }
        None
    }

    /// Returns an iterator over all source files referenced by the pipelines in this archive.
    pub fn source_files<'a>(&'a self) -> impl Iterator<Item = &'a FileDependency> + 'a {
        let data = self.data();
        let entries = &self.0[data.entries];
        entries.iter().flat_map(move |entry| {
            let sources = &self.0[entry.sources];
            sources.iter()
        })
    }

    /// Returns the manifest path used to generate this pipeline archive.
    pub fn manifest_file(&self) -> &FileDependency {
        let data = self.data();
        &data.manifest_file
    }
}

// deref to ArchiveReader
impl Deref for PipelineArchive {
    type Target = ArchiveReader;

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
        let header = writer.write(PipelineArchiveData {
            magic: *b"PARC",
            version: 1,
            manifest_file: FileDependency {
                path: Offset::INVALID,
                mtime: 0,
            },
            entries: Offset::INVALID,
        });
        let color_targets = {
            let color_targets = &[writer.write(ColorTarget {
                format: vk::Format::R8G8B8A8_UNORM,
                blend: ColorBlendEquationData {
                    src_color_blend_factor: vk::BlendFactor::SRC_ALPHA,
                    dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
                    color_blend_op: vk::BlendOp::ADD,
                    src_alpha_blend_factor: vk::BlendFactor::ONE,
                    dst_alpha_blend_factor: vk::BlendFactor::ZERO,
                    alpha_blend_op: vk::BlendOp::ADD,
                },
            })];
            writer.write_slice(color_targets)
        };
        let entries = writer.write_iter(
            1,
            [PipelineEntryData {
                name: ZString64::new("example_pipeline"),
                kind: PipelineKind::Graphics(GraphicsPipelineData {
                    push_constants_size: 128,
                    // just some example state
                    rasterization: RasterizerStateData {
                        polygon_mode: PolygonMode::FILL,
                        cull_mode: CullModeFlags::BACK,
                    },
                    depth_stencil: DepthStencilStateData {
                        format: vk::Format::D32_SFLOAT,
                        depth_compare_op: vk::CompareOp::ALWAYS,
                        enable: true,
                        depth_write_enable: true,
                    },
                    color_targets,
                    shaders: Offset::INVALID,
                }),
                root_params: RootParamLayout {
                    byte_size: 0,
                    parameters: Offset::INVALID,
                },
                sources: Offset::INVALID,
            }],
        );
        writer[header].entries = entries;

        // dump to disk
        fs::write("src/pipeline_archive/example_archive.parc", writer.as_slice()).unwrap();
    }
}
