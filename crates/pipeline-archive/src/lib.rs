//! Pipeline archive files.

// Pipeline files hold a collection of pipelines (which are variants of the same pipeline).
// It contains fixed function state, SPIR-V shader modules, tags for pipeline variants,
// specialization constants, useful reflection data (e.g. push constant sizes),
// and possibly cached pipeline binary blobs.
//
// Information in a pipeline file is sufficient to create a complete pipeline object.
//
// Pipeline files can be directly mapped in memory and read without any copy or parsing step.

use gpu::vk;
use gpu::vk::PolygonMode;
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
    pub entries: Offset<[PipelineEntryData]>,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PipelineEntryData {
    /// Name of the pipeline.
    pub name: ZString<64>,
    pub kind: PipelineKind,
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
    pub entry_point: ZString<32>,
    pub spirv: Offset<[u32]>,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub enum GraphicsPipelineShaders {
    Primitive {
        vertex: Offset<ShaderData>,
    },
    Mesh {
        task: Offset<ShaderData>,
        mesh: Offset<ShaderData>,
    },
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
    pub vertex_or_mesh_shaders: GraphicsPipelineShaders,
    /// Compiled SPIR-V fragment shader module.
    pub fragment_shader: Offset<ShaderData>,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ComputePipelineData {
    /// Size of push constant block in bytes.
    pub push_constants_size: u16,
    pub compute_shader: Offset<ShaderData>,
    pub workgroup_size: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct RasterizerStateData {
    pub polygon_mode: PolygonMode,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct DepthStencilStateData {
    pub format: vk::Format,
    pub depth_compare_op: vk::CompareOp,
    pub depth_test_enable: bool,
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
        Ok(Self(Cow::Owned(archive)))
    }

    pub fn from_bytes(data: &[u8]) -> io::Result<Self> {
        let archive = ArchiveReaderOwned::from_bytes(data);
        Ok(Self(Cow::Owned(archive)))
    }

    pub fn from_bytes_static(data: &'static [u8]) -> io::Result<Self> {
        let archive = ArchiveReader::new(data);
        Ok(Self(Cow::Borrowed(archive)))
    }

    pub fn find_graphics_pipeline(&self, name: &str) -> Option<&GraphicsPipelineData> {
        let archive_data: &PipelineArchiveData = self.0.header().unwrap();
        let entries = &self.0[archive_data.entries];
        for entry in entries {
            if entry.name.as_str() == name {
                match &entry.kind {
                    PipelineKind::Graphics(data) => return Some(data),
                    _ => continue,
                }
            }
        }
        None
    }

    pub fn find_compute_pipeline(&self, name: &str) -> Option<&ComputePipelineData> {
        let archive_data: &PipelineArchiveData = self.0.header().unwrap();
        let entries = &self.0[archive_data.entries];
        for entry in entries {
            if entry.name.as_str() == name {
                match &entry.kind {
                    PipelineKind::Compute(data) => return Some(data),
                    _ => continue,
                }
            }
        }
        None
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
    use utils::zstring::{ZString16, ZString32, ZString64};

    #[test]
    fn generate_archive() {
        let mut writer = ArchiveWriter::new();
        // start with the header since it contains the signature
        let header = writer.write(PipelineArchiveData {
            magic: *b"PARC",
            version: 1,
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
                    },
                    depth_stencil: DepthStencilStateData {
                        format: vk::Format::D32_SFLOAT,
                        depth_compare_op: vk::CompareOp::ALWAYS,
                        depth_test_enable: true,
                        depth_write_enable: true,
                    },
                    color_targets,
                    vertex_or_mesh_shaders: GraphicsPipelineShaders::Primitive {
                        vertex: Offset::INVALID,
                    },
                    fragment_shader: Offset::INVALID,
                }),
            }],
        );
        writer[header].entries = entries;

        // dump to disk
        fs::write("src/pipeline_archive/example_archive.parc", writer.as_slice()).unwrap();
    }
}
