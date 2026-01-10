//! Representation of meshes and other geometry.
#![feature(default_field_values)]

pub mod coat;
pub mod mesh;

use std::borrow::Cow;
use std::ops::Deref;
use std::path::Path;
use utils::archive::{ArchiveError, ArchiveReader, ArchiveReaderOwned, ArchiveRoot, Offset};

pub use coat::*;
pub use mesh::*;

/// Header for geometry archive files.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct GeoArchiveHeader {
    pub vertex_arrays: Offset<[VertexArray]> = Offset::INVALID,
    pub primitives: Offset<[Primitive]> = Offset::INVALID,
    pub indices: Offset<[u32]> = Offset::INVALID,
}

impl ArchiveRoot for GeoArchiveHeader {
    const SIGNATURE: [u8; 4] = *b"GEOM";
    const VERSION: u32 = 2;
}

#[repr(C)]
#[derive(Copy, Clone)]
#[non_exhaustive]
pub enum VertexArray {
    /// Vertices of paint strokes (polylines).
    Stroke(Offset<[StrokeVertex]>),
    SweptStroke(Offset<[SweptStrokeVertex]>),
    /// Generic 3D surface mesh vertex.
    Mesh(Offset<[MeshVertex]>),
    /// 2D position+normal vertex
    PosNorm2D(Offset<[PosNorm2DVertex]>),
}

#[repr(C)]
#[derive(Copy, Clone)]
#[non_exhaustive]
pub enum Primitive {
    Stroke(Offset<[Stroke]>),
    SweptStroke(Offset<[SweptStroke]>),
    Mesh(Offset<[Mesh]>),
    Coat(Offset<[Coat]>),
}

pub struct GeoArchive(Cow<'static, ArchiveReader<GeoArchiveHeader>>);

impl GeoArchive {
    pub fn load(file_path: impl AsRef<Path>) -> Result<Self, ArchiveError> {
        let archive = ArchiveReaderOwned::load(file_path)?;
        let this = Self(Cow::Owned(archive));
        Ok(this)
    }

    pub fn from_bytes(data: &[u8]) -> Result<Self, ArchiveError> {
        let archive = ArchiveReaderOwned::from_bytes(data)?;
        let this = Self(Cow::Owned(archive));
        Ok(this)
    }

    pub fn from_bytes_static(data: &'static [u8]) -> Result<Self, ArchiveError> {
        let archive = ArchiveReader::new(data)?;
        Ok(Self(Cow::Borrowed(archive)))
    }

    /// Returns a slice of all stroke primitives in the archive.
    // TODO this API and all below are dumb
    pub fn strokes(&self) -> &[Stroke] {
        let header = self.deref().root();
        let primitives = &self[header.primitives];
        for prim in primitives {
            if let Primitive::Stroke(offset) = prim {
                return &self[*offset];
            }
        }
        &[]
    }

    pub fn stroke_vertices(&self) -> &[StrokeVertex] {
        let header = self.deref().root();
        let vertex_arrays = &self[header.vertex_arrays];
        for va in vertex_arrays {
            if let VertexArray::Stroke(offset) = va {
                return &self[*offset];
            }
        }
        &[]
    }

    pub fn swept_strokes(&self) -> &[SweptStroke] {
        let header = self.deref().root();
        let primitives = &self[header.primitives];
        for prim in primitives {
            if let Primitive::SweptStroke(offset) = prim {
                return &self[*offset];
            }
        }
        &[]
    }

    pub fn swept_stroke_vertices(&self) -> &[SweptStrokeVertex] {
        let header = self.deref().root();
        let vertex_arrays = &self[header.vertex_arrays];
        for va in vertex_arrays {
            if let VertexArray::SweptStroke(offset) = va {
                return &self[*offset];
            }
        }
        &[]
    }

    pub fn coats(&self) -> &[Coat] {
        let header = self.deref().root();
        let primitives = &self[header.primitives];
        for prim in primitives {
            if let Primitive::Coat(offset) = prim {
                return &self[*offset];
            }
        }
        &[]
    }

    pub fn mesh_vertices(&self) -> &[MeshVertex] {
        let header = self.deref().root();
        let vertex_arrays = &self[header.vertex_arrays];
        for va in vertex_arrays {
            if let VertexArray::Mesh(offset) = va {
                return &self[*offset];
            }
        }
        &[]
    }

    pub fn indices(&self) -> &[u32] {
        let header = self.deref().root();
        &self[header.indices]
    }
}

// deref to ArchiveReader
impl Deref for GeoArchive {
    type Target = ArchiveReader<GeoArchiveHeader>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
