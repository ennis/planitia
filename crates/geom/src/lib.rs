//! Representation of meshes and other geometry.
#![feature(default_field_values)]

pub mod mesh;
pub mod coat;

use std::borrow::Cow;
use std::io;
use std::ops::Deref;
use std::path::Path;
use utils::archive::{ArchiveData, ArchiveReader, ArchiveReaderOwned, Offset, OffsetUntyped};

pub use mesh::*;
pub use coat::*;

/// Header for geometry archive files.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct GeoArchiveHeader {
    /// Magic number identifying the file as a geometry archive.
    pub magic: [u8; 4],
    /// Version of the geometry archive format.
    pub version: u32,

    pub vertex_arrays: Offset<[VertexArray]> = Offset::INVALID,
    pub primitives: Offset<[Primitive]> = Offset::INVALID,
    pub indices: Offset<[u32]> = Offset::INVALID,

    //pub stroke_vertices: Offset<[StrokeVertex]> = Offset::INVALID,
    //pub mesh_vertices: Offset<[MeshVertex]> = Offset::INVALID,
    ///// Indices into `mesh_vertices`, relative to `mesh_part.base_vertex`.
    //pub mesh_indices: Offset<[u32]> = Offset::INVALID,
    //pub strokes: Offset<[Stroke]> = Offset::INVALID,
    //pub coats: Offset<[Coat]> = Offset::INVALID,
    //pub meshes: Offset<[Mesh]> = Offset::INVALID,
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

impl GeoArchiveHeader {
    /// Magic number for geometry archive files.
    pub const MAGIC: [u8; 4] = *b"GEOM";
    /// Current version of the geometry archive format.
    pub const VERSION: u32 = 2;
}

pub struct GeoArchive(Cow<'static, ArchiveReader>);

impl GeoArchive {
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
        let header: &GeoArchiveHeader = self.0.header().unwrap();
        if header.magic != GeoArchiveHeader::MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid archive magic",
            ));
        }
        if header.version != GeoArchiveHeader::VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Unsupported archive version",
            ));
        }
        Ok(())
    }

    pub fn header(&self) -> &GeoArchiveHeader {
        let header: &GeoArchiveHeader = self.0.header().unwrap();
        header
    }


    /// Returns a slice of all stroke primitives in the archive.
    // TODO this API and all below are dumb
    pub fn strokes(&self) -> &[Stroke] {
        let header = self.header();
        let primitives = &self[header.primitives];
        for prim in primitives {
            if let Primitive::Stroke(offset) = prim {
                return &self[*offset];
            }
        }
        &[]
    }

    pub fn stroke_vertices(&self) -> &[StrokeVertex] {
        let header = self.header();
        let vertex_arrays = &self[header.vertex_arrays];
        for va in vertex_arrays {
            if let VertexArray::Stroke(offset) = va {
                return &self[*offset];
            }
        }
        &[]
    }

    pub fn swept_strokes(&self) -> &[SweptStroke] {
        let header = self.header();
        let primitives = &self[header.primitives];
        for prim in primitives {
            if let Primitive::SweptStroke(offset) = prim {
                return &self[*offset];
            }
        }
        &[]
    }

    pub fn swept_stroke_vertices(&self) -> &[SweptStrokeVertex] {
        let header = self.header();
        let vertex_arrays = &self[header.vertex_arrays];
        for va in vertex_arrays {
            if let VertexArray::SweptStroke(offset) = va {
                return &self[*offset];
            }
        }
        &[]
    }

    pub fn coats(&self) -> &[Coat] {
        let header = self.header();
        let primitives = &self[header.primitives];
        for prim in primitives {
            if let Primitive::Coat(offset) = prim {
                return &self[*offset];
            }
        }
        &[]
    }

    pub fn mesh_vertices(&self) -> &[MeshVertex] {
        let header = self.header();
        let vertex_arrays = &self[header.vertex_arrays];
        for va in vertex_arrays {
            if let VertexArray::Mesh(offset) = va {
                return &self[*offset];
            }
        }
        &[]
    }

    pub fn indices(&self) -> &[u32] {
        let header = self.header();
        &self[header.indices]
    }
}

// deref to ArchiveReader
impl Deref for GeoArchive {
    type Target = ArchiveReader;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}


