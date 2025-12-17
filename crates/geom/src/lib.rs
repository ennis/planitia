//! Representation of meshes and other geometry.
#![feature(default_field_values)]

pub mod mesh;
pub mod coat;

use std::borrow::Cow;
use std::io;
use std::ops::Deref;
use std::path::Path;
use utils::archive::{ArchiveData, ArchiveReader, ArchiveReaderOwned, Offset};

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

    pub stroke_vertices: Offset<[StrokeVertex]> = Offset::INVALID,
    pub mesh_vertices: Offset<[MeshVertex]> = Offset::INVALID,
    /// Indices into `mesh_vertices`, relative to `mesh_part.base_vertex`.
    pub mesh_indices: Offset<[u32]> = Offset::INVALID,

    pub strokes: Offset<[Stroke]> = Offset::INVALID,
    pub coats: Offset<[Coat]> = Offset::INVALID,
    pub meshes: Offset<[Mesh]> = Offset::INVALID,
}

impl GeoArchiveHeader {
    /// Magic number for geometry archive files.
    pub const MAGIC: [u8; 4] = *b"GEOM";
    /// Current version of the geometry archive format.
    pub const VERSION: u32 = 1;
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

    pub fn header(&self) -> &GeoArchiveHeader {
        let header: &GeoArchiveHeader = self.0.header().unwrap();
        header
    }
}

// deref to ArchiveReader
impl Deref for GeoArchive {
    type Target = ArchiveReader;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/*
/// Generic vertex/data format descriptor.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub enum TypeDesc {
    /// 8-bit unsigned normalized integer.
    U8Norm(u16),
    /// 8-bit signed normalized integer.
    S8Norm,
    /// 16-bit unsigned normalized integer.
    U16Norm,
    /// 16-bit signed normalized integer.
    S16Norm,
    /// 32-bit unsigned normalized integer.
    U32Norm,
    /// 32-bit signed normalized integer.
    S32Norm,
    /// 32-bit floating point.
    F32,
    /// String
    Str,
}*/