//! Reads Houdini JSON-based geometry files (.geo/.bgeo).
mod error;
mod parser;
pub mod util;

pub use error::Error;
use fixedbitset::FixedBitSet;
use math::{Vec2, Vec3, Vec4};
use smol_str::SmolStr;
use std::collections::BTreeMap;
use std::ops::{Index, Range};
use std::{fmt, slice};

////////////////////////////////////////////////////////////////////////////////////////////////////

#[derive(Copy, Clone, Eq, PartialEq)]
pub enum AttributeClass {
    Point,
    Vertex,
    Primitive,
    Detail,
}

#[derive(Copy, Clone, Eq, PartialEq)]
pub enum StorageKind {
    FpReal32,
    FpReal64,
    Int32,
    Int64,
    String,
}

impl fmt::Debug for StorageKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            StorageKind::FpReal32 => write!(f, "Float"),
            StorageKind::FpReal64 => write!(f, "Float64"),
            StorageKind::Int32 => write!(f, "Int"),
            StorageKind::Int64 => write!(f, "Int64"),
            StorageKind::String => write!(f, "String"),
        }
    }
}

impl StorageKind {
    pub fn byte_size(&self) -> usize {
        match self {
            StorageKind::FpReal32 => 4,
            StorageKind::FpReal64 => 8,
            StorageKind::Int32 => 4,
            StorageKind::Int64 => 8,
            StorageKind::String => 0,
        }
    }
}

// TODO: should have other storage types (RLE, constant, etc)
#[derive(Clone, Debug)]
pub enum AttributeStorage {
    FpReal32(Vec<f32>),
    FpReal64(Vec<f64>),
    Int32(Vec<i32>),
    Int64(Vec<i64>),
    Strings { values: Vec<SmolStr>, indices: Vec<i32> },
}

pub trait AttributeType: Clone + Default {
    const STORAGE_KIND: StorageKind;
    const TUPLE_SIZE: usize;
}

macro_rules! impl_attribute_type {
    ($ty:ty, $kind:expr, $size:expr) => {
        impl AttributeType for $ty {
            const STORAGE_KIND: StorageKind = $kind;
            const TUPLE_SIZE: usize = $size;
        }
    };
}

impl_attribute_type!(f32, StorageKind::FpReal32, 1);
impl_attribute_type!(f64, StorageKind::FpReal64, 1);
impl_attribute_type!(i32, StorageKind::Int32, 1);
impl_attribute_type!(i64, StorageKind::Int64, 1);
impl_attribute_type!(Vec2, StorageKind::FpReal32, 2);
impl_attribute_type!(Vec3, StorageKind::FpReal32, 3);
impl_attribute_type!(Vec4, StorageKind::FpReal32, 4);
impl_attribute_type!(SmolStr, StorageKind::String, 1);

/// Geometry attribute.
#[derive(Clone, Debug)]
pub struct Attribute {
    /// Name of the attribute.
    pub name: SmolStr,
    /// Number of elements per tuple.
    pub size: usize,
    /// Storage.
    pub storage: AttributeStorage,
}

impl Attribute {
    fn data(&self) -> (*const u8, usize) {
        match &self.storage {
            AttributeStorage::FpReal32(data) => (data.as_ptr().cast(), data.len() * 4),
            AttributeStorage::FpReal64(data) => (data.as_ptr().cast(), data.len() * 8),
            AttributeStorage::Int32(data) => (data.as_ptr().cast(), data.len() * 4),
            AttributeStorage::Int64(data) => (data.as_ptr().cast(), data.len() * 8),
            AttributeStorage::Strings { .. } => {
                panic!("Cannot get raw data pointer for string attribute")
            }
        }
    }

    pub fn storage_kind(&self) -> StorageKind {
        match &self.storage {
            AttributeStorage::FpReal32(_) => StorageKind::FpReal32,
            AttributeStorage::FpReal64(_) => StorageKind::FpReal64,
            AttributeStorage::Int32(_) => StorageKind::Int32,
            AttributeStorage::Int64(_) => StorageKind::Int64,
            AttributeStorage::Strings { .. } => StorageKind::String,
        }
    }

    pub fn try_cast<T: AttributeType>(&self) -> Option<&TypedAttribute<T>> {
        if self.storage_kind() != T::STORAGE_KIND {
            return None;
        }
        // It's OK to reinterpret a vec3 array as an array of f32s with 3x the size.
        if !(T::TUPLE_SIZE == 1 || T::TUPLE_SIZE == self.size) {
            return None;
        }
        unsafe { Some(&*(self as *const Attribute as *const TypedAttribute<T>)) }
    }

    pub fn cast<T: AttributeType>(&self) -> &TypedAttribute<T> {
        self.try_cast().unwrap()
    }
}

#[repr(transparent)]
pub struct TypedAttribute<T: AttributeType> {
    pub attribute: Attribute,
    _marker: std::marker::PhantomData<T>,
}

impl<T: AttributeType> TypedAttribute<T> {
    pub fn as_slice(&self) -> &[T] {
        unsafe {
            let (ptr, len) = self.attribute.data();
            slice::from_raw_parts(ptr as *const T, len / size_of::<T>())
        }
    }

    pub fn len(&self) -> usize {
        self.as_slice().len()
    }
}

impl<T: AttributeType> Index<usize> for TypedAttribute<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.as_slice()[index]
    }
}

#[derive(Clone, Debug)]
pub struct Topology {
    pub indices: Vec<u32>,
}

#[derive(Clone, Debug)]
pub struct PrimRun {
    pub base_index: u32,
    pub count: u32,
    pub kind: PrimRunKind,
}

#[derive(Clone, Debug)]
pub enum PrimRunKind {
    BezierRun(BezierRun),
    PolygonRun(PolygonRun),
}

/// The contents of a houdini geometry file.
#[derive(Clone, Debug, Default)]
pub struct Geo {
    pub point_count: usize,
    pub vertex_count: usize,
    pub primitive_count: usize,
    pub topology: Vec<u32>,
    pub point_attributes: Vec<Attribute>,
    pub vertex_attributes: Vec<Attribute>,
    pub primitive_attributes: Vec<Attribute>,
    /// `(prim_start_index, prim_run)`
    pub prim_runs: Vec<PrimRun>,
    pub primitive_groups: BTreeMap<SmolStr, Group>,
    pub point_groups: BTreeMap<SmolStr, Group>,
}

// TODO: houdini group syntax parser

impl Geo {
    /// Finds an attribute by name and class.
    pub fn attribute(&self, class: AttributeClass, name: &str) -> Option<&Attribute> {
        match class {
            AttributeClass::Point => self.point_attribute(name),
            AttributeClass::Vertex => self.vertex_attribute(name),
            AttributeClass::Primitive => self.primitive_attribute(name),
            _ => unimplemented!("detail attributes not supported"),
        }
    }

    /// Find a point attribute by name.
    ///
    /// TODO version that returns a typed attribute?
    pub fn point_attribute(&self, name: &str) -> Option<&Attribute> {
        self.point_attributes.iter().find(|a| a.name == name)
    }

    pub fn point_attribute_typed<T: AttributeType>(&self, name: &str) -> Option<&TypedAttribute<T>> {
        self.point_attribute(name)?.try_cast::<T>()
    }

    pub fn vertex_attribute(&self, name: &str) -> Option<&Attribute> {
        self.vertex_attributes.iter().find(|a| a.name == name)
    }

    pub fn primitive_attribute(&self, name: &str) -> Option<&Attribute> {
        self.primitive_attributes.iter().find(|a| a.name == name)
    }

    pub fn primitive_attribute_typed<T: AttributeType>(&self, name: &str) -> Option<&TypedAttribute<T>> {
        self.primitive_attribute(name)?.try_cast::<T>()
    }

    pub fn vertex_attribute_typed<T: AttributeType>(&self, name: &str) -> Option<&TypedAttribute<T>> {
        self.vertex_attribute(name)?.try_cast::<T>()
    }

    /// Returns the point index for the given vertex.
    pub fn vertexpoint(&self, vertex_index: u32) -> u32 {
        self.topology[vertex_index as usize]
    }

    /// Reads a vertex attribute value.
    pub fn vertex<T: AttributeType>(&self, vertex_index: u32, name: &str) -> T {
        if let Some(attr) = self.vertex_attribute_typed::<T>(name) {
            attr[vertex_index as usize].clone()
        } else if let Some(attr) = self.point_attribute_typed::<T>(name) {
            let point_index = self.topology[vertex_index as usize] as usize;
            attr[point_index].clone()
        } else {
            T::default()
        }
    }

    /// Reads a point attribute value.
    pub fn point<T: AttributeType>(&self, point_index: u32, name: &str) -> T {
        self.point_attribute_typed::<T>(name)
            .map(|a| a[point_index as usize].clone())
            .unwrap_or_default()
    }

    /// Reads a primitive attribute value.
    pub fn prim<T: AttributeType>(&self, prim_index: u32, name: &str) -> T {
        self.primitive_attribute_typed::<T>(name)
            .map(|a| a[prim_index as usize].clone())
            .unwrap_or_default()
    }

    /// Returns the contents of the position attribute (`P`).
    pub fn all_positions(&self) -> &[Vec3] {
        // The first attribute is always the position attribute.
        // The fact that this is a f32 attribute is ensured by the loader.
        self.point_attributes[0].cast::<Vec3>().as_slice()
    }

    /// Returns the contents of the color attribute (`Cd`).
    pub fn all_colors(&self) -> Option<&[Vec3]> {
        let data = self.point_attribute("Cd")?.cast().as_slice();
        Some(data)
    }

    /*/// Returns the position of the given vertex.
    pub fn vertex_position(&self, vertex_index: i32) -> Vec3 {
        // The vertex is an index into the topology array, which gives us the index into the point attribute.
        // The double indirection is because different vertices can share the same point.
        let point = self.topology[vertex_index as usize] as usize;
        self.all_positions()[point]
    }*/

    /*/// Returns the color of the given vertex.
    pub fn vertex_color(&self, vertex_index: i32) -> Option<Vec3> {
        let point = self.topology[vertex_index as usize] as usize;
        Some(self.all_colors()?[point])
    }*/

    /// Iterates over all Bézier curves in the geometry.
    pub fn beziers(&self) -> impl Iterator<Item = BezierRef<'_>> {
        self.prim_runs
            .iter()
            .filter_map(|run| {
                if let PrimRunKind::BezierRun(ref bezier_run) = run.kind {
                    Some(BezierRunIter {
                        run: bezier_run,
                        index: 0,
                    })
                } else {
                    None
                }
            })
            .flatten()
    }

    /// Iterates over all Bézier curves belonging to the specified primitive group.
    pub fn beziers_in_group(&self, primitive_group: &Group) -> impl Iterator<Item = BezierRef<'_>> {
        self.beziers().filter(|p| primitive_group.contains(p.primnum))
    }

    /// Iterates over all polygons in the geometry, including open polygons (polylines).
    pub fn polygons(&self) -> impl Iterator<Item = PolygonRef<'_>> {
        self.prim_runs
            .iter()
            .filter_map(|run| {
                if let PrimRunKind::PolygonRun(ref polygon_run) = run.kind {
                    Some(PolygonRunIter {
                        geo: self,
                        run: polygon_run,
                        index: 0,
                        vertex: polygon_run.start_vertex as usize,
                    })
                } else {
                    None
                }
            })
            .flatten()
    }

    /// Iterates over all polygons belonging to the specified primitive group.
    pub fn polygons_in_group(&self, primitive_group: &Group) -> impl Iterator<Item = PolygonRef<'_>> {
        self.polygons().filter(|p| primitive_group.contains(p.primitive_index))
    }

    /// Iterates over all polylines (open polygons).
    pub fn polylines(&self) -> impl Iterator<Item = PolygonRef<'_>> {
        self.prim_runs
            .iter()
            .filter_map(|run| match run.kind {
                PrimRunKind::PolygonRun(ref polygon_run) if !polygon_run.closed => Some(PolygonRunIter {
                    geo: self,
                    run: polygon_run,
                    index: 0,
                    vertex: polygon_run.start_vertex as usize,
                }),
                _ => None,
            })
            .flatten()
    }

    /// Iterates over all polylines (open polygons) belonging to the specified primitive group.
    pub fn polylines_in_group(&self, primitive_group: &Group) -> impl Iterator<Item = PolygonRef<'_>> {
        self.polylines().filter(|p| primitive_group.contains(p.primitive_index))
    }

    /// Returns an iterator over all primitive groups.
    pub fn primitive_groups(&self) -> impl Iterator<Item = (&SmolStr, &Group)> {
        self.primitive_groups.iter()
    }

    /// Returns an iterator over all point groups.
    pub fn point_groups(&self) -> impl Iterator<Item = (&SmolStr, &Group)> {
        self.point_groups.iter()
    }

    /// Returns the primitive group with the given name, or the empty group if it doesn't exist.
    pub fn primitive_group(&self, name: &str) -> &Group {
        self.primitive_groups.get(name).unwrap_or(Group::empty())
    }

    /// Returns the point group with the given name, or the empty group if it doesn't exist.
    pub fn point_group(&self, name: &str) -> &Group {
        self.point_groups.get(name).unwrap_or(Group::empty())
    }
}

#[derive(Clone, Debug)]
pub enum Group {
    Bitset(FixedBitSet),
    RunLengthEncoded(Vec<Range<u32>>),
}

impl Default for Group {
    fn default() -> Self {
        Group::Bitset(FixedBitSet::with_capacity(0))
    }
}

impl Group {
    pub fn empty() -> &'static Group {
        static EMPTY_GROUP: Group = Group::RunLengthEncoded(Vec::new());
        &EMPTY_GROUP
    }

    pub fn count(&self) -> usize {
        match self {
            Group::Bitset(bitset) => bitset.count_ones(..),
            Group::RunLengthEncoded(rle) => rle.iter().map(|r| (r.end - r.start) as usize).sum(),
        }
    }

    /// Returns whether the group contains the given element index.
    ///
    /// # Arguments
    ///
    /// * `index` - The index of the element to check. Depending on the type of group,
    ///   this can be either a primitive index or a point index.
    pub fn contains(&self, index: u32) -> bool {
        match self {
            Group::Bitset(bitset) => bitset[index as usize],
            Group::RunLengthEncoded(rle) => rle.iter().any(|r| r.contains(&index)),
        }
    }
}

/// Bezier curve basis.
#[derive(Clone, Debug, Default)]
pub struct BezierBasis {
    pub order: u32,
    pub knots: Vec<f32>,
}

/// A geometry variable that can be either uniform over a sequence of elements,
/// or varying per element.
#[derive(Clone, Debug)]
pub enum Var<T> {
    /// Same value for all elements in the sequence.
    Uniform(T),
    /// One value per element in the sequence.
    Varying(Vec<T>),
    // TODO: RLE?
}

impl<T: Default> Default for Var<T> {
    fn default() -> Self {
        Var::Uniform(T::default())
    }
}
////////////////////////////////////////////////////////////////////////////////////////////////////

/// A run of Bézier curves.
#[derive(Clone, Debug)]
pub struct BezierRun {
    pub base_prim_index: u32,
    pub count: u32,
    /// Vertices of the control points.
    ///
    /// They are indices into the `topology` vector.
    /// They are usually `Varying`, because a bezier run with the same control points isn't very useful.
    pub vertices: Var<Vec<i32>>,
    /// Whether the curve is closed.
    pub closed: Var<bool>,
    /// Curve basis information.
    pub basis: Var<BezierBasis>,
}

impl Default for BezierRun {
    fn default() -> Self {
        BezierRun {
            base_prim_index: 0,
            count: 0,
            vertices: Var::Varying(vec![]),
            closed: Var::Varying(vec![]),
            basis: Var::Varying(vec![]),
        }
    }
}

/// Represents a bezier curve.
pub struct BezierRef<'a> {
    /// Primitive index
    pub primnum: u32,
    /// Vertices of the control points.
    ///
    /// They are indices into the `topology` vector.
    /// They are i32 because that's what the loader produces, but they are always positive.
    pub vertices: &'a [i32],
    /// Whether the curve is closed.
    pub closed: bool,
    /// Curve basis information.
    pub basis: &'a BezierBasis,
}

pub struct BezierRunIter<'a> {
    run: &'a BezierRun,
    index: u32,
}

impl<'a> Iterator for BezierRunIter<'a> {
    type Item = BezierRef<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.run.count {
            return None;
        }
        let primitive_index = self.run.base_prim_index + self.index;

        let vertices = match &self.run.vertices {
            Var::Uniform(vertices) => vertices.as_slice(),
            Var::Varying(vertices) => &vertices[self.index as usize],
        };

        let closed = match &self.run.closed {
            Var::Uniform(closed) => *closed,
            Var::Varying(closed) => closed[self.index as usize],
        };

        let basis = match &self.run.basis {
            Var::Uniform(basis) => basis,
            Var::Varying(basis) => &basis[self.index as usize],
        };

        self.index += 1;

        Some(BezierRef {
            primnum: primitive_index,
            vertices,
            closed,
            basis,
        })
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Polygon run.
#[derive(Clone, Debug, Default)]
pub struct PolygonRun {
    pub base_prim_index: u32,
    pub count: u32,
    /// Array of vertex counts for each polygon.
    pub vertex_counts: Var<i32>,
    /// Index of the first vertex of the polygon run.
    pub start_vertex: u32,
    pub closed: bool,
}

pub struct PolygonRef<'a> {
    pub geo: &'a Geo,
    pub primitive_index: u32,
    pub start_vertex: usize,
    pub vertex_count: usize,
    pub closed: bool,
}

impl<'a> PolygonRef<'a> {
    /// Iterates over the vertex indices of the polygon.
    pub fn vertices(&self) -> impl Iterator<Item = u32> + '_ {
        self.start_vertex as u32..(self.start_vertex as u32 + self.vertex_count as u32)
    }
}

struct PolygonRunIter<'a, 'b> {
    geo: &'a Geo,
    run: &'b PolygonRun,
    index: u32,
    vertex: usize,
}

impl<'a, 'b> Iterator for PolygonRunIter<'a, 'b> {
    type Item = PolygonRef<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.run.count {
            return None;
        }
        let primitive_index = self.run.base_prim_index + self.index;

        let vertex_count = match &self.run.vertex_counts {
            Var::Uniform(count) => *count,
            Var::Varying(counts) => counts[self.index as usize],
        };

        let start_vertex = self.vertex;
        self.vertex += vertex_count as usize;
        self.index += 1;

        Some(PolygonRef {
            geo: self.geo,
            primitive_index,
            start_vertex,
            vertex_count: vertex_count as usize,
            closed: self.run.closed,
        })
    }
}
