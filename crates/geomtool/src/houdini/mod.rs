//! Houdini geometry (.geo/.bgeo) file parser.

mod error;
mod parser;

pub use error::Error;
use fixedbitset::FixedBitSet;
use math::Vec3;
use smol_str::SmolStr;
use std::collections::{BTreeMap, HashMap};
use std::ops::{Index, Range};
use std::path::Path;
use std::{fmt, fs, slice};
use color_print::cprintln;
////////////////////////////////////////////////////////////////////////////////////////////////////

#[derive(Copy, Clone, Eq, PartialEq)]
pub enum StorageKind {
    FpReal32,
    FpReal64,
    Int32,
    Int64,
}

impl fmt::Debug for StorageKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            StorageKind::FpReal32 => write!(f, "Float"),
            StorageKind::FpReal64 => write!(f, "Float64"),
            StorageKind::Int32 => write!(f, "Int"),
            StorageKind::Int64 => write!(f, "Int64"),
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
        }
    }
}

#[derive(Clone, Debug)]
pub enum AttributeStorage {
    FpReal32(Vec<f32>),
    FpReal64(Vec<f64>),
    Int32(Vec<i32>),
    Int64(Vec<i64>),
}

pub trait AttributeType {
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
impl_attribute_type!(Vec3, StorageKind::FpReal32, 3);

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
        }
    }

    fn storage_type(&self) -> StorageKind {
        match &self.storage {
            AttributeStorage::FpReal32(_) => StorageKind::FpReal32,
            AttributeStorage::FpReal64(_) => StorageKind::FpReal64,
            AttributeStorage::Int32(_) => StorageKind::Int32,
            AttributeStorage::Int64(_) => StorageKind::Int64,
        }
    }

    pub fn try_cast<T: AttributeType>(&self) -> Option<&TypedAttribute<T>> {
        if self.storage_type() != T::STORAGE_KIND {
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
pub enum PrimRuns {
    BezierRun(BezierRun),
    PolygonCurveRun(PolygonCurveRun),
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
    pub primitive_attributes: Vec<Attribute>,
    pub prim_runs: Vec<PrimRuns>,
    pub primitive_groups: BTreeMap<SmolStr, Group>,
    pub point_groups: BTreeMap<SmolStr, Group>,
}

impl Geo {
    /// Find a point attribute by name.
    ///
    /// TODO version that returns a typed attribute?
    pub fn point_attribute(&self, name: &str) -> Option<&Attribute> {
        self.point_attributes.iter().find(|a| a.name == name)
    }

    pub fn point_attribute_typed<T: AttributeType>(&self, name: &str) -> Option<&TypedAttribute<T>> {
        self.point_attribute(name)?.try_cast::<T>()
    }

    /// Returns the contents of the position attribute (`P`).
    pub fn positions(&self) -> &[Vec3] {
        // The first attribute is always the position attribute.
        // The fact that this is a f32 attribute is ensured by the loader.
        self.point_attributes[0].cast::<Vec3>().as_slice()
    }

    /// Returns the contents of the color attribute (`Cd`).
    pub fn color(&self) -> Option<&[Vec3]> {
        let data = self.point_attribute("Cd")?.cast().as_slice();
        Some(data)
    }

    /// Returns the position of the given vertex.
    pub fn vertex_position(&self, vertex_index: i32) -> Vec3 {
        // The vertex is an index into the topology array, which gives us the index into the point attribute.
        // The double indirection is because different vertices can share the same point.
        let point = self.topology[vertex_index as usize] as usize;
        self.positions()[point]
    }

    /// Returns the color of the given vertex.
    pub fn vertex_color(&self, vertex_index: i32) -> Option<Vec3> {
        let point = self.topology[vertex_index as usize] as usize;
        Some(self.color()?[point])
    }

    /// Iterates over all polyline curves in the geometry.
    pub fn iter_polylines(&self) -> impl Iterator<Item = PolygonCurveRef> {
        self.prim_runs
            .iter()
            .filter_map(|run| {
                if let PrimRuns::PolygonCurveRun(polygon_run) = run {
                    Some(polygon_run.iter(self))
                } else {
                    None
                }
            })
            .flatten()
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

    /// Returns whether the group contains the given element index.
    ///
    /// # Arguments
    ///
    /// * `index` - The index of the element to check. Depending on the type of group,
    ///   this can be either a primitive index or a point index.
    pub fn contains(&self, index: u32) -> bool {
        match self {
            Group::Bitset(bitset) => bitset[index as usize],
            Group::RunLengthEncoded(rle) => rle
                .iter()
                .any(|r| r.contains(&index)),
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

/// A run of Bézier curves.
#[derive(Clone, Debug)]
pub struct BezierRun {
    /// Number of curves in the run.
    pub count: usize,
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

/// A run of polygon curves.
#[derive(Clone, Debug, Default)]
pub struct PolygonCurveRun {
    /// Number of curves in the run.
    pub count: usize,
    /// Array of vertex counts for each curve.
    pub vertex_counts: Var<i32>,
    /// Index of the first vertex of the curve run.
    pub start_vertex: u32,
}

/// Polygon data.
#[derive(Clone, Debug, Default)]
pub struct PolygonRun {
    /// Number of polygons in the run.
    pub count: usize,
    /// Array of vertex counts for each polygon.
    pub vertex_counts: Var<i32>,
    /// Index of the first vertex of the polygon run.
    pub start_vertex: u32,
}

pub struct PolygonCurveRef<'a> {
    //pub run: &'a PolygonCurveRun,
    pub geo: &'a Geo,
    pub start_vertex: usize,
    pub vertex_count: usize,
}

impl<'a> PolygonCurveRef<'a> {
    /// Returns the position of the i-th vertex of the polygon curve.
    pub fn vertex_position(&self, index: usize) -> Vec3 {
        let vertex_index = self.start_vertex + index;
        self.geo.vertex_position(vertex_index as i32)
    }
}

impl PolygonCurveRun {
    /// Returns an iterator over the polygon curves in the run.
    fn iter<'a>(&'a self, geo: &'a Geo) -> impl Iterator<Item = PolygonCurveRef<'a>> + 'a {
        struct Iter<'a> {
            geo: &'a Geo,
            run: &'a PolygonCurveRun,
            index: usize,
            vertex: usize,
        }

        impl<'a> Iterator for Iter<'a> {
            type Item = PolygonCurveRef<'a>;

            fn next(&mut self) -> Option<Self::Item> {
                if self.index >= self.run.count {
                    return None;
                }

                let vertex_count = match &self.run.vertex_counts {
                    Var::Uniform(count) => *count,
                    Var::Varying(counts) => counts[self.index],
                };

                let start_vertex = self.vertex;
                self.vertex += vertex_count as usize;
                self.index += 1;

                Some(PolygonCurveRef {
                    geo: self.geo,
                    start_vertex,
                    vertex_count: vertex_count as usize,
                })
            }
        }

        Iter {
            geo,
            run: self,
            index: 0,
            vertex: self.start_vertex as usize,
        }
    }
}

pub struct BezierRunIter<'a> {
    run: &'a BezierRun,
    index: usize,
}

impl<'a> Iterator for BezierRunIter<'a> {
    type Item = BezierRef<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.run.count {
            return None;
        }

        let vertices = match &self.run.vertices {
            Var::Uniform(vertices) => vertices.as_slice(),
            Var::Varying(vertices) => &vertices[self.index],
        };

        let closed = match &self.run.closed {
            Var::Uniform(closed) => *closed,
            Var::Varying(closed) => closed[self.index],
        };

        let basis = match &self.run.basis {
            Var::Uniform(basis) => basis,
            Var::Varying(basis) => &basis[self.index],
        };

        self.index += 1;

        Some(BezierRef {
            vertices,
            closed,
            basis,
        })
    }
}

impl BezierRun {
    pub fn iter(&self) -> BezierRunIter {
        BezierRunIter { run: self, index: 0 }
    }
}

/// Represents a bezier curve.
pub struct BezierRef<'a> {
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

impl Default for BezierRun {
    fn default() -> Self {
        BezierRun {
            count: 0,
            vertices: Var::Varying(vec![]),
            closed: Var::Varying(vec![]),
            basis: Var::Varying(vec![]),
        }
    }
}


/// Prints a color-formatted summary of the geometry to stdout.
pub fn print_geometry_summary(geo: &Geo) {
    cprintln!("<bold>Geometry Summary</>:");
    cprintln!("  Points:     <b>{}</>", geo.point_count);
    cprintln!("  Primitives: <y>{}</>", geo.primitive_count);
    cprintln!("  Vertices:   <m>{}</>", geo.vertex_count);
    cprintln!("  Point Attributes:");
    for attr in &geo.point_attributes {
        cprintln!("    <bold>{:10}</>   <i>{} × {:?}</>", attr.name, attr.size, attr.storage_type());
    }
    cprintln!("  Primitive Attributes:");
    for attr in &geo.primitive_attributes {
        cprintln!("    <bold>{:10}</>   <i>{} × {:?}</>", attr.name, attr.size, attr.storage_type());
    }
}

//////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod test {
    use super::{print_geometry_summary, Geo};

    #[test]
    fn json_geo() {
        let geo = Geo::load("tests/miku239.geo").unwrap();
        eprintln!("{:#?}", geo);
    }

    #[test]
    fn binary_geo() {
        let geo = Geo::load("../../design/coat.bgeo").unwrap();
        print_geometry_summary(&geo);

        //for pl in geo.iter_polylines() {
        //    eprint!("[{}] ", pl.vertex_count);
        //    for i in 0..pl.vertex_count {
        //        let pos = pl.vertex_position(i);
        //        eprint!("({:.6}, {:.6}, {:.6}) ", pos.x, pos.y, pos.z);
        //    }
        //    eprintln!();
        //}


    }
}
