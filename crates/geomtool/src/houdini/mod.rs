//! Houdini geometry (.geo/.bgeo) file parser.

mod error;
mod parser;

pub use error::Error;
use smol_str::SmolStr;
use std::path::Path;
use std::{fs, slice};

////////////////////////////////////////////////////////////////////////////////////////////////////

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum StorageKind {
    FpReal32,
    FpReal64,
    Int32,
    Int64,
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
    pub fn as_f32_slice(&self) -> Option<&[f32]> {
        match &self.storage {
            AttributeStorage::FpReal32(data) => Some(data),
            _ => None,
        }
    }

    pub fn as_i32_slice(&self) -> Option<&[i32]> {
        match &self.storage {
            AttributeStorage::Int32(data) => Some(data),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Topology {
    pub indices: Vec<u32>,
}

#[derive(Clone, Debug)]
pub enum Primitive {
    BezierRun(BezierRun),
    PolygonCurveRun(PolygonCurveRun)
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
    pub primitives: Vec<Primitive>,
}

impl Geo {
    /// Find a point attribute by name.
    ///
    /// TODO version that returns a typed attribute?
    pub fn find_point_attribute(&self, name: &str) -> Option<&Attribute> {
        self.point_attributes.iter().find(|a| a.name == name)
    }

    /// Returns the contents of the position attribute (`P`).
    pub fn positions(&self) -> &[[f32; 3]] {
        // The first attribute is always the position attribute.
        // The fact that this is a f32 attribute is ensured by the loader.
        let data = self.point_attributes[0].as_f32_slice().unwrap();
        // TODO: replace with as_chunks once it's stable.
        // SAFETY: the length is a multiple of 3, this is ensured by the loader.
        let new_len = data.len() / 3;
        unsafe { slice::from_raw_parts(data.as_ptr().cast(), new_len) }
    }

    /// Returns the contents of the color attribute (`Cd`).
    pub fn color(&self) -> Option<&[[f32; 3]]> {
        let data = self.find_point_attribute("Cd")?.as_f32_slice()?;
        let new_len = data.len() / 3;
        Some(unsafe { slice::from_raw_parts(data.as_ptr().cast(), new_len) })
    }

    /// Returns the position of the given vertex.
    pub fn vertex_position(&self, vertex_index: i32) -> [f32; 3] {
        // The vertex is an index into the topology array, which gives us the index into the point attribute.
        // The double indirection is because different vertices can share the same point.
        let point = self.topology[vertex_index as usize] as usize;
        self.positions()[point]
    }

    /// Returns the color of the given vertex.
    pub fn vertex_color(&self, vertex_index: i32) -> Option<[f32; 3]> {
        let point = self.topology[vertex_index as usize] as usize;
        Some(self.color()?[point])
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
    /// Array of vertex count for each curve.
    pub vertex_counts: Var<i32>,
    /// Index of the first vertex of the curve run.
    pub start_vertex: u32,
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


#[cfg(test)]
mod test {
    use super::Geo;

    #[test]
    fn json_geo() {
        let geo = Geo::load("tests/miku239.geo").unwrap();
        eprintln!("{:#?}", geo);
    }

    #[test]
    fn binary_geo() {
        let geo = Geo::load("tests/strokes.bgeo").unwrap();
        eprintln!("{:#?}", geo);
    }


}
