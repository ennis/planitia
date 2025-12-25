//! Doubly-connected edge list implementation.

//! Half-edge polygon mesh data structure.
//!
//! Used for the calculation of surface curvature.

use math::Vec3;
use std::collections::{BTreeMap, HashMap};

pub type VertexIndex = u32;
pub type HalfEdgeIndex = u32;
pub type FaceIndex = u32;

#[derive(Clone, Copy, Debug)]
pub struct HalfEdge {
    pub origin: VertexIndex,
    pub twin: HalfEdgeIndex,
    pub next: HalfEdgeIndex,
    pub prev: HalfEdgeIndex,
    pub incident_face: Option<FaceIndex>,
}

#[derive(Clone, Copy, Debug)]
pub struct Vertex {
    pub coordinates: Vec3,
    pub incident_edge: HalfEdgeIndex,
}

#[derive(Clone, Copy, Debug)]
pub struct Face {
    pub incident_edge: HalfEdgeIndex,
}

#[derive(Clone, Debug)]
pub struct HalfEdgeMesh {
    pub vertices: Vec<Vertex>,
    pub faces: Vec<Face>,
    pub half_edges: Vec<HalfEdge>,
}

impl HalfEdgeMesh {
    /// Creates a PolyMesh from a triangle mesh specified as a position buffer and an index buffer.
    pub fn from_indexed_triangle_mesh(positions: impl IntoIterator<Item=Vec3>, indices: &[u32]) -> Self {
        assert!(
            indices.len() % 3 == 0,
            "invalid number of indices: must be a multiple of 3"
        );

        let mut vertices = Vec::new();
        let mut faces = Vec::new();
        let mut half_edges = Vec::new();

        // copy positions
        for position in positions {
            vertices.push(Vertex {
                coordinates: Vec3::new(position[0], position[1], position[2]),
                incident_edge: 0,
            });
        }

        let mut half_edge_map = HashMap::new();

        // for each triangle
        for i in 0..indices.len() / 3 {
            let face_index = i as FaceIndex;

            // vertex indices
            let v0 = indices[i * 3] as VertexIndex;
            let v1 = indices[i * 3 + 1] as VertexIndex;
            let v2 = indices[i * 3 + 2] as VertexIndex;

            let (he0, he1, he2) = {
                let mut insert_half_edge = |v0, v1| {
                    let he = half_edges.len() as HalfEdgeIndex;
                    half_edges.push(HalfEdge {
                        origin: 0,
                        twin: 0,
                        next: 0,
                        prev: 0,
                        incident_face: None,
                    });
                    half_edge_map.insert((v0, v1), he);
                    he
                };
                (
                    insert_half_edge(v0, v1),
                    insert_half_edge(v1, v2),
                    insert_half_edge(v2, v0),
                )
            };

            half_edges[he0 as usize].origin = v0;
            half_edges[he0 as usize].prev = he2;
            half_edges[he0 as usize].next = he1;
            half_edges[he0 as usize].incident_face = Some(face_index);

            half_edges[he1 as usize].origin = v1;
            half_edges[he1 as usize].prev = he0;
            half_edges[he1 as usize].next = he2;
            half_edges[he1 as usize].incident_face = Some(face_index);

            half_edges[he2 as usize].origin = v2;
            half_edges[he2 as usize].prev = he1;
            half_edges[he2 as usize].next = he0;
            half_edges[he2 as usize].incident_face = Some(face_index);
        }

        // link twins
        struct BoundaryEdge {
            src: VertexIndex,
            dst: VertexIndex,
            he: HalfEdgeIndex,
        }
        let mut boundary_edges = BTreeMap::new();

        for i in 0..half_edges.len() {
            let v0 = half_edges[i].origin;
            let v1 = half_edges[half_edges[i].next as usize].origin;
            if let Some(twin) = half_edge_map.get(&(v1, v0)) {
                half_edges[i].twin = *twin;
            } else {
                // create twin
                let twin = half_edges.len() as HalfEdgeIndex;
                half_edges[i].twin = twin;
                half_edges.push(HalfEdge {
                    origin: v1,
                    twin: i as HalfEdgeIndex,
                    next: 0,
                    prev: 0,
                    incident_face: None,
                });
                // boundary edge pointing to v0 starting from v1
                boundary_edges.insert(
                    v1,
                    BoundaryEdge {
                        src: v1,
                        dst: v0,
                        he: i as HalfEdgeIndex,
                    },
                );
                half_edge_map.insert((v1, v0), twin);
            }
        }

        // chain boundary edges
        while !boundary_edges.is_empty() {
            let (_, mut be) = boundary_edges.pop_first().unwrap();
            let start = be.src;
            let mut cycle = vec![be.he];
            loop {
                be = boundary_edges.remove(&be.dst).expect("invalid boundary");
                cycle.push(be.he);
                if be.dst == start {
                    break;
                }
            }
            // link up cycle
            for i in 0..cycle.len() {
                let next = (i + 1) % cycle.len();
                half_edges[cycle[i] as usize].next = cycle[next];
                half_edges[cycle[next] as usize].prev = cycle[i];
            }
        }

        Self {
            vertices,
            faces,
            half_edges,
        }
    }

    fn dump(&self) {
        for (i, v) in self.vertices.iter().enumerate() {
            println!("v[{}]: {:?}", i, v.coordinates);
        }
        for (i, f) in self.faces.iter().enumerate() {
            println!("f[{}]: {:?}", i, f.incident_edge);
        }
        for (i, he) in self.half_edges.iter().enumerate() {
            println!(
                "he[{}]: origin={}, twin={}, next={}, prev={}, incident_face={:?}",
                i, he.origin, he.twin, he.next, he.prev, he.incident_face
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_obj() {}
}
