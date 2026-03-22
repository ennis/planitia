use crate::experiments::winged_edge_mesh::WingedEdgeMesh;
use math::Vec3;
use std::path::Path;
use crate::SceneInfo;

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct Point {
    position: Vec3,
    id: i32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct FaceVertex {
    point: u32,
    normal: Vec3,
}

struct ScreenSpaceContours {
    mesh: WingedEdgeMesh<Point, FaceVertex>,
}

impl ScreenSpaceContours {

    fn new() -> Self {
        Self {
            mesh: WingedEdgeMesh::new(&[], &[], |fv: &FaceVertex| fv.point),
        }
    }

    fn load_geometry(&mut self, path: &Path) {
        let geo = hgeo::Geo::load(path).unwrap();

        // convert points
        let points = (0..geo.point_count)
            .map(|ptnum| {
                let position = geo.point(ptnum as u32, "P");
                let id = geo.point(ptnum as u32, "id");
                Point { position, id }
            })
            .collect::<Vec<_>>();

        // convert polygons to triangle mesh and collect vertices
        let vertices = {
            let mut vertices = Vec::new();
            for prim in geo.polygons() {
                if !prim.closed {
                    continue;
                }
                let first_vertex = vertices.len();
                for (i, vi) in prim.vertices().enumerate() {
                    vertices.push(FaceVertex {
                        point: geo.vertexpoint(vi),
                        normal: geo.vertex(vi, "N"),
                    });
                    if i > 2 {
                        let v0 = vertices[first_vertex];
                        let v1 = vertices[first_vertex + i - 1];
                        //let v2 = vertices[first_vertex + i];
                        vertices.push(v0);
                        vertices.push(v1);
                    }
                }
            }
            vertices
        };

        self.mesh = WingedEdgeMesh::new(&points, &vertices, |fv: &FaceVertex| fv.point);
    }

    fn render(&mut self, cmd: &mut gpu::CommandBuffer,
              color_target: &gpu::Image,
              depth_target: &gpu::Image,
              scene_info: &SceneInfo)
    {
        //
    }
}
