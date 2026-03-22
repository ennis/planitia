use crate::Geo;

/// Helper function to convert polygons into a triangle mesh.
///
/// # Arguments
///
/// * `emit_vertex` - called for each output vertex with the houdini point index. Should return the vertex index in the output mesh.
/// * `emit_triangle` - called for each output triangle with the three vertex indices.
pub fn polygons_to_triangle_mesh<VertexFn, TriangleFn>(
    g: &Geo,
    mut emit_vertex: VertexFn,
    mut emit_triangle: TriangleFn,
) where
    VertexFn: FnMut(&Geo, u32, u32) -> u32,
    TriangleFn: FnMut(u32, u32, u32),
{
    // whether the point index has been added to the vertex list
    // (houdini point index -> our mesh vertex index)
    //let mut inserted_points = HashMap::<u32, u32>::new();
    let mut cur_indices = Vec::new();

    for prim in g.polygons() {
        if !prim.closed {
            // skip polylines
            continue;
        }

        for (i, vi) in prim.vertices().enumerate() {
            let vertex_index = emit_vertex(g, vi, prim.primitive_index);
            cur_indices.push(vertex_index);

            if i >= 2 {
                // emit triangle
                emit_triangle(cur_indices[0], cur_indices[i - 1], cur_indices[i]);
            }
        }

        cur_indices.clear()
    }
}


/// Helper function to convert polygons into a triangle mesh.
///
/// # Arguments
///
/// * `emit_vertex` - called for each output vertex with the houdini point index. Should return the vertex index in the output mesh.
/// * `emit_triangle` - called for each output triangle with the three vertex indices.
pub fn polygons_to_triangle_mesh_2<Vertex, VertexFn, TriangleFn>(
    g: &Geo,
    mut emit_vertex: VertexFn,
    mut emit_triangle: TriangleFn,
) where
    VertexFn: FnMut(&Geo, u32, u32) -> Vertex,
    TriangleFn: FnMut(&Vertex, &Vertex, &Vertex),
{
    // whether the point index has been added to the vertex list
    // (houdini point index -> our mesh vertex index)
    //let mut inserted_points = HashMap::<u32, u32>::new();
    //let mut cur_indices = Vec::new();
    let mut cur_vertices = Vec::new();

    for prim in g.polygons() {
        if !prim.closed {
            // skip polylines
            continue;
        }

        for (i, vi) in prim.vertices().enumerate() {
            cur_vertices.push(emit_vertex(g, vi, prim.primitive_index));

            if i >= 2 {
                // emit triangle
                emit_triangle(&cur_vertices[0], &cur_vertices[i - 1], &cur_vertices[i]);
            }
        }

        cur_vertices.clear()
    }
}
