
struct WEPoint<Data> {
    Data data;
}

struct WEFaceVertex<Data> {
    Data data;
    uint incident_edge;
}

struct WEFace {
    uint edges[3];
}

struct WingedEdge {
    uint src_vtx;
    uint dst_vtx;
    uint r_face;
    uint l_face;
    uint r_cw_edge;
    uint r_ccw_edge;
    uint l_ccw_edge;
    uint l_cw_edge;
}

struct WEMeshData<PointData, FaceVertexData> {
    WEPoint<PointData>* points;
    WEFaceVertex<FaceVertexData>* face_vertices;
    WingedEdge *edges;
    WEFace *faces;
    uint point_count;
    uint face_vertex_count;
    uint edge_count;
    uint face_count;
}
