
public struct WEPoint<Data> {
    public Data data;
}

public struct WEFaceVertex<Data> {
    public Data data;
    public uint incident_edge;
}

public struct WEFace {
    public uint edges[3];
    public uint vertices[3];
}

public struct WingedEdge {
    public uint src_vtx;
    public uint dst_vtx;
    public uint r_face;
    public uint l_face;
    public uint r_cw_edge;
    public uint r_ccw_edge;
    public uint l_ccw_edge;
    public uint l_cw_edge;
}

public struct WEMeshData<PointData, FaceVertexData> {
    public WEPoint<PointData>* points;
    public WEFaceVertex<FaceVertexData>* face_vertices;
    public WingedEdge *edges;
    public WEFace *faces;
    public uint point_count;
    public uint face_vertex_count;
    public uint edge_count;
    public uint face_count;

    public WEFaceVertex<FaceVertexData> get_face_vertex(uint face_index, uint triangle_vertex) {
        return face_vertices[face_index * 3 + triangle_vertex];
    }
}
