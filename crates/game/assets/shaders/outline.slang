import common;
import scene_info;

static const uint MAX_EDGES = 128;

struct ClusterEdge {
    uint16_t face_0;
    uint16_t face_1;
    uint16_t vertex_0;
    uint16_t vertex_1;
}

struct ClusterVertex {
    float3 position;
    float3 normal;
}

struct ClusterFace {
    uint16_t3 vertices;
    float3 normal;
}

struct NormalCone {
    float3 axis;
    float angle;
    float3 center;
    float radius;
}

struct OutlineVertex {
    float4 clip_position;
    Srgba8 color;
}

struct Cluster {
    /// Vertices in this cluster.
    ClusterVertex vertices[MAX_EDGES];
    /// All edges (represented by the two faces they connect) in this cluster
    ClusterEdge edges[MAX_EDGES];
    /// All faces in this cluster
    ClusterFace faces[MAX_EDGES*2];
    NormalCone normal_cone;
    uint16_t vertex_count;
    uint16_t edge_count;
    uint16_t face_count;
}

struct OutlineRootParams {
    SceneInfo* scene_info;
    Cluster* clusters;
    uint32_t cluster_count;
    PushBuffer<OutlineVertex> out_vertices;
    PushBuffer<uint32_t> out_indices;
}


[shader("compute")]
[numthreads(MAX_EDGES, 1, 1)]
void outline_main(uint3 group_id : SV_GroupID, uint3 group_thread_id: SV_GroupThreadID, uniform OutlineRootParams* u)
{
    // we spawn one workgroup per edge cluster, and each thread in the workgroup processes one edge

    // cluster index
    uint cli = group_id.x;
    if (cli >= u.cluster_count) {
        return;
    }

    Cluster* c = &u.clusters[cli];

    uint edge_index = group_thread_id.x;
    if (edge_index >= c.edge_count) {
        return;
    }

    let edge = c.edges[edge_index];
    var is_contour = false;

    if (edge.face_0 == 0xFFFF || edge.face_1 == 0xFFFF) {
        // boundary edge, skip for now
        is_contour = true;
    } else {
        let n0 = c.faces[edge.face_0].normal;
        let n1 = c.faces[edge.face_1].normal;
        let eye = u.scene_info.eye;
        let view = eye - c.vertices[edge.vertex_0].position;
        if (dot(n0, view) * dot(n1, view) < 0.0) {
            is_contour = true;
        }
    }

    if (is_contour) {
        let v0 = mul(u.scene_info.view_proj_matrix, float4(c.vertices[edge.vertex_0].position, 1.));
        let v1 = mul(u.scene_info.view_proj_matrix, float4(c.vertices[edge.vertex_1].position, 1.));
        let base_vertex = u.out_vertices.reserve(4);
        let base_index = u.out_indices.reserve(6);
        u.out_vertices.base[base_vertex] = OutlineVertex(v0, Srgba8(0, 255, 0, 255));
        u.out_vertices.base[base_vertex+1] = OutlineVertex(v1, Srgba8(0, 255, 0, 255));
        u.out_vertices.base[base_vertex+2] = OutlineVertex(v1 + float4(0.01, 0., 0., 0.), Srgba8(0, 255, 0, 255));
        u.out_vertices.base[base_vertex+3] = OutlineVertex(v0 + float4(0., 0.01, 0., 0.), Srgba8(0, 255, 0, 255));
        u.out_indices.base[base_index] = base_vertex;
        u.out_indices.base[base_index+1] = base_vertex + 1;
        u.out_indices.base[base_index+2] = base_vertex + 2;
        u.out_indices.base[base_index+3] = base_vertex;
        u.out_indices.base[base_index+4] = base_vertex + 2;
        u.out_indices.base[base_index+5] = base_vertex + 3;
    }
}