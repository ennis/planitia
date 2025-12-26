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
    uint cluster_count;
    uint vertex_count;
    OutlineVertex* out_vertices;
    uint* out_indices;
    DrawIndirectCommand* out_draw_command;
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
        let v0offset = float4(v0.x, v0.y + sign(v0.y) * 0.01, v0.z, v0.w);
        let v1offset = float4(v1.x, v1.y + sign(v1.y) * 0.01, v1.z, v1.w);

        uint base_vertex;
        InterlockedAdd(u.vertex_count, 4, base_vertex);
        uint base_index;
        InterlockedAdd(u.out_draw_command.vertex_count, 6, base_index);

        //let v0dbg = float4(2.0 * float2(base_index, cli*7) / u.scene_info.screen_size - 1.0, 0.0, 1.0);
        //let v1dbg = float4(2.0 * float2(base_index + 6, cli*7) / u.scene_info.screen_size - 1.0, 0.0, 1.0);
        //let v2dbg = float4(2.0 * float2(base_index, cli*7 + 6) / u.scene_info.screen_size - 1.0, 0.0, 1.0);
        //let v3dbg = float4(2.0 * float2(base_index + 6, cli*7 + 6) / u.scene_info.screen_size - 1.0, 0.0, 1.0);

        u.out_vertices[base_vertex] = OutlineVertex(v0);
        u.out_vertices[base_vertex+1] = OutlineVertex(v1);
        u.out_vertices[base_vertex+2] = OutlineVertex(v0offset);
        u.out_vertices[base_vertex+3] = OutlineVertex(v1offset);

        u.out_indices[base_index] = base_vertex;
        u.out_indices[base_index+1] = base_vertex + 1;
        u.out_indices[base_index+2] = base_vertex + 2;
        u.out_indices[base_index+3] = base_vertex + 1;
        u.out_indices[base_index+4] = base_vertex + 3;
        u.out_indices[base_index+5] = base_vertex + 2;
    }
}