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

// See documentation in outlines.rs
struct Cluster {
    ClusterVertex vertices[MAX_EDGES];
    ClusterEdge edges[MAX_EDGES];
    ClusterFace faces[MAX_EDGES*2];
    NormalCone normal_cone;
    uint16_t vertex_count;
    uint16_t edge_count;
    uint16_t face_count;
}

struct OutlineRootParams {
    [bind("scene_info")]
    SceneInfo* scene_info;

    [bind("clusters")]
    Cluster* clusters;

    [bind("cluster_count")]
    uint cluster_count;

    uint vertex_count;

    [bind("outline_vertices")]
    OutlineVertex* out_vertices;

    [bind("outline_indices")]
    uint* out_indices;

    [bind("outline_indirect_command")]
    [len(1)]
    DrawIndirectCommand* out_draw_command;
}

[shader("compute")]
[numthreads(MAX_EDGES, 1, 1)]
[pass("outline")]
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

    float2 px_size = 2. / u.scene_info.screen_size;
    float width = 1.0;
    float filter_width = 0.7;
    float hw_aa = max(width, 1.) * .5; //+ filter_width * sqrt(2.);

    if (is_contour) {
        var v0 = mul(u.scene_info.view_proj_matrix, float4(c.vertices[edge.vertex_0].position, 1.));
        var v1 = mul(u.scene_info.view_proj_matrix, float4(c.vertices[edge.vertex_1].position, 1.));
        var v0offset = v0;
        var v1offset = v1;

        float2 v = (v0.xy/v0.w - v1.xy/v1.w) / px_size;
        float2 n = px_size * normalize(float2(-v.y, v.x));
        v0.xy -= hw_aa * n * v0.w;
        v1.xy -= hw_aa * n * v1.w;
        v0offset.xy += hw_aa * n * v0offset.w;
        v1offset.xy += hw_aa * n * v1offset.w;

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

///////////////////////////////////////////////////////////////////////////////////////////////////
// Depth pass
///////////////////////////////////////////////////////////////////////////////////////////////////

struct DepthPassVertexOutput {
    float4 pos: SV_Position;
    [[vk::location(0)]] float4 color;
    [[vk::location(1)]] float2 local_pos;
};

struct DepthPassRootParams {
    SceneInfo* scene_info;
    Cluster* clusters;
    uint cluster_count;
};

[shader("vertex")]
[pass("depth_pass")]
DepthPassVertexOutput depth_pass_vertex(uint vertex_id : SV_VertexID, uint draw_id: SV_DrawIndex, uniform DepthPassRootParams* u)
{
    let face = u.clusters[draw_id].faces[vertex_id / 3];
    let vertex = u.clusters[draw_id].vertices[face.vertices[vertex_id % 3]];
    DepthPassVertexOutput o;
    o.pos = mul(u.scene_info.view_proj_matrix, float4(vertex.position, 1.0));
    return o;
}

[shader("fragment")]
[pass("depth_pass")]
void depth_pass_fragment(DepthPassVertexOutput i) {
    // nothing
}