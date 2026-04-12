import common;
import scene_info;
import ridge_detection;
import winged_edge;


struct PointData {
    float3 position;
    uint group_id;
}

struct FaceVertexData {
    uint point;
    float3 normal;
}

struct OutlineVertex {
    // clip-space position
    float4 clip_position;
    // normal orientation in screen space
    float angle;
    // CLUSTER_EDGE_FLAGS*
    uint16_t flags;
    // sub-object ID
    uint16_t group_id;
    float2 local_pos;
}

struct ContourSegment {
    float3 position_0;
    uint32_t point_0;
    float3 position_1;
    uint32_t point_1;
    uint16_t group_id;
}

struct ContourSegmentBuffer {
    uint32_t count;
    ContourSegment segments[];
}

struct ContourPoint {
    float3 position;
    uint group_id;
    uint next;
}


// Common root params for contour extraction and rendering
struct ContourRootParams {
    SceneInfo* scene_info;
    WEMeshData<PointData, FaceVertexData> mesh;

    // Camera position used for contour extraction,
    // can be different from scene_info.eye for debugging purposes
    float3 eye;

    // contour extraction output
    ContourSegmentBuffer* segments;

    // contour ranking data
    uint64_t* contours_rank_successors_0;
    uint64_t* contours_rank_successors_1;

    // expanded contour data
    uint expanded_contour_vertex_count;
    OutlineVertex* expanded_contour_vertices;
    uint* expanded_contour_indices;
    DrawIndirectCommand* expanded_contours_draw_command;

    // shading
    float3 main_light_direction;    // in world space
    Srgba8 silhouette_color;
    float line_width;
    float filter_width;
    float isophote_offset;

    // textures
    Texture2D.Handle depth_texture;
    Texture2D<uint16_t4>.Handle angle_texture;
    Texture2D<float4>.Handle normal_texture;
    Texture2D<float4>.Handle shading_texture;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// Depth pass
///////////////////////////////////////////////////////////////////////////////////////////////////

struct DepthPassVertexOutput {
    float4 pos: SV_Position;
    [[vk::location(0)]] float4 color;
    [[vk::location(1)]] float2 local_pos;
};

[shader("vertex")]
[pass("depth_pass")]
DepthPassVertexOutput depth_pass_vertex(uint vertex_id : SV_VertexID, uint draw_id: SV_DrawIndex, uniform ContourRootParams* u)
{
    let vertex = u.mesh.face_vertices[vertex_id].data;
    let point = u.mesh.points[vertex.point].data;
    DepthPassVertexOutput o;
    o.pos = mul(u.scene_info.view_proj_matrix, float4(point.position, 1.0));
    return o;
}

[shader("fragment")]
[pass("depth_pass")]
void depth_pass_fragment(DepthPassVertexOutput i) {
    // nothing
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Base render + depth pre-pass
///////////////////////////////////////////////////////////////////////////////////////////////////

struct BaseRenderVSOut {
    float4 pos: SV_Position;
    float4 color;
    float3 normal;
    float3 view;
    nointerpolation int meshlet_id;
    nointerpolation int group_id;
}

struct BaseRenderFSOut {
    [[vk::location(0)]] float4 color;
    [[vk::location(1)]] float3 normal;
    [[vk::location(2)]] uint16_t4 aux;
}


[shader("vertex")]
[pass("base_render")]
BaseRenderVSOut base_render_vertex(uint vertex_id : SV_VertexID, uint draw_id: SV_DrawIndex, uniform ContourRootParams* u)
{
    let vertex = u.mesh.face_vertices[vertex_id].data;
    let point = u.mesh.points[vertex.point].data;

    BaseRenderVSOut o;
    o.pos = mul(u.scene_info.view_proj_matrix, float4(point.position, 1.0));
    o.normal = mul(float3x3(u.scene_info.view_matrix), vertex.normal);
    o.meshlet_id = draw_id;
    o.group_id = point.group_id;
    let pos_view = mul(u.scene_info.view_matrix, float4(point.position, 1.0)).xyz;
    o.view = normalize(pos_view);
    return o;
}

[shader("fragment")]
[pass("base_render")]
BaseRenderFSOut base_render_fragment(BaseRenderVSOut i, uniform ContourRootParams* u) {
    BaseRenderFSOut o;
    float3 normal = normalize(i.normal);
    let light_dir = -i.view;
    //let light_dir = -normalize(u.main_light_direction);
    float light_intensity = saturate(dot(normal, light_dir));
    o.color = srgba8_to_float4(u.silhouette_color);
    o.color.rgb *= light_intensity;
    o.normal = normal;
    o.aux = uint16_t4(i.group_id, i.meshlet_id, 0, 0);
    // colorize per cluster
    //o.color.rgb = palette(float(i.meshlet_id) * 0.1, float3(0.5,0.5,0.5),float3(0.5,0.5,0.5),float3(1.0,1.0,1.0),float3(0.0,0.33,0.67));
    return o;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// Outline extraction
///////////////////////////////////////////////////////////////////////////////////////////////////


float point_orientation(float3 point, float3 cam, float3 normal, float offset = 0.0) {
    let view = normalize(point - cam);
    return dot(normalize(normal), view) - offset;
}

static const uint EXTRACT_CONTOURS_THREAD_GROUP_SIZE = 128;

[shader("compute")]
[numthreads(EXTRACT_CONTOURS_THREAD_GROUP_SIZE, 1, 1)]
[pass("extract_interpolated_contours")]
void extract_interpolated_contours(uint3 thread_id: SV_DispatchThreadID, uniform ContourRootParams* u)
{
    // one thread per face

    // restrict to valid faces
    uint face_index = thread_id.x;
    if (face_index >= u.mesh.face_count) {
        return;
    }

    // load face vertices
    let face = u.mesh.faces[face_index];

    // face vertices are stored in consecutive order in the face vertex buffer
    let fv1 = u.mesh.get_face_vertex(face_index, 0).data;
    let fv2 = u.mesh.get_face_vertex(face_index, 1).data;
    let fv3 = u.mesh.get_face_vertex(face_index, 2).data;
    // fetch point data for vertices
    let v1 = u.mesh.points[fv1.point].data;
    let v2 = u.mesh.points[fv2.point].data;
    let v3 = u.mesh.points[fv3.point].data;

    // Evaluate point orientation relative to camera (the n.v term) for each vertex of the face.
    // negative orientation means point facing away, positive means facing towards camera.
    let o1 = point_orientation(v1.position, u.eye, fv1.normal, u.isophote_offset);
    let o2 = point_orientation(v2.position, u.eye, fv2.normal, u.isophote_offset);
    let o3 = point_orientation(v3.position, u.eye, fv3.normal, u.isophote_offset);

    // If all vertices point the same way (either all away or all towards camera),
    // there is no contour crossing this face.
    if (sign(o1) == sign(o2) && sign(o2) == sign(o3)) {
        return;
    }

    float3 interp1;
    float3 interp2;
    uint pkpt_1;
    uint pkpt_2;
    let flip1 = sign(o1) != sign(o2);   // does the orientation flip between v1 and v2?
    let flip2 = sign(o2) != sign(o3);   // does the orientation flip between v2 and v3?
    if (flip1 && flip2) {
        // contour passing between v1-v2 and v2-v3
        interp1 = (o2 * v1.position - o1 * v2.position) / (o2 - o1);
        interp2 = (o3 * v2.position - o2 * v3.position) / (o3 - o2);
        pkpt_1 = fv1.point;
        pkpt_2 = fv2.point;
    } else if (flip1) {
        // contour passing between v1-v2 and v3-v1
        interp1 = (o2 * v1.position - o1 * v2.position) / (o2 - o1);
        interp2 = (o1 * v3.position - o3 * v1.position) / (o1 - o3);
        pkpt_1 = fv1.point;
        pkpt_2 = fv3.point;
    } else {
        // contour passing between v2-v3 and v3-v1
        interp1 = (o3 * v2.position - o2 * v3.position) / (o3 - o2);
        interp2 = (o1 * v3.position - o3 * v1.position) / (o1 - o3);
        pkpt_1 = fv2.point;
        pkpt_2 = fv3.point;
    }

    // Write the contour segment
    uint e_i;
    InterlockedAdd(u.segments.count, 1, e_i);
    u.segments.segments[e_i] = ContourSegment(
        interp1,
        pkpt_1,
        interp2,
        pkpt_2,
        v1.group_id
    );
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// Contour list ranking and reordering
///////////////////////////////////////////////////////////////////////////////////////////////////

/*
static const uint SENTINEL = 0xFFFF_FFFF;
static const uint ROOT = 0x8000_0000;
static const uint RANK_BLOCK_SIZE = 256;

struct RankContoursRootParams {
     ContourRootParams* common;
     uint len;
     uint64_t* contours_rank_successors_0;
     uint64_t* contours_rank_successors_1;
}

[shader("compute")]
[numthreads(RANK_BLOCK_SIZE, 1, 1)]
[pass("break_contours_init")]
void break_contours_init(uint3 thread_id : SV_DispatchThreadID, uniform RankContoursRootParams* u) {
    let i = thread_id.x;
    if (i < u.len) {
        // extract_contours outputs global point indices as successors; convert to local contour point indices
        let global_next = u.common.contour_point_list[i].next;
        let local_next = global_next == SENTINEL ? SENTINEL : u.common.global_to_contour_index_map[global_next];
        u.common.contour_point_list[i].next = local_next;

        // rank/successors
        u.contours_rank_successors_0[i] = (uint64_t(local_next) << 32) | (local_next == SENTINEL ? 0 : i);
        u.contours_rank_successors_1[i] = (uint64_t(local_next) << 32) | (local_next == SENTINEL ? 0 : i);
    }
}

[shader("compute")]
[numthreads(RANK_BLOCK_SIZE, 1, 1)]
[pass("break_contours_step")]
void break_contours_step(uint3 thread_id : SV_DispatchThreadID, uniform RankContoursRootParams* u) {
    let i = thread_id.x;
    if (i >= u.len)
        return;

    let sr = u.contours_rank_successors_0[i];
    let s = uint(sr >> 32);
    let r = uint(sr & 0xFFFFFFFF);
    if (s == SENTINEL || s == i) {
        u.contours_rank_successors_1[i] = sr;
        return;
    }

    let ssr = u.contours_rank_successors_0[s];
    let ss = uint(ssr >> 32);
    let rr = uint(ssr & 0xFFFFFFFF);
    u.contours_rank_successors_1[i] = (uint64_t(ss) << 32) | min(r, rr);
}

[shader("compute")]
[numthreads(RANK_BLOCK_SIZE, 1, 1)]
[pass("rank_contours_init")]
void rank_contours_init(uint3 thread_id : SV_DispatchThreadID, uniform RankContoursRootParams* u) {
    let i = thread_id.x;
    if (i < u.len) {
        var successor = u.common.contour_point_list[i].next;
        let root = u.contours_rank_successors_0[i] & 0xFFFFFFFF;
        var init_rank = 1;
        if (root == i || successor == SENTINEL) {
            init_rank = 0;
            successor = SENTINEL;
        }
        u.contours_rank_successors_0[i] = (uint64_t(successor) << 32) | init_rank;
        u.contours_rank_successors_1[i] = (uint64_t(successor) << 32) | init_rank;
    }
}

[shader("compute")]
[numthreads(RANK_BLOCK_SIZE, 1, 1)]
[pass("rank_contours_step")]
void rank_contours_step(uint3 thread_id : SV_DispatchThreadID, uniform RankContoursRootParams* u) {
    let i = thread_id.x;
    if (i >= u.len)
       return;

    let sr = u.contours_rank_successors_0[i];
    let s = uint(sr >> 32);
    let r = uint(sr & 0xFFFFFFFF);
    if (s == SENTINEL || s == i) {
        u.contours_rank_successors_1[i] = sr;
        return;
    }

    let ssr = u.contours_rank_successors_0[s];
    let ss = uint(ssr >> 32);
    let rr = uint(ssr & 0xFFFFFFFF);
    u.contours_rank_successors_1[i] = (uint64_t(ss) << 32) | (r + rr);
}

[shader("compute")]
[numthreads(RANK_BLOCK_SIZE, 1, 1)]
[pass("rank_contours_finalize")]
void rank_contours_finalize(uint3 thread_id : SV_DispatchThreadID, uniform RankContoursRootParams* u) {
    let i = thread_id.x;
    if (i < u.len) {
        let sr = u.contours_rank_successors_0[i];
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// Subdivide contours
///////////////////////////////////////////////////////////////////////////////////////////////////

static const uint SUBDIVIDE_CONTOURS_GROUP_SIZE = 256;

struct SubdivideContoursRootParams {
    ContourRootParams* common;
    uint point_count;
}

[pass("setup_subdivide_contours")]
[shader("compute")]
[numthreads(1, 1, 1)]
void setup_subdivide_contours(uint3 thread_id : SV_DispatchThreadID, uniform SubdivideContoursRootParams* u) {
    u.point_count = u.common.contour_point_count;
}

[pass("subdivide_contours")]
[shader("compute")]
[numthreads(SUBDIVIDE_CONTOURS_GROUP_SIZE, 1, 1)]
void subdivide_contours(uint3 thread_id : SV_DispatchThreadID, uniform SubdivideContoursRootParams* u) {
    let i = thread_id.x;
    if (i >= u.point_count) {
        return;
    }

    let a = u.common.contour_point_list[i];
    let group_id = a.group_id;
    if (a.next == SENTINEL) {
        return;
    }
    let b = u.common.contour_point_list[a.next];

    let c_position = 0.25 * a.position + 0.75 * b.position;
    let a_position = 0.75 * a.position + 0.25 * b.position;

    // Allocate new point
    uint c_index;
    InterlockedAdd(u.common.contour_point_count, 1, c_index);

    // Update positions and links
    u.common.contour_point_list_subdiv[i].position = a_position;
    u.common.contour_point_list_subdiv[i].group_id = group_id;
    u.common.contour_point_list_subdiv[i].next = c_index;
    u.common.contour_point_list_subdiv[c_index].position = c_position;
    u.common.contour_point_list_subdiv[c_index].group_id = group_id;
    u.common.contour_point_list_subdiv[c_index].next = a.next;
}


[pass("finish_subdivide_contours")]
[shader("compute")]
[numthreads(SUBDIVIDE_CONTOURS_GROUP_SIZE, 1, 1)]
void finish_subdivide_contours(uint3 thread_id : SV_DispatchThreadID, uniform SubdivideContoursRootParams* u) {
    let i = thread_id.x;
    if (i >= u.common.contour_point_count) {
        return;
    }
    u.common.contour_point_list[i] = u.common.contour_point_list_subdiv[i];
}*/

///////////////////////////////////////////////////////////////////////////////////////////////////
// Expand outlines
///////////////////////////////////////////////////////////////////////////////////////////////////

static const uint EXPAND_CONTOURS_GROUP_SIZE = 256;

[pass("expand_interpolated_contours")]
[shader("compute")]
[numthreads(EXPAND_CONTOURS_GROUP_SIZE, 1, 1)]
void expand_interpolated_contours(uint3 thread_id : SV_DispatchThreadID, uniform ContourRootParams* u) {

    // one thread per contour segment
    let i = thread_id.x;
    if (i >= u.segments.count)
        return;

    let edge = u.segments.segments[i];

    float2 px_size = 2. / u.scene_info.screen_size;
    float width = u.line_width;
    float filter_width = u.filter_width;
    float hw_aa = max(width, 1.) * .5 + filter_width * sqrt(2.);

    var v0 = mul(u.scene_info.view_proj_matrix, float4(edge.position_0, 1.));
    var v1 = mul(u.scene_info.view_proj_matrix, float4(edge.position_1, 1.));

    float2 v = (v0.xy/v0.w - v1.xy/v1.w) / px_size;
    float2 vn = px_size * normalize(v);
    float2 n = px_size * normalize(float2(-v.y, v.x));

    let angle = atan(n.y / n.x);

    //v0.xy += hw_aa * vn * v0.w;
    //v1.xy -= hw_aa * vn * v1.w;

    var v0offset = v0;
    var v1offset = v1;
    v0.xy -= hw_aa * n * v0.w;
    v1.xy -= hw_aa * n * v1.w;
    v0offset.xy += hw_aa * n * v0offset.w;
    v1offset.xy += hw_aa * n * v1offset.w;

    uint vtx_ptr;
    uint idx_ptr;
    InterlockedAdd(u.expanded_contour_vertex_count, 4, vtx_ptr);
    InterlockedAdd(u.expanded_contours_draw_command.vertex_count, 6, idx_ptr);

    uint16_t flags = i /*is_outside_edge ? CLUSTER_EDGE_FLAG_BORDER :*/;

    let pointy_0 = 0.0; //u.mesh.vertices[base_vertex + edge.vertex_0].pointy;
    let pointy_1 = 1.0; //u.mesh.vertices[base_vertex + edge.vertex_1].pointy;

    u.expanded_contour_vertices[vtx_ptr] = OutlineVertex(v0, angle, flags, edge.group_id, float2(0.0, -hw_aa));
    u.expanded_contour_vertices[vtx_ptr+1] = OutlineVertex(v1, angle, flags, edge.group_id, float2(1.0, -hw_aa));
    u.expanded_contour_vertices[vtx_ptr+2] = OutlineVertex(v0offset, angle, flags, edge.group_id, float2(0.0, hw_aa));
    u.expanded_contour_vertices[vtx_ptr+3] = OutlineVertex(v1offset, angle, flags, edge.group_id, float2(1.0, hw_aa));

    u.expanded_contour_indices[idx_ptr] = vtx_ptr;
    u.expanded_contour_indices[idx_ptr+1] = vtx_ptr + 1;
    u.expanded_contour_indices[idx_ptr+2] = vtx_ptr + 2;
    u.expanded_contour_indices[idx_ptr+3] = vtx_ptr + 1;
    u.expanded_contour_indices[idx_ptr+4] = vtx_ptr + 3;
    u.expanded_contour_indices[idx_ptr+5] = vtx_ptr + 2;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Render outlines
///////////////////////////////////////////////////////////////////////////////////////////////////

struct RenderOutlinesVSOut {
    float4 pos: SV_Position;
    [[vk::location(0)]] float4 color;
    [[vk::location(1)]] float2 local_pos;
    [[vk::location(2)]] nointerpolation float width;
    [[vk::location(3)]] nointerpolation float filter_width;
};

[pass("render_outlines")]
[shader("vertex")]
RenderOutlinesVSOut render_outlines_vertex(uint vertex_id : SV_VertexID, uniform ContourRootParams* u)
{
    let index = u.expanded_contour_indices[vertex_id];
    let vertex = u.expanded_contour_vertices[index];
    RenderOutlinesVSOut o;
    o.pos = vertex.clip_position;
    o.local_pos = vertex.local_pos;
    o.color = float4(colorize_id(vertex.group_id), 1.0);
    o.width = u.line_width;
    o.filter_width = u.filter_width;
    return o;
}

struct RenderOutlinesFSOut {
    [[vk::location(0)]] float4 color;
    float depth: SV_Depth;
};

[pass("render_outlines")]
[shader("fragment")]
RenderOutlinesFSOut render_outlines_fragment(
    float3 frag_coord: SV_Position,
    RenderOutlinesVSOut vsout,
    uniform ContourRootParams* params)
{
    int3 coord = int3(int2(frag_coord.xy), 0);

    //int test = 0;
    //const float ref_depth = frag_coord.z;
    //test += (ref_depth <= params.depth_texture.Load(int3(coord.x - 1, coord.y - 1, 0), 0).r) ? 1 : 0;
    //test += (ref_depth <= params.depth_texture.Load(int3(coord.x,     coord.y - 1, 0), 0).r) ? 1 : 0;
    //test += (ref_depth <= params.depth_texture.Load(int3(coord.x + 1, coord.y - 1, 0), 0).r) ? 1 : 0;
    //test += (ref_depth <= params.depth_texture.Load(int3(coord.x - 1, coord.y,     0), 0).r) ? 1 : 0;
    //test += (ref_depth <= params.depth_texture.Load(int3(coord.x,     coord.y,     0), 0).r) ? 1 : 0;
    //test += (ref_depth <= params.depth_texture.Load(int3(coord.x + 1, coord.y,     0), 0).r) ? 1 : 0;
    //test += (ref_depth <= params.depth_texture.Load(int3(coord.x - 1, coord.y + 1, 0), 0).r) ? 1 : 0;
    //test += (ref_depth <= params.depth_texture.Load(int3(coord.x,     coord.y + 1, 0), 0).r) ? 1 : 0;
    //test += (ref_depth <= params.depth_texture.Load(int3(coord.x + 1, coord.y + 1, 0), 0).r) ? 1 : 0;
    //float4 color = vsout.color;
    //if (test < 2) {
        //discard;
        //color = float4(0.3, 0.3, 0.6, 1);
    //}

    RenderOutlinesFSOut o;
    // clamped width
    float width1 = max(vsout.width, 1.0);
    float h = width1 * 0.5;
    float y = abs(vsout.local_pos.y);
    //float filter_width = 1.5;
    float half_filter_width = vsout.filter_width * 0.5;
    float alpha = (clamp((y + h + half_filter_width), 0., width1) - clamp((y + h - half_filter_width), 0., width1)) / vsout.filter_width;
    alpha *= min(vsout.width, 1.0);
    o.color = vsout.color;
    o.color.a *= alpha;

    //o.color = float4(0.0, 0.0, 0.0, 0.0);

    return o;
}
