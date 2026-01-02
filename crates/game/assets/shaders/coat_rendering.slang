import common;
import scene_info;

struct Stroke {
    float3 ref_position;
    uint start_vertex;
    uint vertex_count;
    uint16_t brush_index;
    uint16_t width_profile;
    uint16_t color_ramp;
};

struct StrokeVertex {
    float3 position;
    Srgba8 color;
    float3 normal;
    float arclength;
    float3 tangent;
};

struct ExpandedStrokeVertex {
    float4 clip_position;
    //Srgba8 color;
};

struct ExpandStrokesData {
    Stroke *strokes;
    StrokeVertex *vertices;
    ExpandedStrokeVertex* out_vertices;
    uint* out_indices;
    Atomic<uint> out_vertex_ptr;
    Atomic<uint> out_index_ptr;
    float main_width;
    float border_width;
};

struct ExpandStrokesPushConstants {
    SceneInfo *scene_info;
    ExpandStrokesData* data;
};

// Number of vertices per stroke vertex after expansion
static const uint NVERTEX = 9;
static const uint NINDEX = 10 * 3;

///////////////////////////////////////////////////////////////////////////////////////////////////
// Stroke expansion
///////////////////////////////////////////////////////////////////////////////////////////////////

// p_l_0 --- p --- p_r_0
struct ProjectedVertex {
    float4 p_l_0;
    float4 p;
    float4 p_r_0;
    float3 n_view;
};

static const uint MAX_STROKE_VERTICES = 128;
groupshared ProjectedVertex[MAX_STROKE_VERTICES] s_projected_vertices;
groupshared uint s_vertex_offset;
groupshared uint s_index_offset;


float2 screen_space_normal(float4 p0, float4 p, float4 p1, float2 px_size)
{
    float2 v0 = normalize((p.xy/p.w - p0.xy/p0.w) / px_size);
    float2 v1 = normalize((p1.xy/p1.w - p.xy/p.w) / px_size);
    float2 vt;
    if (all(p0 == p)) {
        vt = v0;
    } else if (all(p1 == p)) {
        vt = v1;
    } else {
        float2 vt = 0.5 * (v0 + v1);
    }
    float2 n = float2(-vt.y, vt.x);
    return n;
}

float compute_curvature(float2 p_a, float2 p_b, float2 p_c)
{
    float2 ab = p_b - p_a;
    float2 ac = p_c - p_a;
    float2 bc = p_c - p_b;

    float ab_len = length(ab);
    float ac_len = length(ac);
    float bc_len = length(bc);

    float area = abs(cross(float3(ab, 0.0), float3(ac, 0.0)).z) * 0.5;
    if (area < 1e-5) {
        return 0.0;
    }

    float radius = (ab_len * ac_len * bc_len) / (4.0 * area);
    return 1.0 / radius;
}

[shader("compute")]
[numthreads(MAX_STROKE_VERTICES, 1, 1)]
void expand_strokes_main(uint3 group_id : SV_GroupID, uint3 group_thread_id: SV_GroupThreadID, uniform ExpandStrokesPushConstants u)
{
    // load stroke
    let stroke_index = group_id.x;
    let stroke = u.data.strokes[stroke_index];

    // each thread processes one vertex and expands it along the normal
    // i == vertex index
    let i = group_thread_id.x;
    if (i >= stroke.vertex_count) {
        return;
    }
    let vertex = u.data.vertices[stroke.start_vertex + i];

    let n = vertex.normal;
    let tu = vertex.tangent;
    let tv = cross(tu, n);
    let half_width = u.data.main_width * 0.5;
    let p_l = vertex.position - half_width * tv;
    let p_r = vertex.position + half_width * tv;

    // screen-space tangent direction
    let tv_cs = normalize(mul(u.scene_info.view_proj_matrix, float4(tv, 0.0)).xy);

    float2 px_size = 2. / u.scene_info.screen_size; // float2(2. / u.screenSize.x, 2. / screenHeight); // pixel size in clip space
    float2 inv_px_size = u.scene_info.screen_size * 0.5;

    // project vertices to clip space
    let p_view = mul(u.scene_info.view_matrix, float4(vertex.position, 1.0));
    let p_clip = mul(u.scene_info.view_proj_matrix, float4(vertex.position, 1.0));
    var p_l_0_clip = mul(u.scene_info.view_proj_matrix, float4(p_l, 1.0));
    var p_r_0_clip = mul(u.scene_info.view_proj_matrix, float4(p_r, 1.0));

    //let tv_clip = mul(u.scene_info.view_proj_matrix, float4(tv, 0.0));
    //let n_clip = mul(u.scene_info.view_proj_matrix, float4(n, 0.0));
    //p_r_0_clip.xy += 2.0 * half_width * inv_px_size * tv_clip.xy * p_r_0_clip.w;
    //p_l_0_clip.xy -= 2.0 * half_width * inv_px_size * tv_clip.xy * p_l_0_clip.w;

    // view-space normal
    let n_view = normalize(mul(u.scene_info.view_matrix, float4(normalize(n), 0.0)).xyz);

    s_projected_vertices[i] = ProjectedVertex(
        p_l_0_clip,
        p_clip,
        p_r_0_clip,
        n_view
    );

    GroupMemoryBarrierWithGroupSync();

    // Compute curvature in screen-space
    let i_0 = max(0, i - 1);
    let i_1 = min(stroke.vertex_count - 1, i + 1);

    // check for flips in the direction of the normal relative to the screen
    let view_dir = normalize(-p_view.xyz);
    let flip = sign(dot(s_projected_vertices[i_0].n_view, view_dir)) - sign(dot(s_projected_vertices[i].n_view, view_dir));
    //let flip = dot(s_projected_vertices[i].n_view, view_dir);


    // Compute screen-space normals & curvature
    var n_ss = screen_space_normal(
        s_projected_vertices[i_0].p,
        s_projected_vertices[i].p,
        s_projected_vertices[i_1].p,
        px_size);

    //// compute circular arc through three points (p0, p, p1)
    //let k = compute_curvature(
    //    s_projected_vertices[i_0].p.xy / s_projected_vertices[i_0].p.w * inv_px_size,
    //    s_projected_vertices[i].p.xy / s_projected_vertices[i].p.w * inv_px_size,
    //    s_projected_vertices[i_1].p.xy / s_projected_vertices[i_1].p.w * inv_px_size);


    var p_l_1 = s_projected_vertices[i].p_l_0;
    p_l_1.xy += n_ss * u.data.border_width * px_size * p_l_1.w;

    var p_r_1 = s_projected_vertices[i].p_r_0;
    p_r_1.xy -= n_ss * u.data.border_width * px_size * p_r_1.w;



    //var curvature_palette = float4(float3(curvature)*10000, 255);
    //if (flip != 0.0) {
    //    curvature_palette.xyz = float3(smoothstep(1.1, 1.1, abs(flip) * 10) * 255);
    //}
    //linear_to_srgba8(float4(palette(curvature, float3(0.5, 0.5, 0.5), float3(0.5, 0.5, 0.5), float3(1.0, 1.0, 1.0), float3(0.00, 0.33, 0.67)), 1.0));


    //let p_r_0_screen = (s_projected_vertices[i].p_r_0.xy / s_projected_vertices[i].p_r_0.w) * inv_px_size;
    //let p_l_0_screen = (s_projected_vertices[i].p_l_0.xy / s_projected_vertices[i].p_l_0.w) * inv_px_size;
    //if (dot(p_r_0_screen - p_l_0_screen, n_r) > 0.0) {
    //    n_r = -n_r;
    //    n_l = -n_l;
    //}

    //n_l = tv_cs;
    //n_r = tv_cs;



    // allocate space for vertices (NVERTEX vertices per stroke vertex)
    if (i == 0) {
        s_vertex_offset = u.data.out_vertex_ptr.add(NVERTEX * stroke.vertex_count);
        s_index_offset = u.data.out_index_ptr.add(6 * (NVERTEX- 1) * (stroke.vertex_count - 1));
    }



    GroupMemoryBarrierWithGroupSync();


    var color = vertex.color;
    if (flip != 0.0) {
        color = srgba8(255,0,0,255);
    }

    // write expanded geometry
    u.data.out_vertices[s_vertex_offset + i*NVERTEX] = ExpandedStrokeVertex(p_l_1);
    u.data.out_vertices[s_vertex_offset + i*NVERTEX+1] = ExpandedStrokeVertex(p_l_0_clip);
    u.data.out_vertices[s_vertex_offset + i*NVERTEX+2] = ExpandedStrokeVertex(p_l_0_clip);
    u.data.out_vertices[s_vertex_offset + i*NVERTEX+3] = ExpandedStrokeVertex(p_clip);
    u.data.out_vertices[s_vertex_offset + i*NVERTEX+4] = ExpandedStrokeVertex(p_r_0_clip);
    u.data.out_vertices[s_vertex_offset + i*NVERTEX+5] = ExpandedStrokeVertex(p_r_0_clip);
    u.data.out_vertices[s_vertex_offset + i*NVERTEX+6] = ExpandedStrokeVertex(p_r_1);

    u.data.out_vertices[s_vertex_offset + i*NVERTEX+7] = ExpandedStrokeVertex(p_r_1);
    u.data.out_vertices[s_vertex_offset + i*NVERTEX+8] = ExpandedStrokeVertex(p_r_1);


    if (i > 0) {
        var idx_ptr = s_index_offset + (i - 1) * NINDEX;
        var vbase = s_vertex_offset + (i - 1) * NVERTEX;

        var ptr = u.data.out_indices + idx_ptr;
        *ptr = vbase + 0; ptr++;
        *ptr = vbase + NVERTEX + 1; ptr++;
        *ptr = vbase + NVERTEX; ptr++;
        *ptr = vbase + 0; ptr++;
        *ptr = vbase + 1; ptr++;
        *ptr = vbase + NVERTEX + 1; ptr++;

        *ptr = vbase + 2; ptr++;
        *ptr = vbase + NVERTEX + 3; ptr++;
        *ptr = vbase + NVERTEX + 2; ptr++;
        *ptr = vbase + 2; ptr++;
        *ptr = vbase + 3; ptr++;
        *ptr = vbase + NVERTEX + 3; ptr++;

        *ptr = vbase + 3; ptr++;
        *ptr = vbase + NVERTEX + 4; ptr++;
        *ptr = vbase + NVERTEX + 3; ptr++;
        *ptr = vbase + 3; ptr++;
        *ptr = vbase + 4; ptr++;
        *ptr = vbase + NVERTEX + 4; ptr++;

        *ptr = vbase + 5; ptr++;
        *ptr = vbase + NVERTEX + 6; ptr++;
        *ptr = vbase + NVERTEX + 5; ptr++;
        *ptr = vbase + 5; ptr++;
        *ptr = vbase + 6; ptr++;
        *ptr = vbase + NVERTEX + 6; ptr++;

        *ptr = vbase + 7; ptr++;
        *ptr = vbase + NVERTEX + 8; ptr++;
        *ptr = vbase + NVERTEX + 7; ptr++;
        *ptr = vbase + 7; ptr++;
        *ptr = vbase + 8; ptr++;
        *ptr = vbase + NVERTEX + 8; ptr++;
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Debug
///////////////////////////////////////////////////////////////////////////////////////////////////

struct SweepData {
    Stroke *strokes;
    StrokeVertex *vertices;
    PushBuffer<ExpandedStrokeVertex> out_vertices;
    PushBuffer<uint> out_indices;
    PushBuffer<ExpandedStrokeVertex> out_contour_lines;
};

struct SweepPushConstants {
    SceneInfo *scene_info;
    SweepData* data;
};

[shader("compute")]
[numthreads(MAX_STROKE_VERTICES, 1, 1)]
void sweep_stroke_main(uint3 group_id : SV_GroupID, uint3 group_thread_id: SV_GroupThreadID, uniform SweepPushConstants* u)
{


}


///////////////////////////////////////////////////////////////////////////////////////////////////
// Debug
///////////////////////////////////////////////////////////////////////////////////////////////////

struct DebugStrokesVertexOutput {
    float4 pos: SV_Position;
    [[vk::location(0)]] float4 color;
    [[vk::location(1)]] float2 local_pos;
};

struct DebugStrokesPushConstants {
    SceneInfo* scene_info;
    ExpandedStrokeVertex* vertices;
    uint* indices;
    uint index_count;
};

[shader("vertex")]
DebugStrokesVertexOutput debug_strokes_vertex(uint vertex_id : SV_VertexID, uniform DebugStrokesPushConstants* u)
{
    let index = u.indices[vertex_id];
    let vertex = u.vertices[index];
    DebugStrokesVertexOutput o;
    o.pos = vertex.clip_position;
    o.color = srgba8_to_float4(srgba8(0, 255, 0, 255));
    return o;
}

[shader("fragment")]
float4 debug_strokes_fragment(DebugStrokesVertexOutput i) {
    return i.color;
}
