struct LineVertex {
    float3 position;
    uint8_t4 color;
    uint flags;
};

// Buffer containing polyline vertices.
[[vk::binding(0,0)]] StructuredBuffer<LineVertex, ScalarDataLayout> vertices;

struct VertexOutput {
    float4 pos: SV_Position;
    [[vk::location(0)]] float4 color;
    [[vk::location(1)]] float2 localPos;
};

struct PushConstants {
    float4x4 viewProjectionMatrix;
    uint startVertex;
    uint vertexCount;
    float lineWidth;
    float filterWidth;
    float2 screenSize;
};

[shader("vertex")]
VertexOutput line_drawing_vertex(uint vertex_id : SV_VertexID, uniform PushConstants u)
{
    let lineVtxID = vertex_id / 2;

    bool lineStart = lineVtxID == 0;
    bool lineEnd = lineVtxID == (u.vertexCount - 1);

    let v = vertices[u.startVertex + lineVtxID];
    let v0 = lineStart ? v : vertices[u.startVertex + lineVtxID - 1];
    let v1 = lineEnd ? v : vertices[u.startVertex + lineVtxID + 1];

    // project vertices in clip space
    let p = mul(u.viewProjectionMatrix, float4(v.position, 1.));
    let p0 = mul(u.viewProjectionMatrix, float4(v0.position, 1.));
    let p1 = mul(u.viewProjectionMatrix, float4(v1.position, 1.));

    // half-width + anti-aliasing margin
    // clamp the width to 1.0, below that we just fade out the line
    float hw_aa = max(u.lineWidth, 1.) * .5 + u.filterWidth * sqrt(2.);

    float4 a = p;
    float4 b = p;

    float2 pxSize = 2. / u.screenSize; // float2(2. / u.screenSize.x, 2. / screenHeight); // pixel size in clip space

    if (lineStart) {
        float2 v = p1.xy/p1.w - p.xy/p.w;
        float2 n = hw_aa * pxSize * normalize(pxSize * float2(-v.y, v.x));
        a.xy -= n * a.w;
        b.xy += n * b.w;

    } else if (lineEnd) {
        float2 v = p.xy/p.w - p0.xy/p0.w;
        float2 n = hw_aa * pxSize * normalize(pxSize * float2(-v.y, v.x));
        a.xy -= n * a.w;
        b.xy += n * b.w;
    }
    else {
        float2 v0 = normalize((p.xy/p.w - p0.xy/p0.w) / pxSize);
        float2 v1 = normalize((p1.xy/p1.w - p.xy/p.w) / pxSize);
        float2 vt = 0.5 * (v0 + v1);
        float2 n = float2(-vt.y, vt.x);
        // half-width / sin(theta/2)
        float d = hw_aa / max(cross(float3(v0, 0.0), float3(n, 0.0)).z, 0.05);
        // miter points
        a.xy -= d * n * pxSize * a.w;
        b.xy += d * n * pxSize * b.w;
    }

    VertexOutput o;
    o.pos = (vertex_id % 2 == 0) ? b : a;
    o.color = float4(v.color) / 255.0;
    o.localPos = (vertex_id % 2 == 0) ? float2(0.0, -hw_aa) : float2(1.0, hw_aa);
    return o;
}

////////////////////////////////////////////////////////////////////////

struct FragmentOutput
{
    [[vk::location(0)]] float4 color;
};


[shader("fragment")]
FragmentOutput line_drawing_fragment(VertexOutput v, uniform PushConstants u) {
    FragmentOutput o;
    // clamped width
    float width1 = max(u.lineWidth, 1.0);
    float h = width1 * 0.5;
    float y = abs(v.localPos.y);
    //float filterWidth = 1.5;
    float halfFilterWidth = u.filterWidth * 0.5;
    float alpha = (clamp((y + h + halfFilterWidth), 0., width1) - clamp((y + h - halfFilterWidth), 0., width1)) / u.filterWidth;
    alpha *= min(u.lineWidth, 1.0);
    o.color = v.color;
    o.color.a *= alpha;
    return o;
}

////////////////////////////////////////////////////////////////////////

// Converts Normalized Device Coordinates (NDC) to window coordinates.
float3 ndc2win(float3 ndc, float2 screenSize)
{
    return float3(ndc.xy * .5 + .5, ndc.z) * float3(screenSize, 1.);
}

// Converts window coordinates to Normalized Device Coordinates (NDC).
float3 win2ndc(float3 window, float2 screenSize)
{
    return float3(window.xy / float2(screenSize) * 2. - 1., window.z);
}

// Projects a vertex from world space to clip space.
float4 project(float3 pos, float4x4 viewProjectionMatrix)
{
    return mul(viewProjectionMatrix, float4(pos, 1.));
}
