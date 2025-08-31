import shared;

static const uint POSITION_ATTR = 0;
static const uint NORMAL_ATTR = 1;
static const uint TEXCOORD_ATTR = 2;
static const uint COLOR_ATTR = 3;

struct VertexOutput {
    float4 position : SV_Position;
    [[vk::location(0)]] float3 normal;
    [[vk::location(1)]] float2 texcoord;
    [[vk::location(2)]] float4 color;
    [[vk::location(3)]] float4 viewPosition;
};


// Vertex shader for geometry rendering.
// This should be called in non-indexed mode, i.e. SV_VertexID should vary per faceVertex, and not shared between faces.
//
// Attributes are stored in storage buffers, and are indexed by indices in `geom.indices`. The index is the same
// for all attributes of a vertex.
[shader("vertex")]
VertexOutput geometryVertexShader(uint vertexID : SV_VertexID, uniform GeometryData geom) {

    let sceneParams = geom.sceneParams;
    let index = geom.indices[vertexID];
    
    let position = geom.position[index];
    let normal = (geom.normal != 0) ? geom.normal[index] : float3(0.0, 0.0, 0.0);
    let texcoord = (geom.texcoord != 0) ? geom.texcoord[index] : float2(0.0, 0.0);
    let color = (geom.color != 0) ? geom.color[index] : float4(1.0, 0.0, 1.0, 1.0);

    //let viewProjectionMatrix = mul(sceneParams.projectionMatrix, sceneParams.viewMatrix);
    let worldPosition = mul(geom.modelMatrix, float4(position, 1.0));
    let viewPosition = mul(sceneParams.viewMatrix, worldPosition);
    let clipPosition = mul(sceneParams.projectionMatrix, viewPosition);

    let worldNormal = mul(geom.modelMatrix, float4(normal, 0.0));
    let viewNormal = normalize(mul(sceneParams.viewMatrix, worldNormal).xyz);

    VertexOutput output;
    output.position = clipPosition;
    output.normal = viewNormal;
    output.texcoord = texcoord;
    output.viewPosition = viewPosition;
    output.color = color;
    return output;
}

[shader("fragment")]
float4 geometryFragmentShader(VertexOutput input) {
    return float4(input.normal,1.0);
}