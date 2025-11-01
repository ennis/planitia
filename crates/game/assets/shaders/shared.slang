// **WARNING**
//
// This file contains definitions shared between shader and host code.
// It is processed by `shader-bridge` to generate host-side code. As such, there are some restrictions on the syntax:
// - generic types other than (RW)Texture(n)D are not supported
// - no function definitions
// - no type definitions other than structs
// - no preprocessing directives

static const uint BINNING_TILE_SIZE = 16;
static const uint DRAW_CURVES_WORKGROUP_SIZE_X = 16;
static const uint DRAW_CURVES_WORKGROUP_SIZE_Y = 2;
static const uint BINPACK_SUBGROUP_SIZE = 32;
static const uint SUBGROUP_SIZE = 32;
static const uint MAX_VERTICES_PER_CURVE = 64;
static const uint MAX_LINES_PER_TILE = 64;

struct StrokeVertex {
    float3 pos;
    float s;
    uint8_t4 color;
    uint8_t width;
    uint8_t opacity;
};

struct Stroke {
    uint baseVertex;
    uint vertexCount;
    uint8_t brush;
    float arcLength;
};

struct TileLineData {
    float4 lineCoords;
    float2 paramRange;
    uint curveId;
    float depth;
};

struct TileData {
    TileLineData[MAX_LINES_PER_TILE] lines;
};

// 3D bezier control point.
struct ControlPoint {
    float3 pos;
    float3 color;
};

// Scene camera parameters.
struct SceneParams {
    float4x4 viewMatrix;
    float4x4 projectionMatrix;
    float4x4 viewProjectionMatrix;
    float3 eye;
    float nearClip;
    float farClip;
    float left;
    float right;
    float top;
    float bottom;
    uint2 viewportSize;
    float2 cursorPos;
    float time;
};


struct TemporalAverageParams {
    uint2 viewportSize;
    uint frame;
    float falloff;
    RWTexture2D<float4>.Handle newFrame;
    RWTexture2D<float4>.Handle avgFrame;
};

struct ComputeTestParams {
    uint elementCount;
    TileData* data;
    ControlPoint* controlPoints;
    RWTexture2D<float4>.Handle outputImage;
};


//  Represents a range of control points in the position buffer.
struct CurveDesc {
    float4 widthProfile;
    float4 opacityProfile;
    uint start;
    uint count;
    float2 paramRange;
    uint brushIndex;
};

struct DrawCurvesPushConstants {
    ControlPoint* controlPoints;
    CurveDesc* curves;
    SceneParams* sceneParams;
    uint baseCurveIndex;
    float strokeWidth;
    uint tileCountX;
    uint tileCountY;
    uint frame;
    TileData* tileData;
    uint* tileLineCount;
    Texture2D<float4>.Handle* brushTextures;
    RWTexture2D<float4>.Handle outputImage;
    SamplerState.Handle brushTextureSampler;
    uint debugOverflow;
    float strokeBleedExp;
    ComputeTestParams* ptr;
};
/*
struct PushConstants {
    float4x4 matrix;
    float width;
    float* pointer;
};
*/

struct GeometryData {
    SceneParams* sceneParams;
    uint* indices;
    // Attribute buffers
    float3* position;
    float3* normal;
    float2* texcoord;
    float4* color;

    // Object to world transform
    float4x4 modelMatrix;
};