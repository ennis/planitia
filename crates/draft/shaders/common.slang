public static const float PI = 3.14159265358979323846264338327950288;
public static const float TAU = 6.28318530717958647692528676655900577;
public static const float FRAC_PI_2 = 1.57079632679489661923132169163975144;
public static const float FRAC_PI_4 = 0.785398163397448309615660845819875721;


public struct Vertex {
    public float3 pos;
    public Srgba8 color;
    public uint16_t2 uv;
};

/////////////////////////////////////////////////////////////////////
/// SRGB8 color
public typedef uint8_t4 Srgba8;

/// Remaps RGBA8 (0-255) colors to float4 (0.0-1.0) color
public float4 srgba8_to_float4(Srgba8 srgba8) {
    return float4(srgba8 / 255.0);
}

public Srgba8 srgba8(uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
    return Srgba8(r, g, b, a);
}

public Srgba8 linear_to_srgba8(float4 linear) {
    float3 srgb = linear_to_srgb_normalized(linear.rgb);
    return Srgba8(clamp(float4(srgb, linear.a) * float4(255.0), 0.0, 255.0));
}

/// 0-1 linear  from  0-255 sRGB
public float3 srgb_to_linear(float3 srgb) {
    bool3 cutoff = srgb < float3(10.31475);
    float3 lower = srgb / float3(3294.6);
    float3 higher = pow((srgb + float3(14.025)) / float3(269.025), float3(2.4));
    return select(cutoff, lower, higher);
}

/// 0-1 linear  from  0-255 sRGBA
public float4 srgba_to_linear(float4 srgba) {
    return float4(srgb_to_linear(srgba.rgb), srgba.a / 255.0);
}

public float3 linear_to_srgb_normalized(float3 linear) {
    bool3 cutoff = linear.rgb < float3(0.0031308);
    float3 higher = float3(1.055)*pow(linear.rgb, float3(1.0/2.4)) - float3(0.055);
    float3 lower = linear.rgb * float3(12.92);
    return select(cutoff, lower, higher);
}

/// 2D Distance from a point to a segment.
public float distance_to_segment(float2 p, float2 a, float2 b, out float alpha) {
    float2 ab = b - a;
    float2 ap = p - a;
    float side = sign(cross(float3(ab, 0.0), float3(ap, 0.0)).z);
    float d = dot(p - a, ab) / dot(ab, ab);
    d = clamp(d, 0.0, 1.0);
    float2 p0 = a + d * ab;
    alpha = d;
    return side * distance(p, p0);
}

public T threshold<T: __BuiltinFloatingPointType>(T value, T threshold, T edge) {
    return smoothstep(threshold - edge, threshold + edge, value);
}
