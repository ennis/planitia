public static const float PI = 3.14159265358979323846264338327950288;
public static const float TAU = 6.28318530717958647692528676655900577;
public static const float FRAC_PI_2 = 1.57079632679489661923132169163975144;
public static const float FRAC_PI_4 = 0.785398163397448309615660845819875721;

/////////////////////////////////////////////////////////////////////
// Shadertool attributes

// Marks a shader as belonging to the specified pipeline.
[__AttributeUsage(_AttributeTargets.Function)]
struct passAttribute
{
    string pass;
}

[__AttributeUsage(_AttributeTargets.Function)]
struct ColorAttachment
{
    int index;
    string render_world_name;
}

// Specifies a clear depth for a depth attachment.
//
// This should be applied to fragment shaders only.
[__AttributeUsage(_AttributeTargets.Function)]
struct DepthAttachment
{
    string render_world_name;
}


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

/// IQ's palette function
public float3 palette(float t, float3 a, float3 b, float3 c, float3 d)
{
    return a + b*cos(6.28318*(c*t+d));
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
    //float taper = max(0.0, 80.0 - distance(p,b)) / 80.0;
    return side * distance(p, p0);
}

/// Noise stuff
public float hash(in int2 p)
{
    // 2D -> 1D
    int n = p.x*3 + p.y*113;
    // 1D hash by Hugo Elias
    n = (n << 13) ^ n;
    n = n * (n * n * 15731 + 789221) + 1376312589;
    return -1.0+2.0*float(n & 0x0fffffff)/float(0x0fffffff);
}

public float noise(float2 p)
{
    int2 i = int2(floor(p));
    float2 f = fract(p);
    // quintic interpolant
    float2 u = f*f*f*(f*(f*6.0-15.0)+10.0);
    return lerp(
        lerp(hash(i + int2(0, 0)),
             hash(i + int2(1, 0)),
             u.x),
        lerp(hash(i + int2(0, 1)),
             hash(i + int2(1, 1)),
             u.x),
        u.y);
}

public float fbm(float2 uv)
{
    float f = 0.0;
    uv /= 32.0;
    float2x2 m = float2x2(1.6, 1.2, -1.2, 1.6);
    f  = 0.5000*noise(uv);
    uv = mul(m,uv);
    //f += 0.2500*noise( uv ); uv = m*uv;
    //f += 0.1250*noise( uv ); uv = m*uv;
    //f += 0.0625*noise( uv ); uv = m*uv;
    return f;
}

/// Adds time-varying noise dithering to the color.
public float4 dither(float4 color, uint2 frag_coord, float time)
{
    color.rgb += float3(0.02 * noise(float2(frag_coord.xy) + 10000 * fmod(time, 10.0)));
    return color;
}

public float3 hsv2rgb(float3 c)
{
    float4 K = float4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    float3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * lerp(K.xxx, saturate(p - K.xxx), c.y);
}

public float4 angle_to_color(float angle)
{
    float hue = fmod(angle / (2.0 * PI), 1.0);
    return float4(hsv2rgb(float3(hue, 1.0, 1.0)), 1.0);
}

//////////////////////////////////////////////////////////////////
public struct ScreenQuadVSOut
{
    public float4 pos : SV_Position;
    public float2 uv;
};

/// Vertex shader for full-screen quad
public ScreenQuadVSOut screen_quad_vertex_shader(uint vertex_id)
{
    float2 positions[6] = {
        float2(-1.0, -1.0),
        float2( 1.0, -1.0),
        float2(-1.0,  1.0),
        float2(-1.0,  1.0),
        float2( 1.0, -1.0),
        float2( 1.0,  1.0)
    };

    ScreenQuadVSOut o;
    o.pos = float4(positions[vertex_id], 0.0, 1.0);
    o.uv = o.pos.xy * 0.5 + 0.5;
    return o;
}

//////////////////////////////////////////////////////////////////
//

public struct PushBuffer<T> {
    T* base;
    Atomic<uint> offset;
    uint capacity;

    [mutating]
    void push(T value) {
        let off = offset.add(1);
        if (off < capacity) {
            base[off] = value;
        }
    }

    [mutating]
    uint reserve(uint count) {
        let off = offset.add(count);
        if (off + count <= capacity) {
            return off;
        } else {
            return 0;   // TODO: handle overflow
        }
    }
};

public struct DrawIndirectCommand {
    public uint vertex_count;
    public uint instance_count;
    public uint first_vertex;
    public uint first_instance;
};

public struct DrawIndexedIndirectCommand {
    public uint index_count;
    public uint instance_count;
    public uint first_index;
    public int  vertex_offset;
    public uint first_instance;
};

