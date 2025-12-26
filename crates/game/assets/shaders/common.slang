
/// SRGB8 color
typedef uint8_t4 Srgba8;

/// Remaps RGBA8 (0-255) colors to float4 (0.0-1.0) color
float4 srgba8_to_float4(Srgba8 srgba8) {
    return float4(srgba8 / 255.0);
}

Srgba8 srgba8(uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
    return Srgba8(r, g, b, a);
}

Srgba8 linear_to_srgba8(float4 linear) {
    float3 srgb = linear_to_srgb_normalized(linear.rgb);
    return Srgba8(clamp(float4(srgb, linear.a) * float4(255.0), 0.0, 255.0));
}

/// 0-1 linear  from  0-255 sRGB
float3 srgb_to_linear(float3 srgb) {
    bool3 cutoff = srgb < float3(10.31475);
    float3 lower = srgb / float3(3294.6);
    float3 higher = pow((srgb + float3(14.025)) / float3(269.025), float3(2.4));
    return select(cutoff, lower, higher);
}

/// 0-1 linear  from  0-255 sRGBA
float4 srgba_to_linear(float4 srgba) {
    return float4(srgb_to_linear(srgba.rgb), srgba.a / 255.0);
}

float3 linear_to_srgb_normalized(float3 linear) {
    bool3 cutoff = linear.rgb < float3(0.0031308);
    float3 higher = float3(1.055)*pow(linear.rgb, float3(1.0/2.4)) - float3(0.055);
    float3 lower = linear.rgb * float3(12.92);
    return select(cutoff, lower, higher);
}



/// IQ's palette function
float3 palette(float t, float3 a, float3 b, float3 c, float3 d)
{
    return a + b*cos(6.28318*(c*t+d));
}

/// 2D Distance from a point to a segment.
float distance_to_segment(float2 p, float2 a, float2 b, out float alpha) {
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
float hash(in int2 p)
{
    // 2D -> 1D
    int n = p.x*3 + p.y*113;
    // 1D hash by Hugo Elias
    n = (n << 13) ^ n;
    n = n * (n * n * 15731 + 789221) + 1376312589;
    return -1.0+2.0*float(n & 0x0fffffff)/float(0x0fffffff);
}

float noise(float2 p)
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

float fbm(float2 uv)
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

float4 dither(float4 color, uint2 frag_coord, float time)
{
    color.rgb += float3(0.02 * noise(float2(frag_coord.xy) + 10000 * fmod(time, 10.0)));
    return color;
}

//////////////////////////////////////////////////////////////////
//

struct PushBuffer<T> {
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

struct DrawIndirectCommand {
    uint vertex_count;
    uint instance_count;
    uint first_vertex;
    uint first_instance;
};

struct DrawIndexedIndirectCommand {
    Atomic<uint> index_count;
    uint instance_count;
    uint first_index;
    int  vertex_offset;
    uint first_instance;
};