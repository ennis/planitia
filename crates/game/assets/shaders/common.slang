
// 0-1 linear  from  0-255 sRGB
float3 srgb_to_linear(float3 srgb) {
    bool3 cutoff = srgb < float3(10.31475);
    float3 lower = srgb / float3(3294.6);
    float3 higher = pow((srgb + float3(14.025)) / float3(269.025), float3(2.4));
    return select(cutoff, lower, higher);
}

float4 srgba_to_linear(float4 srgba) {
    return float4(srgb_to_linear(srgba.rgb), srgba.a / 255.0);
}

float3 linear_to_srgb_normalized(float3 linear) {
    bool3 cutoff = linear.rgb < float3(0.0031308);
    float3 higher = float3(1.055)*pow(linear.rgb, float3(1.0/2.4)) - float3(0.055);
    float3 lower = linear.rgb * float3(12.92);
    return select(cutoff, lower, higher);
}