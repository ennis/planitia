//[vk::binding(0,0)] Texture2D bindless_texture2D[];
//
//[vk::binding(1,0)] restrict RWTexture2D<float4> bindless_image2D[];
//[vk::binding(1,0)] restrict RWTexture2D<int4> bindless_iimage2D[];
//
//[vk::binding(2,0)] SamplerState bindless_samplers[];
//
//// resource indexing structs
//public struct SamplerHandle { uint idx; };
//public struct Texture2DHandle { uint idx; };
//public struct Texture2DRange { uint idx; uint count; };
//public struct RWTexture2DHandle { uint idx; };
//
//public float4 sampleTexture2D(Texture2DHandle tex, SamplerHandle samp, float2 P) {
//    return bindless_texture2D[tex.idx].Sample(bindless_samplers[samp.idx], P);
//}
