//! This application's shaders and related interface types.
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use gpu::DeviceAddress;
use std::marker::PhantomData;

// Define type aliases for slang types. These are referenced in the generated bindings, which
// are just a syntactical translation of the slang declarations to Rust.
//
// WARNING: these must match the layout of the corresponding slang types in the shaders.
//          Notably, the `Texture*_Handle` types must have the same layout as `[u32;2]`
//          to match slang.
type Pointer<T> = DeviceAddress<T>;
type uint = u32;
type int = i32;
type float = f32;
//type bool = u32;
type float2 = math::Vec2;
type float3 = math::Vec3;
type float4 = [f32; 4];
type uint2 = math::UVec2;
type uint3 = math::UVec3;
type uint4 = math::UVec4;
type uint8_t4 = [u8; 4];
type uint8_t = u8;
type int2 = math::IVec2;
type int3 = math::IVec3;
type int4 = math::IVec4;
type float4x4 = [[f32; 4]; 4];

#[repr(C)]
#[derive(Copy, Clone)]
pub struct Texture2D_Handle<T> {
    handle: gpu::TextureHandle,
    _phantom: PhantomData<fn() -> T>,
}

impl<T> From<gpu::TextureHandle> for Texture2D_Handle<T> {
    fn from(handle: gpu::TextureHandle) -> Self {
        Texture2D_Handle {
            handle,
            _phantom: PhantomData,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct RWTexture2D_Handle<T> {
    handle: gpu::TextureHandle,
    _phantom: PhantomData<fn() -> T>,
}

impl<T> From<gpu::TextureHandle> for RWTexture2D_Handle<T> {
    fn from(handle: gpu::TextureHandle) -> Self {
        RWTexture2D_Handle {
            handle,
            _phantom: PhantomData,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct SamplerState_Handle {
    handle: gpu::SamplerHandle,
}

impl From<gpu::SamplerHandle> for SamplerState_Handle {
    fn from(handle: gpu::SamplerHandle) -> Self {
        SamplerState_Handle { handle }
    }
}

// include generated bindings by the build script
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
