use gamelib::render::RenderTarget;
use gpu::vulkan::{
    VK_FILTER_LINEAR, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R16_UNORM, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
    VK_SAMPLER_MIPMAP_MODE_LINEAR,
};
use gpu::{ClearColorValue, Image, ImageAspect, ImageUsage, SamplerHandle, BARRIER_TEXTURE};
use image::ImageReader;
use math::{Camera, Mat4, vec4};
use std::io;
use std::path::Path;

pub struct Terrain {
    heightmap: Image,
    sampler: SamplerHandle,
    render_target: RenderTarget,
}

#[gpu::shader_module("shaders/terrain2.slang#1")]
mod shaders {}

const METER_PER_PX: f32 = 2.5;

impl Terrain {
    pub fn new<P: AsRef<Path>>(heightmap_image_file: P) -> anyhow::Result<Self> {
        let sampler = gpu::register_sampler(&gpu::SamplerParams {
            min_filter: VK_FILTER_LINEAR,
            mag_filter: VK_FILTER_LINEAR,
            mipmap_mode: VK_SAMPLER_MIPMAP_MODE_LINEAR,
            address_mode_u: VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            address_mode_v: VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            address_mode_w: VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            ..
        });
        let heightmap = load_heightmap_image(heightmap_image_file.as_ref())?;
        Ok(Terrain {
            heightmap,
            sampler,
            render_target: RenderTarget::new(
                VK_FORMAT_R8G8B8A8_UNORM,
                ImageUsage::STORAGE | ImageUsage::SAMPLED | ImageUsage::COLOR_ATTACHMENT,
            ),
        })
    }

    pub fn render(&mut self, camera: &Camera, image: &gpu::Image) {
        self.render_target.setup(image.width(), image.height());

        let depth_divisions = 1024;

        // matrix that converts from terrain UV to NDC pos

        // 0;1 => -1;1
        let scale_x = self.heightmap.width() as f32 * METER_PER_PX;
        let scale_z = self.heightmap.height() as f32 * METER_PER_PX;
        let terrain_uv_to_world = Mat4::from_cols_array_2d(&[
            [scale_x, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, scale_z, 0.0],
            [-0.5 * scale_x, 0.0, -0.5 * scale_z, 1.0],
        ]);

        let terrain_matrix = camera.projection * camera.view * terrain_uv_to_world;
        let inv_terrain_matrix = terrain_uv_to_world.inverse() * camera.view_inverse * camera.projection_inverse;


        let params = Params {
            heightmap: self.heightmap.texture_handle(),
            sampler: self.sampler,
            output: self.render_target.storage_handle(),
            screen_width: image.width(),
            screen_height: image.height(),
            inv_terrain_matrix,
            view_matrix: camera.view,
            inv_view_matrix: camera.view_inverse,
            proj_matrix: camera.projection,
            inv_proj_matrix: camera.projection_inverse,
            eye_pos: camera.eye().as_vec3(),
            depth_size: depth_divisions,
            terrain_matrix,
        };

        // 1 wg per vertical line
        let wg_count = image.width();
        gpu::clear_image(self.render_target.image(), ClearColorValue::Float([0.0, 0.0, 0.0, 1.0]));
        gpu::dispatch(&shaders::main, wg_count, 1, 1, &params);
        gpu::barrier(BARRIER_TEXTURE);
        gpu::blit_full_image_top_mip_level(self.render_target.image(), image);
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
struct Params {
    /*Texture2D<float>.Handle heightmap;
    Sampler.Handle sampler;
    RWTexture2D<float4>.Handle output;
    int screen_width;
    int screen_height;
    float4x4 view_matrix;
    float4x4 inv_view_matrix;
    float4x4 proj_matrix;
    float4x4 inv_proj_matrix;
    float4x4 inv_uv;
    float3 eye_pos;*/
    heightmap: gpu::TextureHandle,
    sampler: gpu::SamplerHandle,
    output: gpu::StorageImageHandle,
    screen_width: u32,
    screen_height: u32,
    view_matrix: math::Mat4,
    inv_view_matrix: math::Mat4,
    proj_matrix: math::Mat4,
    inv_proj_matrix: math::Mat4,
    terrain_matrix: math::Mat4,
    inv_terrain_matrix: math::Mat4,
    eye_pos: math::Vec3,
    depth_size: u32,
}

fn load_heightmap_image(heightmap_image_file: &Path) -> anyhow::Result<Image> {
    let (heightmap, width, height) = {
        let reader = ImageReader::open(heightmap_image_file)?;
        let image = reader.decode()?;
        (image.to_luma16(), image.width(), image.height())
    };
    let image = gpu::create_image_with_mip_maps(VK_FORMAT_R16_UNORM, width, height, ImageAspect::All, &heightmap);
    Ok(image)
}
