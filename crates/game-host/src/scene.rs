use std::ops::Deref;
use gpu::Ptr;
use math::{Camera, Mat4, Vec2, Vec3};

#[repr(C)]
#[derive(Clone, Copy)]
pub struct SceneInfoUniforms {
    pub view_matrix: Mat4,
    pub proj_matrix: Mat4,
    pub view_proj_matrix: Mat4,
    pub screen_size: Vec2,
    pub time: f32,
    pub frame: u32,
    pub eye: Vec3,
}

/// Scene info and GPU buffer containing it.
pub struct SceneInfo {
    pub info: SceneInfoUniforms,
    pub gpu: Ptr<SceneInfoUniforms>,
}

impl SceneInfo {
    pub fn new(camera: &Camera, time: f32, frame: u32) -> Self {
        let uniforms = SceneInfoUniforms {
            view_matrix: camera.view,
            proj_matrix: camera.projection,
            view_proj_matrix: camera.view_projection(),
            screen_size: camera.screen_size.as_vec2(),
            time,
            frame,
            eye: camera.eye().as_vec3(),
        };
        let gpu_ptr = gpu::alloc_temp(&uniforms);
        Self { info: uniforms, gpu: gpu_ptr }
    }
}

impl Deref for SceneInfo {
    type Target = SceneInfoUniforms;

    fn deref(&self) -> &Self::Target {
        &self.info
    }
}