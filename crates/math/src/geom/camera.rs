use std::f32::consts::PI;
use glam::{Vec3Swizzles, Vec4Swizzles};
use crate::Mat4;
use crate::{DVec2, DVec3, Vec3, dvec2, dvec3, vec3};

#[derive(Copy, Clone, Debug, Default)]
pub struct Frustum {
    pub left: f32,
    pub right: f32,
    pub top: f32,
    pub bottom: f32,
    // near clip plane position
    pub near_plane: f32,
    // far clip plane position
    pub far_plane: f32,
}

/// Represents a camera (a view of a scene) and the screen size in pixels.
#[derive(Copy, Clone, Debug)]
pub struct Camera {
    // Projection parameters
    // frustum (for culling)
    pub frustum: Frustum,
    // view matrix
    // (World -> View)
    pub view: Mat4,
    pub view_inverse: Mat4,
    // projection matrix
    // (View -> clip?)
    pub projection: Mat4,
    pub projection_inverse: Mat4,
    pub screen_size: DVec2,
}

impl Camera {
    pub fn view_projection(&self) -> Mat4 {
        self.projection * self.view
    }

    pub fn screen_to_ndc(&self, screen_pos: DVec3) -> DVec3 {
        // Note: Vulkan NDC space (depth 0->1) is different from OpenGL  (-1 -> 1)
        self.screen_to_ndc_2d(screen_pos.xy()).extend(screen_pos.z)
    }

    pub fn screen_to_ndc_2d(&self, screen_pos: DVec2) -> DVec2 {
        dvec2(
            2.0 * screen_pos.x / self.screen_size.x - 1.0,
            1.0 - 2.0 * screen_pos.y / self.screen_size.y,
        )
    }

    /// Unprojects a screen-space position to a view-space ray direction.
    ///
    /// This assumes a normalized depth range of `[0, 1]`.
    pub fn screen_to_view(&self, screen_pos: DVec3) -> DVec3 {
        // Undo viewport transformation
        let ndc = self.screen_to_ndc(screen_pos).as_vec3();
        // TODO matrix ops as f64?
        let inv_proj = self.projection.inverse();
        let clip = inv_proj * ndc.extend(1.0);
        (clip.xyz() / clip.w).as_dvec3()
    }

    /// Unprojects a screen-space position to a view-space ray direction.
    ///
    /// This assumes a normalized depth range of `[0, 1]`.
    pub fn screen_to_view_dir(&self, screen_pos: DVec2) -> DVec3 {
        self.screen_to_view(dvec3(screen_pos.x, screen_pos.y, 0.0)).normalize()
    }

    pub fn screen_to_world(&self, screen_pos: DVec3) -> DVec3 {
        let view_pos = self.screen_to_view(screen_pos).as_vec3();
        let world_pos = self.view.inverse() * view_pos.extend(1.0);
        world_pos.xyz().as_dvec3()
    }

    pub fn eye(&self) -> DVec3 {
        self.view_inverse.transform_point3(Vec3::ZERO).as_dvec3()
    }

    pub fn screen_to_world_ray(&self, screen_pos: DVec2) -> (DVec3, DVec3) {
        let world_pos = self.screen_to_world(screen_pos.extend(0.0));
        let eye_pos = self.view_inverse.transform_point3(Vec3::ZERO).as_dvec3();
        (eye_pos, (world_pos - eye_pos).normalize())
    }

    pub fn world_to_screen(&self, world_pos: DVec3) -> DVec3 {
        let view_pos = self.view * world_pos.extend(1.0).as_vec4();
        let clip_pos = self.projection * view_pos;
        let ndc = clip_pos.xyz() / clip_pos.w;
        let ndc = ndc.as_dvec3();
        dvec3(
            0.5 * (ndc.x + 1.0) * self.screen_size.x,
            0.5 * (1.0 - ndc.y) * self.screen_size.y,
            ndc.z,
        )
    }

    pub fn world_to_screen_line(&self, a: DVec3, b: DVec3) -> (DVec3, DVec3) {
        (self.world_to_screen(a), self.world_to_screen(b))
    }
}

impl Default for Camera {
    fn default() -> Self {
        let view = Mat4::look_at_rh(vec3(0.0, 0.0, -1.0), vec3(0.0, 0.0, 0.0), Vec3::Y);
        let view_inverse = view.inverse();
        let projection = Mat4::perspective_rh(PI / 2.0, 1.0, 0.01, 10.0);
        let projection_inverse = projection.inverse();

        Camera {
            // TODO
            frustum: Default::default(),
            view,
            view_inverse,
            projection,
            projection_inverse,
            screen_size: Default::default(),
        }
    }
}