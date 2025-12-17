use crate::input::{InputEvent, MouseScrollDelta, PointerButton};
use log::debug;
use math::geom::{Box3D, Camera, Frustum};
use math::{DQuat, DVec2, DVec3, Mat4, Vec3, Vec3Swizzles, Vec4Swizzles, dvec2, dvec3, vec3};
use std::cell::Cell;
use std::f32::consts::PI;
use std::f64::consts::TAU;

#[derive(Copy, Clone, Debug)]
struct CameraFrame {
    eye: DVec3,
    up: DVec3,
    center: DVec3,
}

/// Camera movement mode.
#[derive(Copy, Clone, Debug)]
enum Mode {
    None,
    /// Panning motion
    Pan {
        /// Where on the screen the pan started
        anchor_screen: DVec2,
        orig_frame: CameraFrame,
    },
    /// Orbiting motion
    Tumble {
        anchor_screen: DVec2,
        orig_frame: CameraFrame,
    },
}

/// Camera controller state.
#[derive(Clone, Debug)]
pub struct CameraControl {
    fov_y_radians: f64,
    z_near: f64,
    z_far: f64,
    zoom: f32,
    screen_size: DVec2,
    cursor_pos: Option<DVec2>,
    frame: CameraFrame,
    input_mode: Mode,
    last_cam: Cell<Option<Camera>>,
}

#[derive(Copy, Clone, Debug)]
pub enum CameraControlInput {
    MouseInput { button: PointerButton, pressed: bool },
    CursorMoved { position: DVec2 },
}

impl Default for CameraControl {
    fn default() -> Self {
        Self::new(1280, 720)
    }
}

impl CameraControl {
    /// Creates the camera controller state.
    ///
    /// # Arguments
    /// - `width` initial width of the screen in physical pixels
    /// - `height` initial height of the screen in physical pixels
    pub fn new(screen_width: u32, screen_height: u32) -> CameraControl {
        CameraControl {
            fov_y_radians: std::f64::consts::PI / 10.0,
            z_near: 0.1,
            z_far: 100.0,
            zoom: 1.0,
            screen_size: dvec2(screen_width as f64, screen_height as f64),
            cursor_pos: None,
            frame: CameraFrame {
                eye: dvec3(0.0, 0.0, 10.0),
                up: dvec3(0.0, 1.0, 0.0),
                center: dvec3(0.0, 0.0, 0.0),
            },
            input_mode: Mode::None,
            last_cam: Cell::new(None),
        }
    }

    /// Call when the size of the screen changes.
    pub fn resize(&mut self, width: u32, height: u32) {
        self.screen_size = dvec2(width as f64, height as f64);
        self.last_cam.set(None);
    }

    /// Returns the current eye position.
    pub fn eye(&self) -> DVec3 {
        self.frame.eye
    }

    fn handle_pan(&mut self, orig: &CameraFrame, delta_screen: DVec2) {
        let delta = delta_screen / self.screen_size;
        let dir = orig.center - orig.eye;
        let right = dir.normalize().cross(orig.up);
        let dist = dir.length();
        self.frame.eye = orig.eye + dist * (-delta.x * right + delta.y * orig.up);
        self.frame.center = orig.center + dist * (-delta.x * right + delta.y * orig.up);
        self.last_cam.set(None);
    }

    fn to_ndc(&self, p: DVec2) -> DVec2 {
        2.0 * (p / self.screen_size) - dvec2(1.0, 1.0)
    }

    fn handle_tumble(&mut self, orig: &CameraFrame, from: DVec2, to: DVec2) {
        let delta = (to - from) / self.screen_size;
        let eye_dir = orig.eye - orig.center;
        let right = eye_dir.normalize().cross(orig.up);
        let r = DQuat::from_rotation_y(-delta.x * TAU) * DQuat::from_axis_angle(right, delta.y * TAU);
        let new_eye = orig.center + r * eye_dir;
        let new_up = r * orig.up;
        self.frame.eye = new_eye;
        self.frame.up = new_up;
        self.last_cam.set(None);
    }

    pub fn handle_input(&mut self, input: &InputEvent) -> bool {
        match input {
            InputEvent::CursorMoved { x, y } => self.cursor_moved(dvec2(*x as f64, *y as f64)),
            InputEvent::PointerDown { button, .. } => self.mouse_input(*button, true),
            InputEvent::PointerUp { button, .. } => self.mouse_input(*button, false),
            // TODO map lines to pixels properly
            InputEvent::MouseWheel(MouseScrollDelta::LineDelta { y, .. }) => self.mouse_wheel(*y as f64),
            InputEvent::MouseWheel(MouseScrollDelta::PixelDelta { y, .. }) => self.mouse_wheel(*y as f64),
            _ => false,
        }
    }

    /// Call when receiving mouse button input.
    ///
    /// Returns whether the event was handled by the camera controller.
    fn mouse_input(&mut self, button: PointerButton, pressed: bool) -> bool {
        let mut handled = false;
        match button {
            PointerButton::MIDDLE => {
                if let Some(pos) = self.cursor_pos {
                    handled = true;
                    match self.input_mode {
                        Mode::None | Mode::Pan { .. } if pressed => {
                            self.input_mode = Mode::Pan {
                                anchor_screen: pos,
                                orig_frame: self.frame,
                            };
                        }
                        Mode::Pan {
                            orig_frame,
                            anchor_screen,
                        } if !pressed => {
                            self.handle_pan(&orig_frame, pos - anchor_screen);
                            self.input_mode = Mode::None;
                        }
                        _ => {}
                    }
                }
            }
            PointerButton::LEFT => {
                if let Some(pos) = self.cursor_pos {
                    handled = true;
                    match self.input_mode {
                        Mode::None | Mode::Tumble { .. } if pressed => {
                            self.input_mode = Mode::Tumble {
                                anchor_screen: pos,
                                orig_frame: self.frame,
                            };
                        }
                        Mode::Tumble {
                            orig_frame,
                            anchor_screen,
                        } if !pressed => {
                            self.handle_tumble(&orig_frame, anchor_screen, pos);
                            self.input_mode = Mode::None;
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
        handled
    }

    /// Returns whether the event was handled by the camera controller.
    fn mouse_wheel(&mut self, delta: f64) -> bool {
        // TODO orthographic projection
        let delta = - delta / 120.0;
        self.frame.eye = self.frame.center + (1.0 + delta) * (self.frame.eye - self.frame.center);
        self.last_cam.set(None);
        true
    }

    /// Call when receiving cursor moved input.
    ///
    /// Returns whether the event was handled by the camera controller.
    fn cursor_moved(&mut self, position: DVec2) -> bool {
        self.cursor_pos = Some(position);
        match self.input_mode {
            Mode::Tumble {
                orig_frame,
                anchor_screen,
            } => {
                self.handle_tumble(&orig_frame, anchor_screen, position);
                true
            }
            Mode::Pan {
                orig_frame,
                anchor_screen,
            } => {
                self.handle_pan(&orig_frame, position - anchor_screen);
                true
            }
            _ => false,
        }
    }

    /// Centers the camera on the given axis-aligned bounding box.
    /// Orbit angles are reset.
    pub fn center_on_bounds(&mut self, bounds: &Box3D, fov_y_radians: f64) {
        let size = bounds.size().max_element() as f64;
        let new_center: DVec3 = bounds.center().as_dvec3();
        let cam_dist = (0.5 * size) / f64::tan(0.5 * fov_y_radians);

        let new_front = dvec3(0.0, 0.0, -1.0).normalize();
        let new_eye = new_center + (-new_front * cam_dist);

        let new_right = new_front.cross(self.frame.up);
        let new_up = new_right.cross(new_front);

        self.frame.center = new_center;
        self.frame.eye = new_eye;
        self.frame.up = new_up;

        self.z_near = 0.1 * cam_dist;
        self.z_far = 10.0 * cam_dist;
        self.fov_y_radians = fov_y_radians;
        self.last_cam.set(None);

        debug!(
            "center_on_bounds: eye={}, center={}, z_near={}, z_far={}",
            self.frame.eye, self.frame.center, self.z_near, self.z_far
        );
    }

    /// Returns the look-at matrix
    fn look_at(&self) -> Mat4 {
        Mat4::look_at_rh(
            self.frame.eye.as_vec3(),
            self.frame.center.as_vec3(),
            self.frame.up.as_vec3(),
        )
    }

    /// Returns a `Camera` for the current viewpoint.
    pub fn camera(&self) -> Camera {
        if let Some(cam) = self.last_cam.get() {
            return cam;
        }
        let aspect_ratio = self.screen_size.x / self.screen_size.y;
        let view = self.look_at();
        let view_inverse = view.inverse();

        let flip_y = Mat4::from_scale(Vec3::new(1.0, -1.0, 1.0));
        let projection = Mat4::perspective_rh(
            self.fov_y_radians as f32,
            aspect_ratio as f32,
            self.z_near as f32,
            self.z_far as f32,
        ) * flip_y;
        let projection_inverse = projection.inverse();
        let cam = Camera {
            frustum: Frustum {
                left: -aspect_ratio as f32,
                right: aspect_ratio as f32,
                top: 1.0,
                bottom: -1.0,
                near_plane: self.z_near as f32,
                far_plane: self.z_far as f32,
            },
            view,
            view_inverse,
            projection,
            projection_inverse,
            screen_size: self.screen_size,
        };
        self.last_cam.set(Some(cam));
        cam
    }
}
