use crate::Game;
use crate::gui::Gui;
use crate::shader::Shader;
use anyhow::{Context, Result, anyhow};
use math::point_line_dist;
use serde::{Deserialize, Serialize};
use shadertool::{BuildManifest, BuildOptions};
use std::ffi::CString;
use std::path::{Path, PathBuf};

#[derive(Serialize, Deserialize)]
pub struct Geometry {
    pub path: PathBuf,
    #[serde(skip)]
    pub geo: Option<hgeo::Geo>,
}

#[derive(Serialize, Deserialize)]
#[repr(u8)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum OptimizationLevel {
    None = 0,
    Default = 1,
    High = 2,
    Maximal = 3,
}

#[derive(Serialize, Deserialize)]
pub struct ShaderOptions {
    pub profile: String,
    pub debug: bool,
    pub defines: Vec<(String, String)>,
    pub optimization_level: OptimizationLevel,
}

pub const DEFAULT_SHADER_PROFILE: &str = "glsl_460";

impl Default for ShaderOptions {
    fn default() -> Self {
        Self {
            profile: DEFAULT_SHADER_PROFILE.to_string(),
            debug: false,
            defines: Vec::new(),
            optimization_level: OptimizationLevel::Default,
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct ImageResource {
    pub name: String,
    pub ty: gpu::ImageType,
    pub format: gpu::Format,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub mip_levels: u32,
    pub array_layers: u32,
    #[serde(skip)]
    pub image: Option<gpu::Image>,
}

#[derive(Serialize, Deserialize)]
pub struct BufferResource {
    pub name: String,
    pub size: usize,
    #[serde(skip)]
    pub buffer: Option<gpu::BufferUntyped>,
}

#[derive(Serialize, Deserialize)]
pub struct Project {
    /// Project save location.
    #[serde(skip)]
    pub path: PathBuf,
    pub geometry: Vec<Geometry>,
    pub shaders: Vec<Shader>,
    pub images: Vec<ImageResource>,
}

impl Default for Project {
    fn default() -> Self {
        Self { path: PathBuf::new(), geometry: Vec::new(), shaders: vec![], images: vec![] }
    }
}

impl Project {
    /// Saves this project configuration to the specified file.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let ron = ron::to_string(self)?;
        std::fs::write(path, ron)?;
        Ok(())
    }

    /// Loads a project from a saved config file.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let ron = std::fs::read_to_string(path)?;
        let project: Project = ron::from_str(&ron)?;
        Ok(project)
    }

    /// Converts all paths in the project to be relative to the specified base path.
    pub fn make_paths_relative(&mut self) {
        let base_path = self.path.parent().unwrap();
        for g in self.geometry.iter_mut() {
            let abspath = Self::resolve_path(&self.path, &g.path);
            g.path = pathdiff::diff_paths(&abspath, base_path).unwrap_or(abspath);
        }
        for s in self.shaders.iter_mut() {
            let abspath = Self::resolve_path(&self.path, &s.path);
            s.path = pathdiff::diff_paths(&abspath, base_path).unwrap_or(abspath);
        }
    }

    /// Resolves a path relative to the project file location.
    pub fn resolve_path(base: &Path, path: &Path) -> PathBuf {
        if path.is_absolute() { path.to_path_buf() } else { base.parent().unwrap_or_else(|| Path::new("")).join(path) }
    }
}

impl Game {
    pub fn open_project(&mut self) {
        let path = gamelib::pick_file("Project File", &["ron"]);
        let Some(path) = path else {
            info!("user cancelled open project");
            return;
        };
        match Project::load(&path) {
            Ok(project) => {
                self.project = project;
                self.project.path = path;
                self.reload_geometry();
                self.reload_shaders(true);
            }
            Err(err) => {
                self.gui.report_error("Failed to open project", err);
            }
        }
    }

    pub fn save_project(&mut self, save_as: bool) {
        if save_as || self.project.path.is_empty() {
            let path = gamelib::save_file("Project File", &["ron"]);
            let Some(path) = path else {
                info!("user cancelled save");
                return;
            };
            self.project.path = path;
            self.project.make_paths_relative();
        }
        if let Err(err) = self.project.save(&self.project.path) {
            self.gui.report_error("Failed to save project", err);
        }
    }

    pub fn add_shader(&mut self) {
        let path = gamelib::pick_file("Shader File", &["slang"]);
        let Some(path) = path else {
            return;
        };
        let mut shader = Shader::new(path.clone(), ShaderOptions::default());
        if let Err(err) = shader.compile() {
            self.gui.report_error(format!("Failed to compile shader {}", path.display()), err);
        }
        self.project.shaders.push(shader);
    }

    /// Loads a houdini geometry file or prints an error to the console.
    fn load_geometry(path: &Path, gui: &mut Gui) -> Option<hgeo::Geo> {
        match hgeo::Geo::load(path) {
            Ok(geo) => Some(geo),
            Err(err) => {
                gui.report_error(format!("Failed to load geometry from {}", path.display()), err.into());
                None
            }
        }
    }

    pub fn add_geometry(&mut self) {
        let path = gamelib::pick_file("Houdini Geometry File", &["geo", "bgeo"]);
        let Some(path) = path else {
            return;
        };
        if let Some(geo) = Self::load_geometry(&path, &mut self.gui) {
            self.project.geometry.push(Geometry { path: path.clone(), geo: Some(geo) });
        }
    }

    /// Reloads geometry files.
    fn reload_geometry(&mut self) {
        let parent_path = self.project.path.parent().unwrap();
        for g in self.project.geometry.iter_mut() {
            if g.geo.is_none() {
                if g.path.is_relative() {
                    g.path = parent_path.join(&g.path);
                }
                g.geo = Self::load_geometry(&g.path, &mut self.gui);
            }
        }
    }

    /// Reloads shader files.
    fn reload_shaders(&mut self, force_reload: bool) {
        //for sh in
    }

    /// Adds a new (empty) image resource to the project. The resulting resource can be
    /// configured (name, format, dimensions, ...) from the "Images" GUI window.
    pub fn add_image(&mut self) {
        self.project.images.push(ImageResource {
            name: format!("image_{}", self.project.images.len()),
            ty: gpu::ImageType::Image2D,
            format: gpu::vulkan::VK_FORMAT_R8G8B8A8_UNORM,
            width: 256,
            height: 256,
            depth: 1,
            mip_levels: 1,
            array_layers: 1,
            image: None,
        });
    }
}
