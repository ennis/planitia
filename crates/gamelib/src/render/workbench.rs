use crate::asset::{Handle, VfsPath};
use crate::render::load_shader_archive;


pub struct Workbench {
    /// Shader archives containing the workbench manifest.
    archive: Handle<sharc::ShaderArchive>,
}

impl Workbench {

    /// Loads a workbench manifest from a shader archive.
    pub fn load(shader_archive_path: impl AsRef<VfsPath>) -> Workbench {
        let archive = load_shader_archive(shader_archive_path);
        Workbench {
            archive,
        }
    }

    /// Provides external data to the workbench, such as geometry or textures.
    pub fn set_buffer<T: Copy+'static>(&mut self, name: &str, buffer: gpu::Buffer<T>) {

    }

    /// Sets a render target.
    pub fn set_render_target(&mut self, name: &str, target: gpu::Image) {

    }

    /// Sets a parameter value.
    pub fn set_param<T: Copy + 'static>(&mut self, name: &str, value: &T) {

    }

    pub fn dispatch_compute(&mut self, pass_name: &str) {



    }

    pub fn draw(&mut self, pass_name: &str) {

    }
}


fn find_param(a: &sharc::ShaderArchive, pass: &sharc::Pass, name: &str) -> Option<sharc::reflection::Param> {
    let signature = &a[pass.signature];
    for param in &a[signature.params] {
        if &a[param.name] == name {
            return Some(*param);
        }
    }
    None
}

/*
fn find_pass(a: &sharc::ShaderArchive, name: &str) -> Option<sharc::Pass> {
    for pass in &a[a.root().passes] {
        if pass.name.as_str() == name {
            return Some(*pass);
        }
    }
    None
}
*/