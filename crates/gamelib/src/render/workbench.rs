use crate::asset::{AssetLoadError, Handle, VfsPath};
use crate::render::load_shader_archive;
use std::io;

pub use crate::worksheet::*;

pub struct Workbench {
    /// Shader archive containing the workbench manifest.
    archive: Handle<sharc::ShaderArchive>,
    module_name: String,
}

impl Workbench {
    /// Loads a workbench manifest from a shader archive.
    pub fn load(shader_module_path: impl AsRef<VfsPath>) -> Result<Workbench, AssetLoadError> {
        let shader_module_path = shader_module_path.as_ref();
        let module_name = shader_module_path
            .fragment()
            .expect("shader module path must contain a fragment specifying the module name");
        let archive = load_shader_archive(shader_module_path.path_without_fragment());


        {
            let archive_ref = &*archive.read()?;
            let (_,module) = archive_ref
                .find_module_with_index(module_name)
                .ok_or(AssetLoadError::IoError(io::Error::new(
                    io::ErrorKind::NotFound,
                    format!(
                        "module `{}` not found in shader archive `{}`",
                        module_name,
                        shader_module_path.path_without_fragment().as_str()
                    ),
                )))?;
            Self::load_module(archive_ref, module);
        }

        Ok(Workbench {
            archive,
            module_name: module_name.to_string(),
            //module_index
        })
    }

    fn load_module(archive: &sharc::ShaderArchive, module: &sharc::Module) {
        // allocate memory (buffers) for the parameters
        // initialize to zero
        // allocate render targets
    }

    /// Provides external data to the workbench, such as geometry or textures.
    pub fn set_buffer<T: Copy + 'static>(&mut self, name: &str, buffer: gpu::Buffer<T>) {}

    /// Sets a render target.
    pub fn set_render_target(&mut self, name: &str, target: gpu::Image) {}

    /// Sets a parameter value.
    pub fn set_param<T: Copy + 'static>(&mut self, name: &str, value: &T) {}

    pub fn dispatch_compute(&mut self, pass_name: &str) {}

    pub fn draw(&mut self, pass_name: &str) {}


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
