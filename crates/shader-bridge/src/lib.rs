mod compiler;
mod embed_shaders;
mod error;
mod library;
mod rustfmt;
mod session;
mod syntax_bindgen;

/// The profile with which to compile the shaders.
///
/// See the [slang documentation](https://github.com/shader-slang/slang/blob/master/docs/command-line-slangc-reference.md#-profile) for a list of available profiles.
// Not sure if the profile matters in practice, it seems that slang will
// add the required SPIR-V extensions depending on what is used in the shader.
pub const SHADER_PROFILE: &str = "glsl_460";

pub use embed_shaders::compile_and_embed_shaders;
pub use library::{ShaderEntryPointInfo, ShaderLibrary, ShaderLibraryLoadOptions};
pub use rustfmt::rustfmt_file;
pub use syntax_bindgen::translate_slang_shared_decls;
pub use error::Error;
