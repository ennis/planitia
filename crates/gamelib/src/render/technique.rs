use crate::asset::{AssetCache, DefaultLoader, Dependencies, FileMetadata, Handle, LoadResult, Provider, VfsPath};
use crate::render::load_shader_archive;

/// Collection of graphics and compute passes describing a rendering pipeline.
pub struct Technique {
    name: String,
    file: Handle<sharc::ShaderArchive>,
}

/*
impl DefaultLoader for Technique {
    fn load(path: &VfsPath, metadata: &FileMetadata, provider: &dyn Provider, dependencies: &mut Dependencies) -> LoadResult<Self>
    {
        let file = load_shader_archive(path.path_without_fragment());
        dependencies.add(&file);



        Ok(())
    }
}*/

/// Resources allocated for a rendering technique.
pub struct TechniqueInstance {
    tech: Handle<Technique>,
}