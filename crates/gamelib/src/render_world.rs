/*use crate::asset::{AssetCache, FileWatcher, VfsPath, VfsPathBuf};
use crate::component::MainThreadComponent;
use crate::pipeline_cache::{create_compute_pipeline_from_archive, create_graphics_pipeline_from_archive};
use gpu::{ComputePipeline, GraphicsPipeline, vk};
use log::error;
use shader_archive::{ImageResourceSize, ShaderArchive};
use std::collections::HashMap;
struct ImageResource {
    image: gpu::Image,
    /// Whether this image is used as a render target, and should be resized as the window resizes.
    render_target: bool,
}

/// Holds data used by the rendering system: GPU resources (render targets, shader pipelines, etc.).
///
/// This is a singleton; use `RenderWorld::instance()` to get a reference to it.
pub struct RenderWorld {
    archive_path: VfsPathBuf,
    graphics_pipelines: HashMap<String, GraphicsPipeline>,
    compute_pipelines: HashMap<String, ComputePipeline>,
    images: HashMap<String, ImageResource>,
    watch: FileWatcher,
    screen_width: u32,
    screen_height: u32,
}

impl Default for RenderWorld {
    fn default() -> Self {
        RenderWorld {
            archive_path: VfsPathBuf::default(),
            graphics_pipelines: Default::default(),
            compute_pipelines: Default::default(),
            images: HashMap::new(),
            watch: FileWatcher::new(|| ()).unwrap(),
            screen_width: 0,
            screen_height: 0,
        }
    }
}

impl RenderWorld {
    fn reload_archive(&mut self) {
        let asset_cache = AssetCache::instance();
        let archive = asset_cache.load_and_insert(self.archive_path.as_ref(), |data, _, _| {
            let archive = ShaderArchive::from_bytes(data)?;
            Ok(archive)
        });

        self.watch = FileWatcher::new(|| RENDER_WORLD.borrow_mut().reload_archive()).unwrap();

        #[cfg(feature = "hot_reload")]
        if let Ok(archive) = archive.read() {
            let archive = &*archive;

            // watch dependencies
            for dep in archive.dependencies() {
                self.watch.watch_file(&archive[dep.path]);
            }

            self.reload_images(archive);
            self.reload_pipelines(archive);
        }
    }

    /// Configures the render world with the specified shader archive.
    fn set_archive(&mut self, path: impl AsRef<VfsPath>) {
        self.archive_path = path.as_ref().to_path_buf();
        self.reload_archive();
    }

    fn reload_images(&mut self, archive: &ShaderArchive) {
        self.images.clear();
        let root = archive.root();
        for img in &archive[root.images] {
            let (width, height) = match img.size {
                ImageResourceSize::RenderTarget => (self.screen_width, self.screen_height),
                ImageResourceSize::Fixed { width, height } => (width, height),
                ImageResourceSize::Dynamic => continue,
            };

            let image = gpu::Image::new(gpu::ImageCreateInfo {
                memory_location: gpu::MemoryLocation::Unknown,
                type_: gpu::ImageType::Image2D,
                usage: img.usage,
                format: img.format,
                width,
                height,
                ..
            });

            self.images.insert(
                img.name.to_string(),
                ImageResource {
                    image,
                    render_target: matches!(img.size, ImageResourceSize::RenderTarget),
                },
            );
        }
    }

    fn reload_pipelines(&mut self, archive: &ShaderArchive) {
        let root = archive.root();
        for p in &archive[root.pipelines] {
            match p.kind {
                shader_archive::PipelineKind::Graphics(ref data) => {
                    match create_graphics_pipeline_from_archive(archive, data) {
                        Ok(pipeline) => {
                            self.graphics_pipelines.insert(p.name.to_string(), pipeline);
                        }
                        Err(err) => {
                            error!("Failed to create graphics pipeline `{}`: {}", p.name, err);
                        }
                    }
                }
                shader_archive::PipelineKind::Compute(ref data) => {
                    match create_compute_pipeline_from_archive(archive, data) {
                        Ok(pipeline) => {
                            self.compute_pipelines.insert(p.name.to_string(), pipeline);
                        }
                        Err(err) => {
                            error!("Failed to create compute pipeline `{}`: {}", p.name, err);
                        }
                    }
                }
            }
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.screen_width = width;
        self.screen_height = height;
        for (_, res) in self.images.iter_mut() {
            if res.render_target {
                res.image = gpu::Image::new(gpu::ImageCreateInfo {
                    memory_location: gpu::MemoryLocation::Unknown,
                    type_: gpu::ImageType::Image2D,
                    usage: res.image.usage(),
                    format: res.image.format(),
                    width,
                    height,
                    ..
                });
            }
        }
    }
}

static RENDER_WORLD: MainThreadComponent<RenderWorld> = MainThreadComponent::new();
*/