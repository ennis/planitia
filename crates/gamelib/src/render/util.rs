use gpu::{Image, ImageCreateInfo, ImageUsage, Size3D};

/// Helper type to manage a 2D render target.
///
/// Use [`RenderTarget::new`] to create a new render target.
/// At the beginning of the render pass, call [`RenderTarget::setup`] to allocate or, if necessary,
/// resize the render target.
pub struct RenderTarget {
    inner: Option<RenderTargetInner>,
    usage: ImageUsage,
    format: gpu::Format,
}

impl RenderTarget {
    /// Creates a new render target with no allocated image.
    pub fn new(format: gpu::Format, usage: ImageUsage) -> Self {
        RenderTarget { inner: None, usage, format }
    }

    /// Allocates or resizes the render target image as needed to match the specified dimensions and format.
    pub fn setup(&mut self, width: u32, height: u32) {
        if let Some(ref mut inner) = self.inner {
            if inner.width != width || inner.height != height {
                inner.image.discard_resize(Size3D::new(width, height, 1));
                inner.width = width;
                inner.height = height;
            }
            return;
        }

        let _ = self.inner.take();

        // allocate a new image with the specified dimensions and format
        let image = Image::new(ImageCreateInfo {
            width,
            height,
            format: self.format,
            usage: self.usage,
            ..
        });

        self.inner = Some(RenderTargetInner {
            image,
            width,
            height,
        });
    }

    /// Returns a reference to the render target image.
    ///
    /// Panics if [`setup`] has not been called yet.
    pub fn image(&self) -> &gpu::Image {
        &self.inner.as_ref().expect("RenderTarget not initialized").image
    }

    /// Returns a texture descriptor handle for the image.
    ///
    /// Panics if [`setup`] has not been called yet.
    pub fn texture_handle(&self) -> gpu::TextureHandle {
        self.image().texture_handle()
    }

    /// Returns a storage descriptor handle for the image.
    ///
    /// Panics if [`setup`] has not been called yet.
    pub fn storage_handle(&self) -> gpu::StorageImageHandle {
        self.image().storage_handle()
    }
}

struct RenderTargetInner {
    image: gpu::Image,
    width: u32,
    height: u32,
}

/*
/// Utility function to allocate a suitable render target image for the specified pipeline.
///
/// The format of the image is determined from the pipeline's reflection data and specified color output index.
///
/// # Arguments
/// - `cmd`: command buffer
/// - `width`: desired width of the render target
/// - `height`: desired height of the render target
/// - `pipeline`: graphics pipeline for which the render target is being allocated
/// - `output_index`: color output index in the shader, corresponds to the attachment index
///
pub fn allocate_render_target_for_pipeline(cmd: &gpu::CommandBuffer, width: u32, height: u32, pipeline: &Handle<GraphicsPipeline>, output_index: u32) -> Result<gpu::Image, AssetLoadError>
{

}*/
