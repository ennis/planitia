use crate::paint::Srgba8;
use gpu::{BarrierFlags, ImageAspect, ImageCopyView, ImageCreateInfo, MemoryLocation, Size3D, vk};
use log::debug;
use math::geom::{IRect, irect_xywh};
use math::{U16Vec2, u16vec2};
use std::cell::RefCell;
use std::ops::{Index, IndexMut, Range};
use std::slice;

/// Texture atlas.
pub struct Atlas {
    pub width: u32,
    pub height: u32,
    pub data: Vec<Srgba8>,
    // TODO allocate more textures if needed
    pub texture: gpu::Image,
    pub cursor_x: u32,
    pub cursor_y: u32,
    pub row_height: u32,
    /// Dirty lines to upload to GPU
    dirty: Range<u32>,
}

impl Default for Atlas {
    fn default() -> Self {
        Self::new(1024, 1024)
    }
}

impl Atlas {
    /// Creates a new, empty texture atlas.
    ///
    /// The width is fixed, but the height will grow as needed (up to max_height).
    pub fn new(width: u32, height: u32) -> Self {
        Atlas {
            width,
            height,
            data: vec![Srgba8::TRANSPARENT; width as usize * height as usize],
            texture: gpu::Image::new(ImageCreateInfo {
                memory_location: MemoryLocation::GpuOnly,
                width,
                height,
                depth: 1,
                format: gpu::Format::R8G8B8A8_UNORM,
                usage: gpu::ImageUsage::SAMPLED | gpu::ImageUsage::TRANSFER_DST,
                mip_levels: 1,
                array_layers: 1,
                samples: 1,
                type_: gpu::ImageType::Image2D,
                ..
            }),
            cursor_x: 0,
            cursor_y: 0,
            row_height: 0,
            dirty: 0..0,
        }
    }

    /// Allocates space for an image of the specified size and returns a reference to the allocated rect for writing.
    ///
    /// # Arguments
    /// * `width` - Width of the image to allocate.
    /// * `height` - Height of the image to allocate.
    /// * `gap_x` - Horizontal gap to leave after the image (for padding).
    /// * `gap_y` - Vertical gap to leave after the image (for padding).
    pub fn allocate(&mut self, width: u32, height: u32, gap_after_x: u32, gap_after_y: u32) -> AtlasSliceMut<'_> {
        assert!(width <= self.width, "Image width exceeds atlas width");

        if self.cursor_x + width + gap_after_x > self.width {
            self.cursor_x = 0;
            self.cursor_y += self.row_height;
            self.row_height = 0;
        }

        if self.cursor_y + height + gap_after_y > self.height {
            panic!("Atlas is full, cannot allocate more space");
            //self.reserve_lines(self.cursor_y + height - self.height);
        }

        // target rect
        let rect = irect_xywh(self.cursor_x as i32, self.cursor_y as i32, width as i32, height as i32);
        self.cursor_x += width + gap_after_x;
        self.row_height = self.row_height.max(height + gap_after_y);

        let start = rect.min.y as usize * self.width as usize + rect.min.x as usize;

        // mark dirty
        self.dirty.start = (rect.min.y as u32).min(self.dirty.start);
        self.dirty.end = (rect.max.y as u32).max(self.dirty.end);

        AtlasSliceMut {
            data: &mut self.data[start..],
            rect,
            stride: self.width,
        }
    }

    /// Writes the specified image in the atlas and returns the rectangle.
    pub fn write(&mut self, width: u32, height: u32, data: &[Srgba8], gap_after_x: u32, gap_after_y: u32) -> IRect {
        assert_eq!(
            data.len(),
            width as usize * height as usize,
            "Data size does not match image dimensions"
        );

        let mut slice = self.allocate(width, height, gap_after_x, gap_after_y);

        // Copy data into the atlas
        let mut src = 0;
        for y in 0..height {
            slice.row_mut(y).copy_from_slice(&data[src..src + width as usize]);
            src += width as usize;
        }
        slice.rect
    }

    /// Converts an area into normalized texture coordinates.
    pub fn rect_to_normalized_texcoords(&self, rect: IRect) -> [U16Vec2; 2] {
        [
            u16vec2(
                ((rect.min.x as f32) / (self.width as f32) * 65535.0) as u16,
                ((rect.min.y as f32) / (self.height as f32) * 65535.0) as u16,
            ),
            u16vec2(
                ((rect.max.x as f32) / (self.width as f32) * 65535.0) as u16,
                ((rect.max.y as f32) / (self.height as f32) * 65535.0) as u16,
            ),
        ]
    }

    /*/// Reserves additional lines in the atlas, growing its height if necessary.
    fn reserve_lines(&mut self, additional_lines: u32) {
        self.texture.replace(None);
        let new_height = u32::max(self.height + additional_lines, self.height * 2);
        debug!("atlas growing to {}×{}", self.width, new_height);
        self.data
            .resize(new_height as usize * self.width as usize, Srgba32::default());
    }*/

    ///
    fn upload_to_gpu(&mut self, cmd: &mut gpu::CommandBuffer) {
        unsafe fn slice_to_u8<T: Copy>(slice: &[T]) -> &[u8] {
            unsafe {
                use std::mem::size_of;
                let len = size_of::<Srgba8>() * slice.len();
                slice::from_raw_parts_mut(slice.as_ptr() as *mut u8, len)
            }
        }

        if self.dirty.is_empty() {
            return;
        }

        let height = self.dirty.end - self.dirty.start;
        let range = (self.dirty.start * self.width) as usize..(self.dirty.end * self.width) as usize;

        debug!("uploading atlas rows {}..{}", self.dirty.start, self.dirty.end);

        cmd.upload_image_data(
            ImageCopyView {
                image: &self.texture,
                mip_level: 0,
                origin: gpu::Offset3D {
                    x: 0,
                    y: self.dirty.start as i32,
                    z: 0,
                },
                aspect: ImageAspect::All,
            },
            Size3D {
                width: self.width,
                height,
                depth: 1,
            },
            unsafe { slice_to_u8(&self.data[range]) },
        );

        self.dirty.start = self.dirty.end;
    }

    pub(crate) fn texture_handle(&self) -> gpu::TextureHandle {
        self.texture.texture_handle()
    }

    /// Returns the GPU image handle for the atlas, uploading it if necessary.
    ///
    /// The image is prepared for shader read access.
    pub(crate) fn prepare_texture(&mut self, cmd: &mut gpu::CommandBuffer) -> gpu::TextureHandle {
        self.upload_to_gpu(cmd);
        cmd.barrier(BarrierFlags::ALL_SHADER_STAGES | BarrierFlags::SAMPLED_READ);
        self.texture.texture_handle()
    }
}

pub struct AtlasSliceMut<'a> {
    pub rect: IRect,
    data: &'a mut [Srgba8],
    stride: u32,
}

impl<'a> AtlasSliceMut<'a> {
    /// Writes a pixel at the specified coordinates within the region.
    pub fn write(&mut self, x: u32, y: u32, color: Srgba8) {
        assert!(x < self.rect.width() as u32 && y < self.rect.height() as u32);
        let index = (y * self.stride + x) as usize;
        self.data[index] = color;
    }

    /// Returns a mutable slice for the specified row within the region.
    pub fn row_mut(&mut self, y: u32) -> &mut [Srgba8] {
        assert!(y < self.rect.height() as u32);
        let start = (y * self.stride) as usize;
        let end = start + self.rect.width() as usize;
        &mut self.data[start..end]
    }
}
