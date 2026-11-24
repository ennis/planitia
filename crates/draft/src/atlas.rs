use crate::{Srgba8, slice_to_u8};
use gpu::vulkan::*;
use gpu::{ImageAspect, ImageCopyView, Size3D};
use log::debug;
use math::{IRect, U16Vec2, irect_xywh, u16vec2};
use std::ops::Range;

/// Dynamic texture atlas.
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
    /// Creates a new, empty texture atlas with the specified maximum size.
    pub fn new(width: u32, height: u32) -> Self {
        Atlas {
            width,
            height,
            data: vec![Srgba8::TRANSPARENT; width as usize * height as usize],
            texture: gpu::Image::new_texture(width, height, VK_FORMAT_R8G8B8A8_UNORM),
            cursor_x: 0,
            cursor_y: 0,
            row_height: 0,
            dirty: 0..0,
        }
    }

    /// Allocates space in the atlas for an image of the specified size.
    ///
    /// # Arguments
    /// * `width` - Width of the image to allocate.
    /// * `height` - Height of the image to allocate.
    /// * `gap_x` - Horizontal gap to leave after the image (for padding).
    /// * `gap_y` - Vertical gap to leave after the image (for padding).
    ///
    /// # Return value
    /// A writable slice to the pixels allocated in the atlas, or `None` if there is not enough space in the atlas.
    ///
    /// # Note
    /// In order for the written pixels to be visible in the GPU texture, call `update_image()` after writing pixels.
    pub fn try_allocate(
        &mut self,
        width: u32,
        height: u32,
        gap_after_x: u32,
        gap_after_y: u32,
    ) -> Option<AtlasSliceMut<'_>> {
        assert!(width <= self.width, "specified width exceeds atlas width");
        // Move cursor to the next row if necessary.
        if self.cursor_x + width + gap_after_x > self.width {
            self.cursor_x = 0;
            self.cursor_y += self.row_height;
            self.row_height = 0;
        }
        if self.cursor_y + height + gap_after_y > self.height {
            return None;
        }
        // target rect in texel coords
        let rect = irect_xywh(self.cursor_x as i32, self.cursor_y as i32, width as i32, height as i32);
        self.cursor_x += width + gap_after_x;
        self.row_height = self.row_height.max(height + gap_after_y);
        let start = rect.min.y as usize * self.width as usize + rect.min.x as usize;
        // update dirty lines
        self.dirty.start = (rect.min.y as u32).min(self.dirty.start);
        self.dirty.end = (rect.max.y as u32).max(self.dirty.end);
        Some(AtlasSliceMut { data: &mut self.data[start..], rect, stride: self.width })
    }

    /// Writes the specified image in the atlas and returns the allocated rectangle.
    pub fn try_write(
        &mut self,
        width: u32,
        height: u32,
        data: &[Srgba8],
        gap_after_x: u32,
        gap_after_y: u32,
    ) -> Option<IRect> {
        assert_eq!(data.len(), width as usize * height as usize, "data size does not match image dimensions");
        let mut slice = self.try_allocate(width, height, gap_after_x, gap_after_y)?;
        // Copy data into the atlas
        let mut src = 0;
        for y in 0..height {
            slice.row_mut(y).copy_from_slice(&data[src..src + width as usize]);
            src += width as usize;
        }
        Some(slice.rect)
    }

    /// Converts a rectangle from atlas coordinates to normalized texture coordinates.
    pub fn rect_to_normalized_texcoords(&self, rect: IRect) -> [U16Vec2; 2] {
        [
            u16vec2(
                ((rect.min.x as f32) / (self.width as f32) * u16::MAX as f32) as u16,
                ((rect.min.y as f32) / (self.height as f32) * u16::MAX as f32) as u16,
            ),
            u16vec2(
                ((rect.max.x as f32) / (self.width as f32) * u16::MAX as f32) as u16,
                ((rect.max.y as f32) / (self.height as f32) * u16::MAX as f32) as u16,
            ),
        ]
    }

    ///
    pub fn update_image(&mut self) {
        if self.dirty.is_empty() {
            return;
        }
        let height = self.dirty.end - self.dirty.start;
        let range = (self.dirty.start * self.width) as usize..(self.dirty.end * self.width) as usize;
        debug!("uploading atlas rows {}..{}", self.dirty.start, self.dirty.end);
        gpu::upload_image_data(
            ImageCopyView {
                image: &self.texture,
                mip_level: 0,
                origin: gpu::Offset3D { x: 0, y: self.dirty.start as i32, z: 0 },
                aspect: ImageAspect::All,
            },
            Size3D { width: self.width, height, depth: 1 },
            unsafe { slice_to_u8(&self.data[range]) },
        );
        self.dirty.start = self.dirty.end;
    }

    pub fn texture_handle(&self) -> gpu::TextureHandle {
        self.texture.texture_handle()
    }
}

/// Represents a mutable slice of pixels in a texture atlas.
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
