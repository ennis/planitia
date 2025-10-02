use std::cell::RefCell;
use std::ops::{Index, IndexMut};
use std::slice;
use crate::paint::Srgba32;
use log::debug;
use gpu::{vk, Barrier, ImageCopyView, ImageCreateInfo, MemoryLocation, Size3D};
use gpu::util::CommandStreamExt;
use math::geom::{IRect, irect_xywh};
use crate::context::get_gpu_device;

pub struct Atlas {
    pub width: u32,
    pub height: u32,
    pub data: Vec<Srgba32>,
    pub texture: Option<gpu::Image>,
    pub cursor_x: u32,
    pub cursor_y: u32,
    pub row_height: u32,
}

impl Default for Atlas {
    fn default() -> Self {
        Self::new(1024, 256)
    }
}

impl Atlas {
    /// Creates a new, empty texture atlas.
    ///
    /// The width is fixed, but the height will grow as needed.
    pub fn new(width: u32, init_height: u32) -> Self {
        Atlas {
            width,
            height: init_height,
            data: vec![Srgba32::TRANSPARENT; width as usize * init_height as usize],
            texture: None,
            cursor_x: 0,
            cursor_y: 0,
            row_height: 0,
        }
    }

    /// Allocates space for an image of the specified size and returns a reference to the allocated rect for writing.
    pub fn allocate(&mut self, width: u32, height: u32) -> AtlasSliceMut<'_> {
        assert!(width <= self.width, "Image width exceeds atlas width");

        if self.cursor_x + width > self.width {
            self.cursor_x = 0;
            self.cursor_y += self.row_height;
            self.row_height = 0;
        }

        if self.cursor_y + height > self.height {
            self.reserve_lines(self.cursor_y + height - self.height);
        }

        // target rect
        let rect = irect_xywh(self.cursor_x as i32, self.cursor_y as i32, width as i32, height as i32);
        self.cursor_x += width;
        self.row_height = self.row_height.max(height);

        let start = rect.min.y as usize * self.width as usize + rect.min.x as usize;
        AtlasSliceMut {
            data: &mut self.data[start..],
            rect,
            stride: self.width,
        }
    }

    /// Writes the specified image in the atlas and returns the rectangle.
    pub fn write(&mut self, width: u32, height: u32, data: &[Srgba32]) -> IRect {

        assert_eq!(data.len(), width as usize * height as usize, "Data size does not match image dimensions");

        let mut slice = self.allocate(width, height);

        // Copy data into the atlas
        let mut src = 0;
        for y in 0..height {
            slice.row_mut(y).copy_from_slice(&data[src..src + width as usize]);
            src += width as usize;
        }
        let rect = slice.rect;

        // invalidate texture
        self.texture = None;
        rect
    }

    /// Reserves additional lines in the atlas, growing its height if necessary.
    fn reserve_lines(&mut self, additional_lines: u32) {
        self.texture    = None;
        let new_height = u32::max(self.height + additional_lines, self.height * 2);
        debug!("atlas growing to {}×{}", self.width, new_height);
        self.data
            .resize(new_height as usize * self.width as usize, Srgba32::default());
    }

    /*///
    fn upload_to_gpu(&self, cmd: &mut gpu::CommandStream) {

        unsafe fn slice_to_u8<T: Copy>(slice: &[T]) -> &[u8] {
            use std::mem::size_of;
            let len = size_of::<Srgba32>() * slice.len();
            slice::from_raw_parts_mut(slice.as_ptr() as *mut u8, len)
        }

        let gpu = get_gpu_device();
        if self.texture.borrow().is_none() {

            let image = cmd.create_image_with_data(&ImageCreateInfo {
                memory_location: MemoryLocation::GpuOnly,
                width: self.width,
                height: self.height,
                format: gpu::Format::R8G8B8A8_UNORM,
                usage: gpu::ImageUsage::SAMPLED | gpu::ImageUsage::TRANSFER_DST,
                mip_levels: 1,
                array_layers: 1,
                samples: 1,
                type_: gpu::ImageType::Image2D,
                depth: 0,
            },
                                unsafe {
                                    slice_to_u8(&self.data)
                                }
            );

            self.texture.replace(Some(image));
        }
        self.texture.as_ref().unwrap()
    }

    /// Returns the GPU image handle for the atlas, uploading it if necessary.
    ///
    /// The image is prepared for shader read access.
    pub(crate) fn texture_handle(&self, cmd: &mut gpu::CommandStream) -> gpu::ImageHandle {
        self.upload_to_gpu(cmd);
        let tex = self.texture.borrow().as_ref().unwrap();
        cmd.barrier(Barrier::new().sample_read_image(tex));
        tex.handle()
    }*/
}

pub struct AtlasSliceMut<'a> {
    pub rect: IRect,
    data: &'a mut [Srgba32],
    stride: u32,
}

impl<'a> AtlasSliceMut<'a> {
    /// Writes a pixel at the specified coordinates within the region.
    pub fn write(&mut self, x: u32, y: u32, color: Srgba32) {
        assert!(x < self.rect.width() as u32 && y < self.rect.height() as u32);
        let index = (y * self.stride + x) as usize;
        self.data[index] = color;
    }

    /// Returns a mutable slice for the specified row within the region.
    pub fn row_mut(&mut self, y: u32) -> &mut [Srgba32] {
        assert!(y < self.rect.height() as u32);
        let start = (y * self.stride) as usize;
        let end = start + self.rect.width() as usize;
        &mut self.data[start..end]
    }
}