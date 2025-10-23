use crate::util::DeviceExt;
use crate::{
    vk, BufferUsage, CommandStream, DeviceAddress, Image, ImageAspect, ImageCopyBuffer, ImageCopyView, ImageCreateInfo,
    ImageDataLayout, ImageSubresourceLayers, ImageUsage, Offset3D, Rect3D, Size3D,
};

pub trait CommandStreamExt {
    /// Copies the data to a region of an image.
    fn upload_image_data(&mut self, image: ImageCopyView, size: Size3D, data: &[u8]);

    /// Creates an image and copies the data to it.
    ///
    /// # Notes
    /// The image will be created with `ImageUsage::TRANSFER_DST` if it is not already specified in
    /// `create_info.usage`.
    fn create_image_with_data(&mut self, create_info: &ImageCreateInfo, aspect: ImageAspect, data: &[u8]) -> Image;

    /// Shorthand to `blit_image` for blitting the top-level mip level of an image.
    fn blit_full_image_top_mip_level(&mut self, src: &Image, dst: &Image);

    /// Uploads data to a read-only storage buffer and returns a device pointer to it, valid
    /// **only** for the current submission (i.e. until the next call to `present` or `flush`).
    fn upload_temporary<T: Copy>(&mut self, data: &T) -> DeviceAddress<T>;

    /// Uploads a slice of data to a read-only storage buffer and returns a device pointer to it, valid
    /// **only** for the current submission (i.e. until the next call to `present` or `flush`).
    fn upload_temporary_slice<T: Copy>(&mut self, data: &[T]) -> DeviceAddress<[T]>;
}

impl CommandStreamExt for CommandStream {
    fn upload_image_data(&mut self, image: ImageCopyView, size: Size3D, data: &[u8]) {
        let staging_buffer = self.device().upload_slice(BufferUsage::TRANSFER_SRC, data);

        self.copy_buffer_to_image(
            ImageCopyBuffer {
                buffer: staging_buffer.as_bytes(),
                layout: ImageDataLayout {
                    offset: 0,
                    texel_row_length: Some(size.width),
                    row_count: Some(size.height),
                },
            },
            image,
            vk::Extent3D {
                width: size.width,
                height: size.height,
                depth: size.depth,
            },
        );
    }

    fn create_image_with_data(&mut self, create_info: &ImageCreateInfo, aspect: ImageAspect, data: &[u8]) -> Image {
        let mut create_info_with_transfer_dst = create_info.clone();
        create_info_with_transfer_dst.usage |= ImageUsage::TRANSFER_DST;
        let image = self.device().create_image(create_info);
        self.upload_image_data(
            ImageCopyView {
                image: &image,
                mip_level: 0,
                origin: Offset3D::ZERO,
                aspect,
            },
            Size3D {
                width: create_info.width,
                height: create_info.height,
                depth: create_info.depth,
            },
            data,
        );
        image
    }

    fn blit_full_image_top_mip_level(&mut self, src: &Image, dst: &Image) {
        let width = src.width() as i32;
        let height = src.height() as i32;
        self.blit_image(
            &src,
            ImageSubresourceLayers { layer_count: 1, .. },
            Rect3D {
                min: Offset3D { x: 0, y: 0, z: 0 },
                max: Offset3D {
                    x: width,
                    y: height,
                    z: 1,
                },
            },
            &dst,
            ImageSubresourceLayers { layer_count: 1, .. },
            Rect3D {
                min: Offset3D { x: 0, y: 0, z: 0 },
                max: Offset3D {
                    x: width,
                    y: height,
                    z: 1,
                },
            },
            vk::Filter::NEAREST,
        );
    }

    fn upload_temporary<T: Copy>(&mut self, data: &T) -> DeviceAddress<T> {
        let buffer = self.device().upload(BufferUsage::STORAGE, data);
        self.reference_resource(&buffer);
        buffer.device_address()
    }

    fn upload_temporary_slice<T: Copy>(&mut self, data: &[T]) -> DeviceAddress<[T]> {
        let buffer = self.device().upload_slice(BufferUsage::STORAGE, data);
        self.reference_resource(&buffer);
        buffer.device_address()
    }
}
