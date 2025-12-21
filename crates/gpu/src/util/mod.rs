mod command;
mod upload_buffer;
/*
pub unsafe fn blit_images(
    queue: &mut Queue,
    src_image: ImageHandle,
    dst_image: ImageHandle,
    width: u32,
    height: u32,
    aspect_mask: vk::ImageAspectFlags,
) {
    let cb = queue.create_command_buffer();
    let regions = &[vk::ImageBlit {
        src_subresource: vk::ImageSubresourceLayers {
            aspect_mask,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        },
        src_offsets: [
            vk::Offset3D { x: 0, y: 0, z: 0 },
            vk::Offset3D {
                x: width as i32,
                y: height as i32,
                z: 1,
            },
        ],
        dst_subresource: vk::ImageSubresourceLayers {
            aspect_mask,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        },
        dst_offsets: [
            vk::Offset3D { x: 0, y: 0, z: 0 },
            vk::Offset3D {
                x: width as i32,
                y: height as i32,
                z: 1,
            },
        ],
    }];

    let device = queue.device();
    device
        .begin_command_buffer(
            cb,
            &vk::CommandBufferBeginInfo {
                flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                ..Default::default()
            },
        )
        .unwrap();
    device.cmd_blit_image(
        cb,
        src_image.vk,
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        dst_image.vk,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        regions,
        vk::Filter::NEAREST,
    );
    device.end_command_buffer(cb).unwrap();

    let mut blit = Submission::new();
    blit.set_name("blit_images");
    blit.use_image(src_image.id, ResourceState::TRANSFER_SRC);
    blit.use_image(dst_image.id, ResourceState::TRANSFER_DST);
    blit.push_command_buffer(cb);
    queue.submit(blit).expect("blit_images failed");
}
*/

/// Returns a reference to a temporary instance of an anonymous repr(C) struct to be used as shader root parameters.
///
/// # Example
///
///```rust
/// encoder.push_constants(root_params! {
///     time: f32 = 1.0,
///     resolution: [f32; 2] = [800.0, 600.0],
/// });
///```
#[macro_export]
macro_rules! root_params {
    ( $( $field:ident : $ty:ty = $val:expr ),* ) => {
        {
            #[repr(C)]
            #[derive(Copy, Clone)]
            struct PushConstants {
                $( $field: $ty, )*
            }
            &PushConstants {
                $( $field: $val, )*
            }
        }
    };
}

pub use root_params;