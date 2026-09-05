mod push_buffer;

pub use push_buffer::*;


/// Returns a reference to a temporary instance of an anonymous repr(C) struct to be used as shader root parameters.
///
/// # Example
///
///```rust
/// cmd.dispatch(1, 1, 1, root_params! {
///     time: f32 = 1.0,
///     resolution: [f32; 2] = [800.0, 600.0],
/// });
///```
#[macro_export]
macro_rules! root_params {
    ( $( $field:ident : $ty:ty = $val:expr ),* ) => {
        $crate::PushDataSource::IndirectUpload(&{
            #[repr(C)]
            #[derive(Copy, Clone)]
            struct Params {
                $( $field: $ty, )*
            }
            Params {
                $( $field: $val, )*
            }
        })
    };
}

pub use root_params;
