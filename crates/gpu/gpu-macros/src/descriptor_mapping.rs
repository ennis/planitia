//! The struct represents the push data written into the command buffer.
//! Mappings from push data to descriptor bindings are defined as attributes on fields.
//!
//!  # Example
//!
//! ```
//! #[derive(DescriptorMapping)]
//! struct PushData {
//!     // Fetch the descriptor in the descriptor heap,
//!     // using the provided index.
//!     #[descriptor(set=0, binding=0, push_index)]
//!     index: u32,
//!     // Fetch the descriptor in the descriptor heap,
//!     // using the index loaded from the specified address.
//!     #[descriptor(set=0, binding=0, indirect_index)]
//!     index: Ptr<u32>,
//!     // Fetch the descriptor right here in push data.
//!     #[descriptor(set=0, binding=0, push_data)]
//!     descriptor: Descriptor,
//!
//!
//!
//! }
//! ```
//!

/*
pub enum DescriptorMappingSourceKind {
    /*     VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_CONSTANT_OFFSET_EXT          = 0,
    VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_PUSH_INDEX_EXT               = 1,
    VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_INDIRECT_INDEX_EXT           = 2,
    VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_INDIRECT_INDEX_ARRAY_EXT     = 3,
    VK_DESCRIPTOR_MAPPING_SOURCE_RESOURCE_HEAP_DATA_EXT                 = 4,
    VK_DESCRIPTOR_MAPPING_SOURCE_PUSH_DATA_EXT                          = 5,
    VK_DESCRIPTOR_MAPPING_SOURCE_PUSH_ADDRESS_EXT                       = 6,
    VK_DESCRIPTOR_MAPPING_SOURCE_INDIRECT_ADDRESS_EXT                   = 7,
    VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_SHADER_RECORD_INDEX_EXT      = 8,
    VK_DESCRIPTOR_MAPPING_SOURCE_SHADER_RECORD_DATA_EXT                 = 9,
    VK_DESCRIPTOR_MAPPING_SOURCE_SHADER_RECORD_ADDRESS_EXT              = 10,*/
    HeapWithConstantOffset {},
    HeapWithPushIndex {},
    HeapWithIndirectIndex {},
    HeapWithIndirectIndexArray {},
    ResourceHeapData,
    PushData,
    PushAddress,
    IndirectAddress,
}*/



//
// The mapping is defined via attributes on the field like this:

// #[descriptor(set=0, binding=0, push_index)]
// index: u32,
// #[descriptor(..., indirect_index)]
// index: Ptr<u32>
// #[descriptor(..., resource_heap_data)]
// offset: <....>
// #[descriptor(..., push_data)] OR #[descriptor(...)] (default):
//
//

//
/*
pub(crate) fn derive_descriptor_mapping(input: proc_macro::TokenStream) -> syn::Result<TokenStream> {
    let derive_input: syn::DeriveInput = syn::parse(input)?;
    let struct_name = &derive_input.ident;

    let fields = match derive_input.data {
        syn::Data::Struct(ref s) => &s.fields,
        _ => {
            return Err(syn::Error::new(
                derive_input.span(),
                "`DescriptorMapping` can only be derived on structs",
            ));
        }
    };
}*/