#![recursion_limit = "256"]
//#![feature(proc_macro_diagnostic)]
extern crate proc_macro;
extern crate quote;
extern crate syn;

use proc_macro2::{Span, TokenStream};
use quote::{ToTokens, TokenStreamExt};
use syn::spanned::Spanned;

mod vertex;
mod shader_module;
mod descriptor_mapping;

//--------------------------------------------------------------------------------------------------
struct CrateName;
const CRATE: CrateName = CrateName;

impl ToTokens for CrateName {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.append(syn::Ident::new("gpu", Span::call_site()))
    }
}

//--------------------------------------------------------------------------------------------------

#[proc_macro_derive(Vertex, attributes(normalized))]
pub fn vertex_derive(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    match vertex::derive_vertex(input) {
        Ok(tokens) => tokens.into(),
        Err(e) => e.into_compile_error().into(),
    }
}

/*
#[proc_macro_derive(DescriptorMapping, attributes(descriptor))]
pub fn descriptor_mapping_derive(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    match descriptor_mapping::derive_descriptor_mapping(input) {
        Ok(tokens) => tokens.into(),
        Err(e) => e.into_compile_error().into(),
    }
}*/

#[proc_macro_attribute]
pub fn shader_module(attr: proc_macro::TokenStream, item: proc_macro::TokenStream) -> proc_macro::TokenStream {
    match shader_module::shader_module_impl(attr, item) {
        Ok(tokens) => tokens.into(),
        Err(e) => e.into_compile_error().into(),
    }
}
