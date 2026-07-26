#![feature(default_field_values)]
use gpu_macros::shader_module;

// Test that the shader_module macro compiles and generates the expected static items
#[shader_module("tests/test_shader.slang")]
mod test_shader {}

#[test]
fn test_shader_module_macro() {
    // This test verifies that the shader module compiles and generates the expected items
    // Note: This won't actually run the shader, just verifies compilation
    
    // The macro should generate a COMPUTEMAIN static
    let _ = &test_shader::computeMain;
}
