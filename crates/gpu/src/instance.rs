use ash::vk;
use core::ptr;
use log::log;
use std::ffi::{c_void, CStr, CString};
use std::os::raw::c_char;
use std::sync::LazyLock;

#[cfg(windows)]
const INSTANCE_EXTENSIONS: &[&str] = &[
    "VK_KHR_get_surface_capabilities2",
    "VK_EXT_debug_utils",
    "VK_KHR_surface",
    "VK_KHR_win32_surface",
];
// TODO other platforms

//--------------------------------------------------------------------------------------------------

/// Returns the global `ash::Entry` object.
pub fn get_vulkan_entry() -> &'static ash::Entry {
    &VULKAN_ENTRY
}

/// Returns the global vulkan instance object.
pub fn get_vulkan_instance() -> &'static ash::Instance {
    &VULKAN_INSTANCE
}

/// Returns the list of instance extensions that the instance was created with.
pub fn get_instance_extensions() -> &'static [&'static str] {
    INSTANCE_EXTENSIONS
}

pub fn vk_khr_surface() -> &'static ash::khr::surface::Instance {
    &VK_KHR_SURFACE
}

/// List of validation layers to enable
const VALIDATION_LAYERS: &[&str] = &[/*"VK_LAYER_KHRONOS_validation"*/];

/// Vulkan entry points.
static VULKAN_ENTRY: LazyLock<ash::Entry> = LazyLock::new(initialize_vulkan_entry);

/// Vulkan instance and instance function pointers.
static VULKAN_INSTANCE: LazyLock<ash::Instance> = LazyLock::new(create_vulkan_instance);

static VK_KHR_SURFACE: LazyLock<ash::khr::surface::Instance> =
    LazyLock::new(|| ash::khr::surface::Instance::new(&*VULKAN_ENTRY, &*VULKAN_INSTANCE));

/*
static VK_EXT_DEBUG_UTILS_INSTANCE: LazyLock<ash::ext::debug_utils::Instance> =
    LazyLock::new(|| ash::ext::debug_utils::Instance::new(&*VULKAN_ENTRY, &*VULKAN_INSTANCE));

static _DEBUG_MESSENGER: LazyLock<vk::DebugUtilsMessengerEXT> = LazyLock::new(|| {
    unsafe extern "system" fn debug_utils_message_callback(
        message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
        _message_types: vk::DebugUtilsMessageTypeFlagsEXT,
        p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
        _p_user_data: *mut c_void,
    ) -> vk::Bool32 {
        let message = CStr::from_ptr((*p_callback_data).p_message).to_str().unwrap();
        // translate message severity into tracing's log level
        match message_severity {
            vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => {
                log!(log::Level::Trace, "{}", message);
            }
            vk::DebugUtilsMessageSeverityFlagsEXT::INFO => {
                log!(log::Level::Info, "{}", message);
            }
            vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => {
                log!(log::Level::Warn, "{}", message);
            }
            vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => {
                log!(log::Level::Error, "{}", message);
            }
            _ => {
                panic!("unexpected message severity flags")
            }
        };

        vk::FALSE
    }

    let debug_utils_messenger_create_info = vk::DebugUtilsMessengerCreateInfoEXT {
        flags: vk::DebugUtilsMessengerCreateFlagsEXT::empty(),
        message_severity: vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
            | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
            | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
            | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
        message_type: vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
            | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
            | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
        pfn_user_callback: Some(debug_utils_message_callback),
        p_user_data: ptr::null_mut(),
        ..Default::default()
    };

    unsafe {
        // SAFETY: basic FFI and the create info should be correct
        let debug_messenger = VK_EXT_DEBUG_UTILS_INSTANCE
            .create_debug_utils_messenger(&debug_utils_messenger_create_info, None)
            .expect("failed to create debug messenger");
        debug_messenger
    }
});*/

fn initialize_vulkan_entry() -> ash::Entry {
    unsafe { ash::Entry::load().expect("failed to initialize vulkan entry points") }
}

/// Checks if all validation layers are supported
unsafe fn check_validation_layer_support() -> bool {
    let available_layers = VULKAN_ENTRY
        .enumerate_instance_layer_properties()
        .expect("failed to enumerate instance layers");

    VALIDATION_LAYERS.iter().all(|&required_layer| {
        let c_required_layer = CString::new(required_layer).unwrap();
        available_layers
            .iter()
            .any(|&layer| CStr::from_ptr(layer.layer_name.as_ptr()) == c_required_layer.as_c_str())
    })
}

fn create_vulkan_instance() -> ash::Instance {
    unsafe {
        let validation_available = check_validation_layer_support();
        if !validation_available {
            eprintln!("validation layer not available");
        }

        // Convert instance extension strings into C-strings
        let c_instance_extensions: Vec<_> = INSTANCE_EXTENSIONS.iter().map(|&s| CString::new(s).unwrap()).collect();
        let instance_extensions: Vec<_> = c_instance_extensions.iter().map(|s| s.as_ptr()).collect();

        // Convert validation layer names into C-strings
        let c_validation_layers: Vec<_> = VALIDATION_LAYERS.iter().map(|&s| CString::new(s).unwrap()).collect();
        let validation_layers: Vec<_> = c_validation_layers.iter().map(|s| s.as_ptr()).collect();

        let application_info = vk::ApplicationInfo {
            // TODO let the user provide their own name here
            p_application_name: b"GRAAL\0".as_ptr() as *const c_char,
            application_version: 0,
            p_engine_name: b"GRAAL\0".as_ptr() as *const c_char,
            engine_version: 0,
            // require vulkan 1.3
            api_version: vk::make_api_version(0, 1, 3, 0),
            ..Default::default()
        };

        let mut instance_create_info = vk::InstanceCreateInfo {
            flags: Default::default(),
            p_application_info: &application_info,
            enabled_layer_count: 0,
            pp_enabled_layer_names: ptr::null(),
            enabled_extension_count: instance_extensions.len() as u32,
            pp_enabled_extension_names: instance_extensions.as_ptr(),
            ..Default::default()
        };

        if validation_available {
            instance_create_info.enabled_layer_count = validation_layers.len() as u32;
            instance_create_info.pp_enabled_layer_names = validation_layers.as_ptr();
        }

        VULKAN_ENTRY
            .create_instance(&instance_create_info, None)
            .expect("failed to create vulkan instance")
    }
}
