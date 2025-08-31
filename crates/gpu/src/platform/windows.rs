use crate::device::{get_vk_sample_count, ResourceAllocation};
use crate::{vk, Device, Image, ImageCreateInfo, ImageInner, RcDevice, Size3D};
use ash::vk::{HANDLE, SECURITY_ATTRIBUTES};
use gpu_allocator::MemoryLocation;
use std::ffi::{c_void, OsStr};
use std::ptr;
use std::rc::Rc;
use std::sync::atomic::AtomicU64;
use std::sync::Arc;
/*/// External memory handle types.
#[derive(Clone, Debug)]
pub enum Win32ExternalMemoryHandle<'a> {
    D3D11Texture(crate::platform::windows::Win32Handle<'a>),
    D3D11TextureKMT(crate::platform::windows::Win32Handle<'a>),
}

impl Win32ExternalMemoryHandle {
    pub(crate) fn handle_type(&self) -> vk::ExternalMemoryHandleTypeFlags {
        match self {
            Win32ExternalMemoryHandle::D3D11Texture(_) => {
                vk::ExternalMemoryHandleTypeFlags::D3D11_TEXTURE
            }
            Win32ExternalMemoryHandle::D3D11TextureKMT(_) => {
                vk::ExternalMemoryHandleTypeFlags::D3D11_TEXTURE_KMT
            }
            _ => unreachable!(), // or is it?
        }
    }
}*/

/*#[derive(Copy, Clone, Debug)]
pub struct Win32Handle<'a> {
    pub handle: HANDLE,
    pub name: &'a str,
}*/

fn handle_name_to_wstr(name: Option<&str>) -> (Vec<u16>, *const u16) {
    use std::os::windows::ffi::OsStrExt;

    if let Some(name) = name {
        let mut w_name: Vec<u16> = OsStr::new(name).encode_wide().collect();
        w_name.push(0);
        let ptr = w_name.as_ptr();
        (w_name, ptr)
    } else {
        (Vec::new(), ptr::null())
    }
}

pub(crate) enum DedicatedAllocation {
    Buffer(vk::Buffer),
    Image(vk::Image),
}

pub(crate) unsafe fn import_external_memory(
    device: &RcDevice,
    memory_requirements: &vk::MemoryRequirements,
    required_flags: vk::MemoryPropertyFlags,
    preferred_flags: vk::MemoryPropertyFlags,
    handle_type: vk::ExternalMemoryHandleTypeFlags,
    handle: HANDLE,
    handle_name: Option<&str>,
    dedicated: Option<DedicatedAllocation>,
) -> vk::DeviceMemory {
    // TODO proper error handling
    let win32_handle_properties = device
        .platform_extensions
        .khr_external_memory_win32
        .get_memory_win32_handle_properties(handle_type, handle)
        .expect("vkGetMemoryWin32HandlePropertiesKHR failed");

    // find a memory type that both matches the resource requirement and the external handle requirements for importing
    let memory_type_bits = memory_requirements.memory_type_bits & win32_handle_properties.memory_type_bits;
    let memory_type_index = device
        .find_compatible_memory_type(memory_type_bits, required_flags, preferred_flags)
        .expect("could not find a compatible memory type for importing external memory");

    // import memory
    let (_, handle_name_wstr) = handle_name_to_wstr(handle_name);

    let mut dedicated_allocate_info: vk::MemoryDedicatedAllocateInfo;
    let mut p_dedicated_allocate_info = ptr::null();

    if let Some(dedicated) = dedicated {
        dedicated_allocate_info = vk::MemoryDedicatedAllocateInfo {
            image: Default::default(),
            buffer: Default::default(),
            ..Default::default()
        };

        match dedicated {
            DedicatedAllocation::Buffer(buffer) => {
                dedicated_allocate_info.buffer = buffer;
            }
            DedicatedAllocation::Image(image) => {
                dedicated_allocate_info.image = image;
            }
        }
        p_dedicated_allocate_info = &dedicated_allocate_info as *const _ as *const c_void;
    }

    let import_memory_win32_handle_info = vk::ImportMemoryWin32HandleInfoKHR {
        p_next: p_dedicated_allocate_info,
        handle_type,
        handle,
        name: handle_name_wstr,
        ..Default::default()
    };

    let memory_allocate_info = vk::MemoryAllocateInfo {
        p_next: &import_memory_win32_handle_info as *const _ as *const c_void,
        allocation_size: memory_requirements.size,
        memory_type_index,
        ..Default::default()
    };

    let device_memory = device.raw.allocate_memory(&memory_allocate_info, None).unwrap();

    device_memory
}

pub trait DeviceExtWindows {
    unsafe fn create_imported_image_win32(
        self: &Rc<Self>,
        image_info: &ImageCreateInfo,
        required_memory_flags: vk::MemoryPropertyFlags,
        preferred_memory_flags: vk::MemoryPropertyFlags,
        handle_type: vk::ExternalMemoryHandleTypeFlags,
        handle: HANDLE,
        handle_name: Option<&str>,
    ) -> Image;

    unsafe fn create_exported_image_win32(
        self: &Rc<Self>,
        memory_location: MemoryLocation,
        image_info: &ImageCreateInfo,
        handle_type: vk::ExternalMemoryHandleTypeFlags,
        security_attributes: *const SECURITY_ATTRIBUTES,
        access_flags: u32,
        handle_name: Option<&str>,
    ) -> (Image, HANDLE);

    /// Creates a semaphore and exports it to a Windows handle of the specified type.
    /// The returned semaphore should be deleted with `vkDestroySemaphore`.
    unsafe fn create_exported_semaphore_win32(
        self: &Rc<Self>,
        handle_type: vk::ExternalSemaphoreHandleTypeFlags,
        security_attributes: *const SECURITY_ATTRIBUTES,
        access_flags: u32,
        handle_name: Option<&str>,
    ) -> (vk::Semaphore, HANDLE);

    unsafe fn create_imported_semaphore_win32(
        self: &Rc<Self>,
        import_flags: vk::SemaphoreImportFlags,
        handle_type: vk::ExternalSemaphoreHandleTypeFlags,
        handle: HANDLE,
        handle_name: Option<&str>,
    ) -> vk::Semaphore;
}

impl DeviceExtWindows for Device {
    unsafe fn create_imported_image_win32(
        self: &Rc<Self>,
        image_info: &ImageCreateInfo,
        required_memory_flags: vk::MemoryPropertyFlags,
        preferred_memory_flags: vk::MemoryPropertyFlags,
        win32_handle_type: vk::ExternalMemoryHandleTypeFlags,
        win32_handle: HANDLE,
        win32_handle_name: Option<&str>,
    ) -> Image {
        let external_memory_image_create_info = vk::ExternalMemoryImageCreateInfo {
            handle_types: win32_handle_type,
            ..Default::default()
        };
        let create_info = vk::ImageCreateInfo {
            p_next: &external_memory_image_create_info as *const _ as *const c_void,
            image_type: image_info.type_.to_vk_image_type(),
            format: image_info.format,
            extent: vk::Extent3D {
                width: image_info.width,
                height: image_info.height,
                depth: image_info.depth,
            },
            mip_levels: image_info.mip_levels,
            array_layers: image_info.array_layers,
            samples: get_vk_sample_count(image_info.samples),
            tiling: vk::ImageTiling::OPTIMAL,
            usage: image_info.usage.to_vk_image_usage_flags(),
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            queue_family_index_count: 0,
            p_queue_family_indices: ptr::null(),
            ..Default::default()
        };
        let handle = self
            .raw
            .create_image(&create_info, None)
            .expect("failed to create image");
        let mem_req = self.raw.get_image_memory_requirements(handle);
        let device_memory = import_external_memory(
            self,
            &mem_req,
            required_memory_flags,
            preferred_memory_flags,
            win32_handle_type,
            win32_handle,
            win32_handle_name,
            Some(DedicatedAllocation::Image(handle)),
        );
        self.raw.bind_image_memory(handle, device_memory, 0).unwrap();
        let id = self.image_ids.lock().unwrap().insert(());

        Image {
            handle,
            inner: Some(Arc::new(ImageInner {
                device: self.clone(),
                id,
                last_submission_index: AtomicU64::new(0),
                allocation: ResourceAllocation::DeviceMemory { device_memory },
                handle,
                swapchain_image: false,
            })),
            usage: image_info.usage,
            type_: image_info.type_,
            format: image_info.format,
            size: Size3D {
                width: image_info.width,
                height: image_info.height,
                depth: image_info.depth,
            },
        }
    }

    unsafe fn create_exported_image_win32(
        self: &Rc<Self>,
        memory_location: MemoryLocation,
        image_info: &ImageCreateInfo,
        handle_type: vk::ExternalMemoryHandleTypeFlags,
        security_attributes: *const SECURITY_ATTRIBUTES,
        access_flags: u32,
        handle_name: Option<&str>,
    ) -> (Image, HANDLE) {
        let external_memory_image_create_info = vk::ExternalMemoryImageCreateInfo {
            handle_types: handle_type,
            ..Default::default()
        };
        let create_info = vk::ImageCreateInfo {
            p_next: &external_memory_image_create_info as *const _ as *const c_void,
            image_type: image_info.type_.to_vk_image_type(),
            format: image_info.format,
            extent: vk::Extent3D {
                width: image_info.width,
                height: image_info.height,
                depth: image_info.depth,
            },
            mip_levels: image_info.mip_levels,
            array_layers: image_info.array_layers,
            samples: get_vk_sample_count(image_info.samples),
            tiling: vk::ImageTiling::OPTIMAL,
            usage: image_info.usage.to_vk_image_usage_flags(),
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            queue_family_index_count: 0,
            p_queue_family_indices: ptr::null(),
            ..Default::default()
        };
        let handle = self
            .raw
            .create_image(&create_info, None)
            .expect("failed to create image");
        let mem_req = self.raw.get_image_memory_requirements(handle);

        let (_, handle_name_wstr) = handle_name_to_wstr(handle_name);

        let (required_memory_properties, preferred_memory_properties) = match memory_location {
            MemoryLocation::Unknown => Default::default(),
            MemoryLocation::GpuOnly => (
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            ),
            MemoryLocation::CpuToGpu => (
                vk::MemoryPropertyFlags::HOST_VISIBLE
                    | vk::MemoryPropertyFlags::HOST_COHERENT
                    | vk::MemoryPropertyFlags::DEVICE_LOCAL,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            ),
            MemoryLocation::GpuToCpu => (
                vk::MemoryPropertyFlags::HOST_VISIBLE
                    | vk::MemoryPropertyFlags::HOST_COHERENT
                    | vk::MemoryPropertyFlags::HOST_CACHED,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            ),
        };

        let memory_type_index = self
            .find_compatible_memory_type(
                mem_req.memory_type_bits,
                required_memory_properties,
                preferred_memory_properties,
            )
            .expect("could not find a compatible memory type for exporting memory");

        let win32_handle_info = vk::ExportMemoryWin32HandleInfoKHR {
            p_attributes: security_attributes,
            dw_access: access_flags,
            name: handle_name_wstr,
            ..Default::default()
        };
        let export_memory_allocate_info = vk::ExportMemoryAllocateInfo {
            p_next: &win32_handle_info as *const _ as *const c_void,
            handle_types: handle_type,
            ..Default::default()
        };
        let memory_allocate_info = vk::MemoryAllocateInfo {
            p_next: &export_memory_allocate_info as *const _ as *const c_void,
            allocation_size: mem_req.size,
            memory_type_index,
            ..Default::default()
        };

        let device_memory = self
            .raw
            .allocate_memory(&memory_allocate_info, None)
            .expect("failed to allocate exported memory");

        // retrieve the win32 handle
        let get_win32_handle_info = vk::MemoryGetWin32HandleInfoKHR {
            memory: device_memory,
            handle_type,
            ..Default::default()
        };

        // TODO proper error handling
        let win32_handle = self
            .platform_extensions
            .khr_external_memory_win32
            .get_memory_win32_handle(&get_win32_handle_info)
            .expect("vkGetMemoryWin32HandleKHR failed");

        // bind memory
        self.raw.bind_image_memory(handle, device_memory, 0).unwrap();

        let id = self.image_ids.lock().unwrap().insert(());
        let image = Image {
            handle,
            inner: Some(Arc::new(ImageInner {
                device: self.clone(),
                id,
                last_submission_index: AtomicU64::new(0),
                allocation: ResourceAllocation::DeviceMemory { device_memory },
                handle,
                swapchain_image: false,
            })),
            usage: image_info.usage,
            type_: image_info.type_,
            format: image_info.format,
            size: Size3D {
                width: image_info.width,
                height: image_info.height,
                depth: image_info.depth,
            },
        };
        (image, win32_handle)
    }

    unsafe fn create_exported_semaphore_win32(
        self: &Rc<Self>,
        handle_type: vk::ExternalSemaphoreHandleTypeFlags,
        security_attributes: *const SECURITY_ATTRIBUTES,
        access_flags: u32,
        handle_name: Option<&str>,
    ) -> (vk::Semaphore, HANDLE) {
        let (_, handle_name_wstr) = handle_name_to_wstr(handle_name);

        let export_semaphore_win32_handle_info = vk::ExportSemaphoreWin32HandleInfoKHR {
            p_attributes: security_attributes,
            dw_access: access_flags,
            name: handle_name_wstr,
            ..Default::default()
        };
        let export_semaphore_create_info = vk::ExportSemaphoreCreateInfo {
            p_next: &export_semaphore_win32_handle_info as *const _ as *const c_void,
            handle_types: handle_type,
            ..Default::default()
        };
        let semaphore_create_info = vk::SemaphoreCreateInfo {
            p_next: &export_semaphore_create_info as *const _ as *const c_void,
            ..Default::default()
        };

        let semaphore = self.raw.create_semaphore(&semaphore_create_info, None).unwrap();

        let get_win32_handle_info = vk::SemaphoreGetWin32HandleInfoKHR {
            semaphore,
            handle_type,
            ..Default::default()
        };

        let handle = self
            .platform_extensions
            .khr_external_semaphore_win32
            .get_semaphore_win32_handle(&get_win32_handle_info)
            .expect("vkGetSemaphoreWin32HandleKHR failed");
        (semaphore, handle)
    }

    unsafe fn create_imported_semaphore_win32(
        self: &Rc<Self>,
        import_flags: vk::SemaphoreImportFlags,
        handle_type: vk::ExternalSemaphoreHandleTypeFlags,
        handle: HANDLE,
        handle_name: Option<&str>,
    ) -> vk::Semaphore {
        let (_, handle_name_wstr) = handle_name_to_wstr(handle_name);

        // create the semaphore
        let is_timeline = match handle_type {
            vk::ExternalSemaphoreHandleTypeFlags::D3D12_FENCE => true,
            _ => panic!("unsupported external semaphore type"),
        };

        let timeline_create_info = vk::SemaphoreTypeCreateInfo {
            semaphore_type: vk::SemaphoreType::TIMELINE,
            initial_value: 0,
            ..Default::default()
        };

        let semaphore_create_info = vk::SemaphoreCreateInfo {
            p_next: if is_timeline {
                &timeline_create_info as *const _ as *const c_void
            } else {
                ptr::null()
            },
            ..Default::default()
        };

        let semaphore = self.raw.create_semaphore(&semaphore_create_info, None).unwrap();

        let import_semaphore_win32_handle_info = vk::ImportSemaphoreWin32HandleInfoKHR {
            semaphore,
            flags: import_flags, // ?????
            handle_type,
            handle,
            name: handle_name_wstr,
            ..Default::default()
        };

        self.platform_extensions
            .khr_external_semaphore_win32
            .import_semaphore_win32_handle(&import_semaphore_win32_handle_info)
            .expect("vkImportSemaphoreWin32HandleKHR failed");

        semaphore
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

const PLATFORM_DEVICE_EXTENSIONS: &[&str] = &["VK_KHR_external_memory_win32", "VK_KHR_external_semaphore_win32"];

/// Windows-specific vulkan extensions
pub struct PlatformExtensions {
    pub khr_external_memory_win32: ash::extensions::khr::ExternalMemoryWin32,
    pub khr_external_semaphore_win32: ash::extensions::khr::ExternalSemaphoreWin32,
}

impl PlatformExtensions {
    pub(crate) fn names() -> &'static [&'static str] {
        PLATFORM_DEVICE_EXTENSIONS
    }

    pub(crate) fn load(_entry: &ash::Entry, instance: &ash::Instance, device: &ash::Device) -> PlatformExtensions {
        let khr_external_memory_win32 = ash::extensions::khr::ExternalMemoryWin32::new(instance, device);
        let khr_external_semaphore_win32 = ash::extensions::khr::ExternalSemaphoreWin32::new(instance, device);
        PlatformExtensions {
            khr_external_memory_win32,
            khr_external_semaphore_win32,
        }
    }
}
