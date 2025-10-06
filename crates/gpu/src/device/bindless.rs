//! TODO: implement a single table for all descriptors, with VK_EXT_mutable_descriptor_type

use crate::{Device, ResourceDescriptorIndex, SamplerDescriptorIndex, SamplerHandle};
use ash::vk;
use ash::vk::DescriptorType;
use log::trace;
use std::ffi::c_void;
use std::ptr;

type DT = vk::DescriptorType;

const SAMPLER_TABLE_BINDING: u32 = 0;
const IMAGE_TABLE_BINDING: u32 = 2;

/// Bindless descriptor table.
#[derive(Debug)]
pub(crate) struct BindlessDescriptorTable {
    pub(crate) layout: vk::DescriptorSetLayout,
    pub(crate) set: vk::DescriptorSet,
    pool: vk::DescriptorPool,
    count: usize,
}

impl BindlessDescriptorTable {
    pub(crate) unsafe fn new(device: &ash::Device, count: usize) -> BindlessDescriptorTable {
        let bindings = [
            // samplers
            vk::DescriptorSetLayoutBinding {
                binding: 0,
                descriptor_type: vk::DescriptorType::MUTABLE_EXT,
                descriptor_count: count as u32,
                stage_flags: vk::ShaderStageFlags::ALL,
                p_immutable_samplers: ptr::null(),
            },
            // combined image samplers (unused)
            vk::DescriptorSetLayoutBinding {
                binding: 1,
                descriptor_type: vk::DescriptorType::MUTABLE_EXT,
                descriptor_count: 0,
                stage_flags: vk::ShaderStageFlags::ALL,
                p_immutable_samplers: ptr::null(),
            },
            vk::DescriptorSetLayoutBinding {
                binding: 2,
                descriptor_type: vk::DescriptorType::MUTABLE_EXT,
                descriptor_count: count as u32,
                stage_flags: vk::ShaderStageFlags::ALL,
                p_immutable_samplers: ptr::null(),
            },
        ];

        let binding_0_types = [vk::DescriptorType::SAMPLER];
        let binding_1_types = [vk::DescriptorType::COMBINED_IMAGE_SAMPLER];
        let binding_2_types = [vk::DescriptorType::SAMPLED_IMAGE, vk::DescriptorType::STORAGE_IMAGE];

        let mutable_descriptor_type_list = [
            vk::MutableDescriptorTypeListEXT {
                descriptor_type_count: binding_0_types.len() as u32,
                p_descriptor_types: binding_0_types.as_ptr(),
            },
            vk::MutableDescriptorTypeListEXT {
                descriptor_type_count: binding_1_types.len() as u32,
                p_descriptor_types: binding_1_types.as_ptr(),
            },
            vk::MutableDescriptorTypeListEXT {
                descriptor_type_count: binding_2_types.len() as u32,
                p_descriptor_types: binding_2_types.as_ptr(),
            },
        ];

        let mutable_desc = vk::MutableDescriptorTypeCreateInfoEXT {
            mutable_descriptor_type_list_count: mutable_descriptor_type_list.len() as u32,
            p_mutable_descriptor_type_lists: mutable_descriptor_type_list.as_ptr(),
            ..Default::default()
        };

        let flags = [
            vk::DescriptorBindingFlags::PARTIALLY_BOUND | vk::DescriptorBindingFlags::UPDATE_UNUSED_WHILE_PENDING,
            vk::DescriptorBindingFlags::PARTIALLY_BOUND | vk::DescriptorBindingFlags::UPDATE_UNUSED_WHILE_PENDING,
            vk::DescriptorBindingFlags::PARTIALLY_BOUND | vk::DescriptorBindingFlags::UPDATE_UNUSED_WHILE_PENDING,
        ];
        let dslbfci = vk::DescriptorSetLayoutBindingFlagsCreateInfo {
            p_next: &mutable_desc as *const _ as *const c_void,
            binding_count: flags.len() as u32,
            p_binding_flags: flags.as_ptr(),
            ..Default::default()
        };

        let dslci = vk::DescriptorSetLayoutCreateInfo {
            p_next: &dslbfci as *const _ as *const c_void,
            flags: Default::default(),
            binding_count: bindings.len() as u32,
            p_bindings: bindings.as_ptr(),
            ..Default::default()
        };

        let layout = device
            .create_descriptor_set_layout(&dslci, None)
            .expect("failed to create descriptor set layout");

        // pool for all descriptors
        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::SAMPLER,
                descriptor_count: count as u32,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::MUTABLE_EXT,
                descriptor_count: count as u32,
            },
        ];

        let pool_create_info = vk::DescriptorPoolCreateInfo {
            p_next: &mutable_desc as *const _ as *const c_void,
            max_sets: 1,
            pool_size_count: pool_sizes.len() as u32,
            p_pool_sizes: pool_sizes.as_ptr(),
            ..Default::default()
        };

        let pool = unsafe {
            device
                .create_descriptor_pool(&pool_create_info, None)
                .expect("failed to create descriptor pool")
        };

        // and allocate a new descriptor set from it
        let set = unsafe {
            device
                .allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo {
                    descriptor_pool: pool,
                    descriptor_set_count: 1,
                    p_set_layouts: &layout,
                    ..Default::default()
                })
                .expect("failed to allocate descriptor set")[0]
        };

        BindlessDescriptorTable {
            layout,
            pool,
            set,
            count,
        }
    }
}

impl Device {
    pub(crate) unsafe fn create_global_image_descriptor(
        &self,
        image_view: vk::ImageView,
        descriptor_type: vk::DescriptorType,
        image_layout: vk::ImageLayout,
    ) -> ResourceDescriptorIndex {
        let d = self.global_descriptors.lock().unwrap();
        let index = self.resource_descriptor_indices.lock().unwrap().insert(());
        let dst_array_element = index.index();
        assert!(dst_array_element < d.count as u32);
        let write = vk::WriteDescriptorSet {
            dst_set: d.set,
            dst_binding: IMAGE_TABLE_BINDING,
            dst_array_element,
            descriptor_count: 1,
            descriptor_type: descriptor_type,
            p_image_info: &vk::DescriptorImageInfo {
                image_view,
                image_layout: image_layout,
                ..Default::default()
            },
            ..Default::default()
        };
        trace!("image_descriptors[{}] = {:?}", dst_array_element, image_view);
        unsafe {
            self.raw.update_descriptor_sets(&[write], &[]);
        }
        index
    }

    pub(crate) unsafe fn create_global_sampler_descriptor(&self, sampler: vk::Sampler) -> SamplerDescriptorIndex {
        let d = self.global_descriptors.lock().unwrap();
        let index = self.sampler_descriptor_indices.lock().unwrap().insert(());
        let dst_array_element = index.index();
        assert!(dst_array_element < d.count as u32);
        let write = vk::WriteDescriptorSet {
            dst_set: d.set,
            dst_binding: SAMPLER_TABLE_BINDING,
            dst_array_element,
            descriptor_count: 1,
            descriptor_type: DT::SAMPLER,
            p_image_info: &vk::DescriptorImageInfo {
                sampler,
                ..Default::default()
            },
            ..Default::default()
        };
        trace!("sampler_descriptors[{}] = {:?}", dst_array_element, sampler);
        unsafe {
            self.raw.update_descriptor_sets(&[write], &[]);
        }
        index
    }
}
