use crate::{Device, ModuleId};
use crate::helper::HasPrivateData;
use crate::spirv::{EntryPointId, EntryPointInfo, Module};
use crate::util::find_next;
use vulkan::*;
use std::ffi::CStr;
use std::slice::from_raw_parts;

pub struct ShaderStageInfo {
    pub module: ModuleId,
    pub entry_point: EntryPointId,
}

#[derive(Default)]
pub struct PipelineData {
    pub name: String,
    // Entry point reflection information
    pub vertex: Option<ShaderStageInfo>,
    pub fragment: Option<ShaderStageInfo>,
    pub mesh: Option<ShaderStageInfo>,
    pub task: Option<ShaderStageInfo>,
    pub compute: Option<ShaderStageInfo>,
}

impl HasPrivateData for VkPipeline {
    type PrivateData = PipelineData;
}

impl Device {
    fn register_shader_module(&self, smci: &VkShaderModuleCreateInfo, entry_point: &str) -> ShaderStageInfo {
        let mut mods = self.modules.lock();
        let spirv = unsafe { from_raw_parts((*smci).pCode, (*smci).codeSize / 4) };
        let module = Module::parse(spirv).unwrap();
        let entry_point_id = module.find_entry_point(&entry_point).unwrap();
        let module_id = mods.insert(module);
        let info = ShaderStageInfo { module: module_id, entry_point: entry_point_id };
        info
    }

    unsafe fn create_graphics_pipeline_data(&self, create_info: &VkGraphicsPipelineCreateInfo) -> PipelineData {
        let mut data = PipelineData::default();
        let stages = from_raw_parts(create_info.pStages, create_info.stageCount as usize);
        for (i, stage) in stages.iter().enumerate() {
            let ep_name = CStr::from_ptr(stage.pName).to_string_lossy();
            if let Some(smci) = find_next::<VkShaderModuleCreateInfo>(stage) {
                let ep = self.register_shader_module(&*smci, &ep_name);
                match stage.stage {
                    VK_SHADER_STAGE_VERTEX_BIT => data.vertex = Some(ep),
                    VK_SHADER_STAGE_FRAGMENT_BIT => data.fragment = Some(ep),
                    VK_SHADER_STAGE_MESH_BIT_EXT => data.mesh = Some(ep),
                    VK_SHADER_STAGE_TASK_BIT_EXT => data.task = Some(ep),
                    VK_SHADER_STAGE_COMPUTE_BIT => data.compute = Some(ep),
                    _ => {
                        eprintln!("unsupported shader stage: {:?}", stage.stage);
                    }
                }
            }
            if i != 0 {
                data.name.push('/');
            }
            // Set a default name composed of all the entry point names (hopefully they are not all called "main").
            // This may be later overridden by vkSetDebugUtilsObjectName.
            data.name.push_str(&ep_name);
        }
        data
    }

    unsafe fn create_compute_pipeline_data(&self, create_info: &VkComputePipelineCreateInfo) -> PipelineData {
        let mut data = PipelineData::default();
        let stage = &create_info.stage;
        if let Some(smci) = find_next::<VkShaderModuleCreateInfo>(stage) {
            let ep_name = CStr::from_ptr(stage.pName).to_string_lossy();
            let ep = self.register_shader_module(&*smci, &ep_name);
            data.compute = Some(ep);
            // Same as for graphics pipelines, this may be overridden by vkSetDebugUtilsObjectName.
            data.name = ep_name.into_owned();
        }
        data
    }

    pub unsafe fn hook_create_graphics_pipelines(
        &self,
        device: VkDevice,
        pipeline_cache: VkPipelineCache,
        create_info_count: u32,
        p_create_infos: *const VkGraphicsPipelineCreateInfo,
        p_allocator: *const VkAllocationCallbacks,
        p_pipelines: *mut VkPipeline,
    ) -> VkResult {
        let result = self.CreateGraphicsPipelines(
            device,
            pipeline_cache,
            create_info_count,
            p_create_infos,
            p_allocator,
            p_pipelines,
        );
        if result < 0 {
            return result;
        }
        let create_infos = from_raw_parts(p_create_infos, create_info_count as usize);
        let pipelines = from_raw_parts(p_pipelines, create_info_count as usize);
        for (i, create_info) in create_infos.iter().enumerate() {
            self.tracked_objects.lock().pipelines.push(pipelines[i]);
            let data = self.create_graphics_pipeline_data(create_info);
            self.set_private_data(pipelines[i], data);
        }
        result
    }

    pub unsafe fn hook_create_compute_pipelines(
        &self,
        device: VkDevice,
        pipeline_cache: VkPipelineCache,
        create_info_count: u32,
        p_create_infos: *const VkComputePipelineCreateInfo,
        p_allocator: *const VkAllocationCallbacks,
        p_pipelines: *mut VkPipeline,
    ) -> VkResult {
        let result = self.CreateComputePipelines(
            device,
            pipeline_cache,
            create_info_count,
            p_create_infos,
            p_allocator,
            p_pipelines,
        );
        if result < 0 {
            return result;
        }
        let create_infos = from_raw_parts(p_create_infos, create_info_count as usize);
        let pipelines = from_raw_parts(p_pipelines, create_info_count as usize);
        for (i, create_info) in create_infos.iter().enumerate() {
            self.tracked_objects.lock().pipelines.push(pipelines[i]);
            let data = self.create_compute_pipeline_data(create_info);
            self.set_private_data(pipelines[i], data);
        }
        result
    }

    pub unsafe fn hook_destroy_pipeline(
        &self,
        device: VkDevice,
        pipeline: VkPipeline,
        p_allocator: *const VkAllocationCallbacks,
    ) {
        let _ = self.take_private_data(pipeline);
        self.DestroyPipeline(device, pipeline, p_allocator);
    }
}
