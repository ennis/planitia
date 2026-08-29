use crate::Device;
use crate::debugger::ModuleId;
use crate::helper::HasPrivateData;
use crate::spirv::{EntryPointId, EntryPointInfo, Module};
use crate::util::find_next;
use ash::vk;
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

impl HasPrivateData for vk::Pipeline {
    type PrivateData = PipelineData;
}

impl Device {
    fn register_shader_module(&self, smci: &vk::ShaderModuleCreateInfo, entry_point: &str) -> ShaderStageInfo {
        let mut mods = self.modules.lock();
        let spirv = unsafe { from_raw_parts((*smci).p_code, (*smci).code_size / 4) };
        let module = Module::parse(spirv).unwrap();
        let entry_point_id = module.find_entry_point(&entry_point).unwrap();
        let module_id = mods.insert(module);
        let info = ShaderStageInfo { module: module_id, entry_point: entry_point_id };
        info
    }

    unsafe fn create_graphics_pipeline_data(&self, create_info: &vk::GraphicsPipelineCreateInfo) -> PipelineData {
        let mut data = PipelineData::default();
        let stages = from_raw_parts(create_info.p_stages, create_info.stage_count as usize);
        for (i, stage) in stages.iter().enumerate() {
            let ep_name = CStr::from_ptr(stage.p_name).to_string_lossy();
            if let Some(smci) = find_next::<vk::ShaderModuleCreateInfo>(stage) {
                let ep = self.register_shader_module(&*smci, &ep_name);
                match stage.stage {
                    vk::ShaderStageFlags::VERTEX => data.vertex = Some(ep),
                    vk::ShaderStageFlags::FRAGMENT => data.fragment = Some(ep),
                    vk::ShaderStageFlags::MESH_EXT => data.mesh = Some(ep),
                    vk::ShaderStageFlags::TASK_EXT => data.task = Some(ep),
                    vk::ShaderStageFlags::COMPUTE => data.compute = Some(ep),
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

    unsafe fn create_compute_pipeline_data(&self, create_info: &vk::ComputePipelineCreateInfo) -> PipelineData {
        let mut data = PipelineData::default();
        let stage = &create_info.stage;
        if let Some(smci) = find_next::<vk::ShaderModuleCreateInfo>(stage) {
            let ep_name = CStr::from_ptr(stage.p_name).to_string_lossy();
            let ep = self.register_shader_module(&*smci, &ep_name);
            data.compute = Some(ep);
            // Same as for graphics pipelines, this may be overridden by vkSetDebugUtilsObjectName.
            data.name = ep_name.into_owned();
        }
        data
    }

    pub unsafe fn hook_create_graphics_pipelines(
        &self,
        device: vk::Device,
        pipeline_cache: vk::PipelineCache,
        create_info_count: u32,
        p_create_infos: *const vk::GraphicsPipelineCreateInfo<'_>,
        p_allocator: *const vk::AllocationCallbacks<'_>,
        p_pipelines: *mut vk::Pipeline,
    ) -> vk::Result {
        let r = (self.fp_v1_0().create_graphics_pipelines)(
            device,
            pipeline_cache,
            create_info_count,
            p_create_infos,
            p_allocator,
            p_pipelines,
        );

        if r != vk::Result::SUCCESS {
            return r;
        }

        let create_infos = from_raw_parts(p_create_infos, create_info_count as usize);
        let pipelines = from_raw_parts(p_pipelines, create_info_count as usize);

        for (i, create_info) in create_infos.iter().enumerate() {
            self.tracked_objects.lock().pipelines.push(pipelines[i]);
            let data = self.create_graphics_pipeline_data(create_info);
            self.set_private_data(pipelines[i], data);
        }

        vk::Result::SUCCESS
    }

    pub unsafe fn hook_create_compute_pipelines(
        &self,
        device: vk::Device,
        pipeline_cache: vk::PipelineCache,
        create_info_count: u32,
        p_create_infos: *const vk::ComputePipelineCreateInfo<'_>,
        p_allocator: *const vk::AllocationCallbacks<'_>,
        p_pipelines: *mut vk::Pipeline,
    ) -> vk::Result {
        let r = (self.fp_v1_0().create_compute_pipelines)(
            device,
            pipeline_cache,
            create_info_count,
            p_create_infos,
            p_allocator,
            p_pipelines,
        );

        if r != vk::Result::SUCCESS {
            return r;
        }

        let create_infos = from_raw_parts(p_create_infos, create_info_count as usize);
        let pipelines = from_raw_parts(p_pipelines, create_info_count as usize);

        for (i, create_info) in create_infos.iter().enumerate() {
            self.tracked_objects.lock().pipelines.push(pipelines[i]);
            let data = self.create_compute_pipeline_data(create_info);
            self.set_private_data(pipelines[i], data);
        }

        vk::Result::SUCCESS
    }

    pub unsafe fn hook_destroy_pipeline(
        &self,
        device: vk::Device,
        pipeline: vk::Pipeline,
        p_allocator: *const vk::AllocationCallbacks<'_>,
    ) {
        let _ = self.take_private_data(pipeline);
        (self.fp_v1_0().destroy_pipeline)(device, pipeline, p_allocator);
    }
}
