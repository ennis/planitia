use crate::helper::PrivateData;
use crate::reflection::{EntryPoint, ShaderReflection};
use crate::util::find_next;
use crate::{device_state, DeviceState};
use ash::vk;
use std::slice;

#[derive(Default)]
pub struct PipelineData {
    // Shader reflection information
    vertex_refl: Option<ShaderReflection>,
    fragment_refl: Option<ShaderReflection>,
    mesh_refl: Option<ShaderReflection>,
    task_refl: Option<ShaderReflection>,
    compute_refl: Option<ShaderReflection>,
}

impl PrivateData for PipelineData {
    type Handle = vk::Pipeline;
}

impl DeviceState {
    unsafe fn create_graphics_pipeline_state(&self, create_info: &vk::GraphicsPipelineCreateInfo) -> PipelineData {
        let mut pipeline_state = PipelineData::default();
        let stages = slice::from_raw_parts(create_info.p_stages, create_info.stage_count as usize);
        for stage in stages {
            if let Some(smci) = find_next::<vk::ShaderModuleCreateInfo>(stage) {
                let refl = match stage.stage {
                    vk::ShaderStageFlags::VERTEX => &mut pipeline_state.vertex_refl,
                    vk::ShaderStageFlags::FRAGMENT => &mut pipeline_state.fragment_refl,
                    vk::ShaderStageFlags::MESH_EXT => &mut pipeline_state.mesh_refl,
                    vk::ShaderStageFlags::TASK_EXT => &mut pipeline_state.task_refl,
                    vk::ShaderStageFlags::COMPUTE => &mut pipeline_state.compute_refl,
                    _ => {
                        // TODO warning unsupported stage
                        continue;
                    }
                };

                let spirv = slice::from_raw_parts((*smci).p_code, (*smci).code_size / 4);
                let Ok(reflection) = ShaderReflection::new(spirv) else {
                    eprintln!("failed to generate reflection for shader");
                    continue;
                };
                for ep in reflection.entry_points() {
                    dump_entry_point_reflection(ep);
                }
                *refl = Some(reflection);
            }
        }
        pipeline_state
    }

    unsafe fn create_compute_pipeline_state(&self, create_info: &vk::ComputePipelineCreateInfo) -> PipelineData {
        let mut pipeline_state = PipelineData::default();
        let stage = &create_info.stage;
        if let Some(smci) = find_next::<vk::ShaderModuleCreateInfo>(stage) {
            let spirv = slice::from_raw_parts((*smci).p_code, (*smci).code_size / 4);
            let Ok(reflection) = ShaderReflection::new(spirv) else {
                eprintln!("failed to generate reflection for shader");
                return pipeline_state;
            };
            pipeline_state.compute_refl = Some(reflection);
        }
        pipeline_state
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

        let create_infos = slice::from_raw_parts(p_create_infos, create_info_count as usize);
        let pipelines = slice::from_raw_parts(p_pipelines, create_info_count as usize);

        for (i, create_info) in create_infos.iter().enumerate() {
            self.tracked_resources.lock().unwrap().pipelines.push(pipelines[i]);
            let pipeline_state = self.create_graphics_pipeline_state(create_info);
            self.set_private_data(pipelines[i], pipeline_state);
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

        let create_infos = slice::from_raw_parts(p_create_infos, create_info_count as usize);
        let pipelines = slice::from_raw_parts(p_pipelines, create_info_count as usize);

        for (i, create_info) in create_infos.iter().enumerate() {
            self.tracked_resources.lock().unwrap().pipelines.push(pipelines[i]);
            let pipeline_state = self.create_compute_pipeline_state(create_info);
            self.set_private_data(pipelines[i], pipeline_state);
        }

        vk::Result::SUCCESS
    }

    pub unsafe fn hook_destroy_pipeline(
        &self,
        device: vk::Device,
        pipeline: vk::Pipeline,
        p_allocator: *const vk::AllocationCallbacks<'_>,
    ) {
        (self.fp_v1_0().destroy_pipeline)(device, pipeline, p_allocator);
    }
}

fn dump_entry_point_reflection(entry_point: &EntryPoint) {
    eprintln!("Entry point: {}", entry_point.name);
    for param in entry_point.params {
        eprintln!("  Param: {} (type: {:#?})", param.name, param.ty);
    }
}
