use std::slice;
use ash::vk;
use ash::vk::PFN_vkCreateGraphicsPipelines;
use crate::{state, layer_fn, PipelineData};
use crate::util::find_next;

layer_fn! {
    #[proc(PFN_vkCreateGraphicsPipelines)]
    fn layer_vkCreateGraphicsPipelines(
        device: vk::Device,
        pipeline_cache: vk::PipelineCache,
        create_info_count: u32,
        p_create_infos: *const vk::GraphicsPipelineCreateInfo<'_>,
        p_allocator: *const vk::AllocationCallbacks<'_>,
        p_pipelines: *mut vk::Pipeline,
    ) -> vk::Result {
        eprintln!("[planitia-layer] vkCreateGraphicsPipelines {:?} count={}", device, create_info_count);

        // TODO:
        // - read shader module data, extract SPIR-V
        // - extract types
        // - create the debug overlay pipeline
        // - hook vkQueuePresent
        //    - before present: wait on the semaphores specified in vkQueuePresent
        //          - for now we can rely on those being binary semaphores only
        //    - vkDeviceWaitIdle, just to be sure
        //    - extract swapchain image
        //    - render overlay
        //    - signal semaphore
        //    - vkQueuePresent with the other semaphore

        let create_infos = slice::from_raw_parts(p_create_infos, create_info_count as usize);

        for create_info in create_infos {
            let stages = slice::from_raw_parts(create_info.p_stages, create_info.stage_count as usize);
            for stage in stages {
                if let Some(smci) = find_next::<vk::ShaderModuleCreateInfo>(stage) {
                    // Process the shader module create info
                    eprintln!(
                        "[planitia-layer] Found ShaderModuleCreateInfo for stage {:?}, code size: {} bytes",
                        stage.stage,
                        (*smci).code_size
                    );
                }
            }
        }

        let result = (state(device).fp_v1_0().create_graphics_pipelines)(
            device,
            pipeline_cache,
            create_info_count,
            p_create_infos,
            p_allocator,
            p_pipelines,
        );

        if result == vk::Result::SUCCESS {
            let pipelines = slice::from_raw_parts(p_pipelines, create_info_count as usize);
            for i in 0..create_info_count {
                let dd = state(device);
                dd.tracked_resources.lock().unwrap().pipelines.push(PipelineData { pipeline: pipelines[i as usize] });
            }
        }

        result
    }
}