use crate::generated::*;

pub trait Handle {
    const TYPE: VkObjectType;
}

macro_rules! impl_handle {
    ($ty:ty, $object_type:expr) => {
        impl Handle for $ty {
            const TYPE: VkObjectType = $object_type;
        }
    };
}
impl_handle!(VkInstance, VK_OBJECT_TYPE_INSTANCE);
impl_handle!(VkPhysicalDevice, VK_OBJECT_TYPE_PHYSICAL_DEVICE);
impl_handle!(VkDevice, VK_OBJECT_TYPE_DEVICE);
impl_handle!(VkQueue, VK_OBJECT_TYPE_QUEUE);
impl_handle!(VkSemaphore, VK_OBJECT_TYPE_SEMAPHORE);
impl_handle!(VkCommandBuffer, VK_OBJECT_TYPE_COMMAND_BUFFER);
impl_handle!(VkFence, VK_OBJECT_TYPE_FENCE);
impl_handle!(VkDeviceMemory, VK_OBJECT_TYPE_DEVICE_MEMORY);
impl_handle!(VkBuffer, VK_OBJECT_TYPE_BUFFER);
impl_handle!(VkImage, VK_OBJECT_TYPE_IMAGE);
impl_handle!(VkEvent, VK_OBJECT_TYPE_EVENT);
impl_handle!(VkQueryPool, VK_OBJECT_TYPE_QUERY_POOL);
impl_handle!(VkBufferView, VK_OBJECT_TYPE_BUFFER_VIEW);
impl_handle!(VkImageView, VK_OBJECT_TYPE_IMAGE_VIEW);
impl_handle!(VkShaderModule, VK_OBJECT_TYPE_SHADER_MODULE);
impl_handle!(VkPipelineCache, VK_OBJECT_TYPE_PIPELINE_CACHE);
impl_handle!(VkPipelineLayout, VK_OBJECT_TYPE_PIPELINE_LAYOUT);
impl_handle!(VkRenderPass, VK_OBJECT_TYPE_RENDER_PASS);
impl_handle!(VkPipeline, VK_OBJECT_TYPE_PIPELINE);
impl_handle!(VkDescriptorSetLayout, VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT);
impl_handle!(VkSampler, VK_OBJECT_TYPE_SAMPLER);
impl_handle!(VkDescriptorPool, VK_OBJECT_TYPE_DESCRIPTOR_POOL);
impl_handle!(VkDescriptorSet, VK_OBJECT_TYPE_DESCRIPTOR_SET);
impl_handle!(VkFramebuffer, VK_OBJECT_TYPE_FRAMEBUFFER);
impl_handle!(VkCommandPool, VK_OBJECT_TYPE_COMMAND_POOL);
