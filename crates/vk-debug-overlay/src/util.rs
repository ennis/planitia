use std::env;
use vulkan::*;

/// Returns a matching structure in a pNext chain.
pub unsafe fn find_next<N>(prev: &impl VkTaggedStructure) -> Option<*const N>
where
    N: VkTaggedStructure,
{
    let base_in_struct = prev as *const _ as *const VkBaseInStructure;
    let mut p_next = (*base_in_struct).pNext;
    while let Some(base) = p_next.as_ref() {
        if base.s_type == N::STRUCTURE_TYPE {
            return Some(p_next.cast::<N>());
        }
        p_next = base.p_next;
    }
    None
}

/// Returns the value of an environment variable as a boolean flag.
/// The variable is considered true if its value is "1", "true", or "yes".
pub fn env_flag(name: &str) -> bool {
    env::var(name).map(|v| v == "1" || v == "true" || v == "yes").unwrap_or(false)
}
