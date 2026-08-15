use ash::vk;

/// Returns a matching structure in a pNext chain.
pub unsafe fn find_next<N>(prev: &impl vk::TaggedStructure) -> Option<*const N>
where
    N: vk::TaggedStructure,
{
    let base_in_struct = prev as *const _ as *const vk::BaseInStructure;
    let mut p_next = (*base_in_struct).p_next;
    while let Some(base) = p_next.as_ref() {
        if base.s_type == N::STRUCTURE_TYPE {
            return Some(p_next.cast::<N>());
        }
        p_next = base.p_next;
    }
    None
}