use std::alloc::Layout;
use crate::helper::{include_bytes_as_u32, Buffer, DeviceHelper, Pipeline};
use crate::reflection::{PointerType, ScalarType, TypeDesc};
use crate::state_tracker::command::{CmdKey, Command};
use ash::vk;
use spirv_headers::StorageClass;
use crate::DeviceState;

#[derive(Copy, Clone, Debug)]
enum IndexVarKind {
    Constant(usize),
    Varying,
    // The index of the expression that the variable is bound to (if any).
    // The expression should resolve to a pointer to an integer.
    Indirect(usize),
}

#[derive(Copy, Clone, Debug)]
struct IndexVar {
    kind: IndexVarKind,
}

#[derive(Clone, Debug)]
enum AccessKind {
    // Dereferences a pointer
    Load,
    // Given a pointer expr to a struct, returns a pointer to the specified field
    Field(usize),
    // Given a pointer expr to an array, returns a pointer to the element at the specified index
    Index(usize),
}

// TODO:
// - don't need a type on each access node, it can be inferred from the root type
// - need a way to construct new TypeDescs
// - AccessKind::Load should be AccessKind::Indirect
// -

// An expression evaluating to a pointer to a variable.
#[derive(Clone, Debug)]
struct Access {
    ty: &'static TypeDesc<'static>,
    kind: AccessKind,
}

#[derive(Copy, Clone, Default)]
pub struct ResolvedQuery {
    push_offset: usize,
    src_base: vk::DeviceAddress,
    src_ac_count: u32,
    src_ac: [AccessGpu; MAX_ACCESS_COUNT],
}

impl ResolvedQuery {
    fn append(&mut self, push_data: &[u8], ac: &Access, query: &Query) {
        let access_kind;
        let offset_or_stride;
        match ac.kind {
            AccessKind::Load => {
                access_kind = AccessKindGpu::Indirect;
                offset_or_stride = 0;
            }
            AccessKind::Field(index) => {
                access_kind = AccessKindGpu::UniformOffset;
                offset_or_stride = ac.ty.field_or_element_offset(index).unwrap();
            }
            AccessKind::Index(index_var_id) => {
                let stride = ac.ty.stride().unwrap();
                match query.indexvars[index_var_id].kind {
                    IndexVarKind::Constant(index) => {
                        access_kind = AccessKindGpu::UniformOffset;
                        offset_or_stride = stride * index;
                    }
                    IndexVarKind::Varying => {
                        access_kind = AccessKindGpu::VaryingOffset;
                        offset_or_stride = stride;
                    }
                    IndexVarKind::Indirect(_) => {
                        todo!("indirect index variable");
                    }
                }
            }
        };

        if self.src_ac_count == 0 {
            match access_kind {
                AccessKindGpu::UniformOffset => {
                    self.push_offset += offset_or_stride;
                }
                AccessKindGpu::Indirect => {
                    self.src_base = unsafe { (push_data.as_ptr().add(self.push_offset) as *const u64).read() };
                }
                _ => {}
            }
        } else {
            let i = self.src_ac_count as usize;
            self.src_ac[i].offset_or_stride = offset_or_stride as u32;
            self.src_ac[i].kind = match access_kind {
                AccessKindGpu::UniformOffset => AccessKindGpu::UniformOffset,
                AccessKindGpu::VaryingOffset => AccessKindGpu::VaryingOffset,
                AccessKindGpu::Indirect => AccessKindGpu::Indirect,
            };
            self.src_ac_count += 1;
        }
    }
}

pub struct Query {
    indexvars: Vec<IndexVar>,
    comp: Vec<Access>,
}

//--------------------------------------------------------------------------------------------------
static COPY_1D_SHADER: &[u32] = include_bytes_as_u32!("copy.spv");

const MAX_ACCESS_COUNT: usize = 16;
const COPY_WORKGROUP_SIZE: u32 = 32;

#[repr(u8)]
#[derive(Copy, Clone, Default)]
enum AccessKindGpu {
    #[default]
    UniformOffset = 0,
    VaryingOffset,
    Indirect,
}

#[repr(C)]
#[derive(Copy, Clone, Default)]
struct AccessGpu {
    kind: AccessKindGpu,
    offset_or_stride: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Default)]
struct CopyFromAccessChainParams {
    src_base: vk::DeviceAddress,
    dst: vk::DeviceAddress,
    size: u32,
    //stride: u32,
    src_ac_count: u32,
    src_ac: [AccessGpu; MAX_ACCESS_COUNT],
}

/// Debugger probes: a query that should run before/after a specified command.
pub struct Probe {
    // Identifies the command to record.
    cmd: CmdKey,
    // Whether to sample before (true) or after (false) the command.
    before: bool,
    // The query to record
    query: Query,
    // Offset and size of the data to record
    offset: usize,
    size: usize,
    // Query result buffer
    query_result: Buffer,
}

pub struct Debugger {
    copy_from_access_chain_1d: Pipeline,
    probes: Vec<Probe>,
    //results: Vec<Buffer>,
}

impl Debugger {
    pub unsafe fn new(device_helper: &DeviceHelper) -> Debugger {
        let copy_from_access_chain_1d = device_helper.create_compute_pipeline_helper(
            COPY_1D_SHADER,
            c"copy_from_access_chain_1d",
            &[],
            size_of::<CopyFromAccessChainParams>(),
        );
        Debugger { copy_from_access_chain_1d, probes: Vec::new() }
    }

    unsafe fn resolve_query(
        &self,
        dh: &DeviceHelper,
        push_data: &[u8],
        query: &Query,
        byte_size: usize,
        cmd_buf: vk::CommandBuffer,
        out_buffer: vk::DeviceAddress,
    ) {
        let mut rq = ResolvedQuery { push_offset: 0, src_base: 0, src_ac_count: 0, src_ac: Default::default() };
        for ac in query.comp.iter() {
            rq.append(push_data, ac, query);
        }

        if rq.src_base != 0 {
            // the query needs to read device memory
            // dispatch one thread per element to read
            let n_workgroups = byte_size.div_ceil(COPY_WORKGROUP_SIZE as usize) as u32;

            dh.cmd_bind_pipeline(cmd_buf, vk::PipelineBindPoint::COMPUTE, self.copy_from_access_chain_1d.pipeline);
            dh.push_constants_helper(
                cmd_buf,
                self.copy_from_access_chain_1d.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                &CopyFromAccessChainParams {
                    src_base: rq.src_base,
                    dst: out_buffer,
                    size: byte_size as u32,
                    //stride: rq.stride,
                    src_ac_count: rq.src_ac_count,
                    src_ac: rq.src_ac,
                },
            );
            dh.cmd_dispatch(cmd_buf, n_workgroups, 1, 1);

        } else {
            // TODO read directly from push data buffer
        }
    }
}

impl DeviceState {
    pub fn process_probes_for_command(&self, command: &mut Command, cmd_buf: vk::CommandBuffer) {
        let mut dbg = self.debugger.lock();
        static DATA: TypeDesc<'static> = TypeDesc::Scalar(ScalarType::U64);
        static INNER: TypeDesc<'static> = TypeDesc::Pointer(PointerType::new(
            StorageClass::PhysicalStorageBuffer,
            &DATA,
        ));
        /*static TY: TypeDesc<'static> = const {
            TypeDesc::Pointer(PointerType::new(
                StorageClass::PushConstant,
                const {
                    &INNER
                },
            ))
        };*/
        let query = Query {
            indexvars: vec![],
            comp: vec![Access {
                ty: &INNER,
                kind: AccessKind::Load, // load ptr to storage
            }],
        };
        unsafe {
            let mut bump = self.bump.lock();
            let query_results = bump.alloc(self, Layout::from_size_align(8, 8).unwrap());
            dbg.resolve_query(self, command.push.as_slice(), &query, 8, cmd_buf, query_results.dev_addr);
            command.readback = Some(query_results);
        }
    }

    pub fn handle_probes_after_render_pass(&self, cmd_buf: vk::CommandBuffer, commands: &mut [Command]) {
        for cmd in commands {
            self.process_probes_for_command(cmd, cmd_buf);
        }
    }

    /*
    pub fn add_probe(&self, cmd_key: &CmdKey, query: Query, offset: usize, size: usize) {
        let mut dbg = self.debugger.lock();
        dbg.probes.push(Probe {
            cmd: cmd_key.clone(),
            before: false,
            query,
            offset,
            size,
            query_result: Buffer::default(),
        });
    }

    pub fn set_probe(&self, index: usize, offset: usize, size: usize) {
        let mut dbg = self.debugger.lock();
        dbg.probes[index].offset = offset;
        dbg.probes[index].size = size;
    }

    pub fn remove_probe(&self, index: usize) {
        let mut dbg = self.debugger.lock();
        dbg.probes.remove(index);
    }*/
}
