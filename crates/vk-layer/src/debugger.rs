use crate::Device;
use crate::event::EId;
use crate::helper::{Buffer, DeviceHelper, Pipeline, include_bytes_as_u32};
use crate::reflection::{Type, TypeDesc};
use crate::state_tracker::command::{CmdIdx, CmdKey, Command};
use ash::vk;
use parking_lot::Mutex;
use rustc_hash::FxHasher;
use slotmap::{SlotMap, new_key_type};
use std::hash::{Hash, Hasher};
use std::mem::discriminant;
use std::sync::atomic::Ordering::Relaxed;
use std::{fmt, mem, ptr};

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
enum IndexVarKind {
    Constant(usize),
    Varying,
    // The index of the expression that the variable is bound to (if any).
    // The expression should resolve to a pointer to an integer.
    Indirect(usize),
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
struct IndexVar {
    kind: IndexVarKind,
}

// Type of indirection relative to a base pointer expr
#[derive(Clone, Debug)]
enum Access {
    // Dereferences a pointer
    Load,
    // Given a pointer expr to an array (resp. struct), returns a pointer to the element at the specified index (resp. field)
    Index(usize),
    //
    VarIndex(usize),
}

//
#[repr(u8)]
#[derive(Copy, Clone, Default, Eq, PartialEq)]
enum AccessKind {
    #[default]
    UniformOffset = 0,
    VaryingOffset,
    Indirect,
}

// GPU repr of `Access`
#[repr(C)]
#[derive(Copy, Clone, Default)]
struct ResolvedAccess {
    kind: AccessKind,
    offset_or_stride: u32,
}

//
#[derive(Copy, Clone, Default)]
pub struct DeviceAccessChain {
    base: vk::DeviceAddress,
    src_ac_count: u32,
    src_ac: [ResolvedAccess; MAX_ACCESS_COUNT],
}

#[derive(Clone)]
pub enum DeviceAccessChainBase {
    // Constant device address
    Address(vk::DeviceAddress),
    // Address in push data
    PushDataAddress(usize),
}

#[derive(Clone)]
pub struct PlaceResolveResult {
    ty: Type,
    kind: PlaceKind,
}

#[derive(Clone)]
pub enum PlaceKind {
    // The place resolves to a location in push data
    PushData(usize),
    // The place resolves to a location in device memory, accessible with the given access chain
    DeviceMemory { base: vk::DeviceAddress, access_chain: Vec<ResolvedAccess> },
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
enum IndexType {
    Constant(usize),
    Var(usize),
}

pub struct ResolveCtx<'a> {
    push_data: &'a [u8],
}

#[derive(Clone, Debug)]
pub enum PlaceExpr {
    PushData(Type),
    FieldOrIndex { parent: Box<PlaceExpr>, index: IndexType },
    Deref(Box<PlaceExpr>),
}

impl PartialEq for PlaceExpr {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (PlaceExpr::PushData(ty1), PlaceExpr::PushData(ty2)) => ptr::eq(ty1, ty2),
            (PlaceExpr::FieldOrIndex { parent: p1, index: i1 }, PlaceExpr::FieldOrIndex { parent: p2, index: i2 }) => {
                p1 == p2 && i1 == i2
            }
            (PlaceExpr::Deref(p1), PlaceExpr::Deref(p2)) => p1 == p2,
            _ => false,
        }
    }
}

impl Hash for PlaceExpr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            PlaceExpr::PushData(ty) => {
                state.write_u8(0);
                state.write_usize(*ty as *const _ as usize);
            }
            PlaceExpr::FieldOrIndex { parent, index } => {
                state.write_u8(1);
                parent.hash(state);
                index.hash(state);
            }
            PlaceExpr::Deref(parent) => {
                state.write_u8(2);
                parent.hash(state);
            }
        }
    }
}

impl PlaceExpr {
    fn resolve(&self, ctx: &ResolveCtx) -> PlaceResolveResult {
        match self {
            PlaceExpr::PushData(ty) => PlaceResolveResult { ty: *ty, kind: PlaceKind::PushData(0) },
            PlaceExpr::FieldOrIndex { parent, index } => {
                let mut parent = parent.resolve(ctx);
                match index {
                    IndexType::Constant(index) => {
                        let offset = parent.ty.field_or_element_offset(*index).unwrap();
                        let result_ty = parent.ty.field_or_element_type(*index).unwrap();
                        parent.ty = result_ty;
                        match parent.kind {
                            PlaceKind::PushData(ref mut push_offset) => {
                                *push_offset += offset;
                            }
                            PlaceKind::DeviceMemory { ref mut access_chain, .. } => {
                                if let Some(last_access) = access_chain.last_mut()
                                    && last_access.kind == AccessKind::UniformOffset
                                {
                                    // merge with previous uniform offset
                                    last_access.offset_or_stride += offset as u32;
                                } else {
                                    access_chain.push(ResolvedAccess {
                                        kind: AccessKind::UniformOffset,
                                        offset_or_stride: offset as u32,
                                    });
                                }
                            }
                        }
                        parent
                    }
                    IndexType::Var(var) => {
                        todo!("index variable resolution");
                    }
                }
            }
            PlaceExpr::Deref(parent) => {
                let parent = parent.resolve(ctx);
                match parent.kind {
                    PlaceKind::PushData(push_offset) => {
                        let result_ty = parent.ty.field_or_element_type(0).unwrap();
                        // read the base address in push data
                        let base =
                            unsafe { (ctx.push_data.as_ptr().add(push_offset) as *const vk::DeviceAddress).read() };
                        PlaceResolveResult {
                            ty: result_ty,
                            kind: PlaceKind::DeviceMemory { base, access_chain: vec![] },
                        }
                    }
                    PlaceKind::DeviceMemory { base, mut access_chain } => {
                        let result_ty = parent.ty.field_or_element_type(0).unwrap();
                        access_chain.push(ResolvedAccess { kind: AccessKind::Indirect, offset_or_stride: 0 });
                        PlaceResolveResult { ty: result_ty, kind: PlaceKind::DeviceMemory { base, access_chain } }
                    }
                }
            }
        }
    }

    // Resolves the type of the expression.
    fn ty(&self) -> Type {
        match self {
            PlaceExpr::PushData(ty) => *ty,
            PlaceExpr::FieldOrIndex { parent, index } => {
                let parent_ty = parent.ty();
                match index {
                    IndexType::Constant(index) => parent_ty.field_or_element_type(*index).unwrap(),
                    IndexType::Var(var) => {
                        todo!("index variable resolution");
                    }
                }
            }
            PlaceExpr::Deref(parent) => {
                let parent_ty = parent.ty();
                parent_ty.field_or_element_type(0).unwrap()
            }
        }
    }
}

impl fmt::Display for PlaceExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PlaceExpr::PushData(_ty) => {
                write!(f, "(push data)")
            }
            PlaceExpr::FieldOrIndex { index, parent } => {
                // FIXME this is accidentally O(n2) in number of components in the place
                let ty = parent.ty();
                match ty {
                    TypeDesc::Array { .. } => {
                        write!(
                            f,
                            "{}[{}]",
                            parent,
                            match index {
                                IndexType::Constant(i) => i.to_string(),
                                IndexType::Var(v) => format!("var({})", v),
                            }
                        )
                    }
                    TypeDesc::RuntimeArray { .. } => {
                        write!(
                            f,
                            "{}[{}]",
                            parent,
                            match index {
                                IndexType::Constant(i) => i.to_string(),
                                IndexType::Var(v) => format!("var({})", v),
                            }
                        )
                    }
                    TypeDesc::Struct(sty) => {
                        let i = match index {
                            IndexType::Constant(i) => *i,
                            _ => unreachable!("dynamic field expression"),
                        };
                        write!(f, "{}.{}", parent, sty.fields[i].name)
                    }
                    _ => {
                        write!(f, "<unsupported>")
                    }
                }
            }
            PlaceExpr::Deref(parent) => {
                write!(f, "(*{})", parent)
            }
        }
    }
}

#[derive(Clone, Hash)]
pub struct Query {
    indexvars: Vec<IndexVar>,
    place: PlaceExpr,
}

impl Query {
    /// Creates a new query from the specified expression.
    pub fn new(place: PlaceExpr) -> Query {
        Query { place, indexvars: Vec::new() }
    }

    /// Returns the type of the query.
    pub fn result_ty(&self) -> Type {
        self.place.ty()
    }

    /// Creates a new query by appending a field or array index expression to the query.
    pub fn with_index(self, index: usize) -> Query {
        Query {
            place: PlaceExpr::FieldOrIndex { parent: Box::new(self.place), index: IndexType::Constant(index) },
            indexvars: self.indexvars,
        }
    }
}

impl fmt::Display for Query {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.place)
    }
}

//--------------------------------------------------------------------------------------------------

static COPY_1D_SHADER: &[u32] = include_bytes_as_u32!("copy.spv");
const COPY_WORKGROUP_SIZE: u32 = 32;
const MAX_ACCESS_COUNT: usize = 16;

#[repr(C)]
#[derive(Copy, Clone, Default)]
struct CopyFromAccessChainParams {
    src_base: vk::DeviceAddress,
    dst: vk::DeviceAddress,
    size: u32,
    src_ac_count: u32,
    src_ac: [ResolvedAccess; MAX_ACCESS_COUNT],
}

pub struct DebuggerResources {
    copy_from_access_chain_1d: Pipeline,
}

impl DebuggerResources {
    pub unsafe fn new(device_helper: &DeviceHelper) -> DebuggerResources {
        let copy_from_access_chain_1d = device_helper.create_compute_pipeline_helper(
            COPY_1D_SHADER,
            c"copy_from_access_chain_1d",
            &[],
            size_of::<CopyFromAccessChainParams>(),
        );
        DebuggerResources { copy_from_access_chain_1d }
    }
}

/// Debugger watch: a query that should run before/after a specified command.
pub struct Watch {
    // Hash of (event_id, stage, query)
    hash: u64,
    // Event ID
    eid: EId,
    // Which shader stage we are looking at
    stage: vk::ShaderStageFlags,
    // The query (on push data) to record
    query: Query,
    // Resolved type of the query
    ty: Type,
    // Size in bytes of the data to record.
    // Not necessarily sizeof(ty) because we can request part of an array
    size: usize,
    // Result buffer for holding the result of the query
    result: Buffer,
    // Whether this watch is temporary (removed if not read in the last frame)
    transient: bool,
    // Last frame in which the watch was requested, for transient watches.
    last_request: u64,
    // Last frame in which the watch matched an existing command
    last_match: u64,
}

new_key_type! {
    pub struct WatchId;
}

pub struct Debugger {
    pub watches: SlotMap<WatchId, Watch>,
}

impl Debugger {
    pub fn new() -> Debugger {
        Debugger { watches: SlotMap::with_key() }
    }
}

impl Device {
    unsafe fn read_query_async(
        &self,
        query: &Query,
        cmd_buf: vk::CommandBuffer,
        push_data: &[u8],
        size: usize,
        result_buffer: &Buffer,
        buffer_offset: usize,
    ) {
        let place = query.place.resolve(&ResolveCtx { push_data });
        match place.kind {
            PlaceKind::PushData(offset) => {
                // copy directly from push data
                ptr::copy_nonoverlapping(
                    push_data.as_ptr().add(offset),
                    result_buffer.ptr.add(buffer_offset) as *mut u8,
                    size,
                );
            }
            PlaceKind::DeviceMemory { base, access_chain } => {
                // the query needs to read device memory
                let mut params = CopyFromAccessChainParams {
                    src_base: base,
                    dst: result_buffer.device_address + buffer_offset as u64,
                    size: size as u32,
                    src_ac_count: access_chain.len() as u32,
                    src_ac: [Default::default(); 16],
                };
                for (i, ac) in access_chain.iter().enumerate() {
                    params.src_ac[i] = *ac;
                }

                // dispatch one thread per byte to read
                let n_workgroups = size.div_ceil(COPY_WORKGROUP_SIZE as usize) as u32;

                self.cmd_bind_pipeline(
                    cmd_buf,
                    vk::PipelineBindPoint::COMPUTE,
                    self.debugger_resources.copy_from_access_chain_1d.pipeline,
                );
                self.push_constants_helper(
                    cmd_buf,
                    self.debugger_resources.copy_from_access_chain_1d.pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    &params,
                );
                self.cmd_dispatch(cmd_buf, n_workgroups, 1, 1);
            }
        }
    }

    pub fn add_watch(&self, eid: EId, stage: vk::ShaderStageFlags, query: Query) -> WatchId {
        self.add_watch_inner(eid, stage, query, false)
    }

    pub fn add_temporary_watch(&self, eid: EId, stage: vk::ShaderStageFlags, query: Query) {
        self.add_watch_inner(eid, stage, query, true);
    }

    fn add_watch_inner(&self, eid: EId, stage: vk::ShaderStageFlags, query: Query, transient: bool) -> WatchId {
        let frame_index = self.get_frame_index();
        let mut dbg = self.debugger.lock();

        // compute query hash
        let mut h = FxHasher::default();
        (eid, stage, &query).hash(&mut h);
        let hash = h.finish();

        // find a matching, existing watch by query hash
        for (_id, watch) in dbg.watches.iter_mut() {
            if watch.hash == hash {
                // found a matching watch, mark it as live, return its id
                watch.last_match = frame_index;
                return _id;
            }
        }

        // no match, create a new watch
        let ty = query.result_ty();
        let byte_size = ty.byte_size().unwrap();
        let result = unsafe {
            self.create_buffer_helper(
                vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::STORAGE_BUFFER,
                byte_size,
                None,
            )
        };
        dbg.watches.insert(Watch {
            hash,
            eid,
            stage,
            query,
            size: byte_size,
            ty,
            result,
            transient,
            last_match: 0,
            last_request: frame_index,
        })
    }

    pub fn retire_transient_watches(&self) {
        let frame_index = self.get_frame_index();
        let mut dbg = self.debugger.lock();
        dbg.watches.retain(|_id, watch| {
            if watch.transient && watch.last_request < frame_index {
                // no request for this transient watch in this frame, delete it
                unsafe { self.destroy_buffer_helper(mem::take(&mut watch.result)) };
                false
            } else {
                true
            }
        });
    }

    /// Called after every command to update relevant watches.
    pub fn update_watches_for_command(&self, command: &Command) {
        let dbg = self.debugger.lock();
        for watch in dbg.watches.values() {
            if watch.eid == command.eid {
                unsafe {
                    self.read_query_async(
                        &watch.query,
                        command.cmd_buf,
                        command.push.as_slice(),
                        watch.size,
                        &watch.result,
                        0,
                    );
                }
            }
        }
    }

    pub fn read_watch(&self, id: WatchId) -> Option<Vec<u8>> {
        let frame_index = self.frame_index.load(Relaxed);
        let dbg = self.debugger.lock();
        let watch = dbg.watches.get(id)?;
        if watch.last_match == frame_index {
            // valid data
            let mut result = vec![0u8; watch.size];
            unsafe {
                ptr::copy_nonoverlapping(watch.result.ptr as *const u8, result.as_mut_ptr(), watch.size);
            }
            Some(result)
        } else {
            None
        }
    }

    //pub fn read_value(&self, cmd_id: CommandId, query: Query) -> Option<Vec<u8>> {
    //    todo!()
    //}
}
