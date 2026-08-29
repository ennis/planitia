use core::fmt;
use crate::Device;
use crate::event::EId;
use crate::helper::{Buffer, DeviceHelper, Pipeline, include_bytes_as_u32};
use crate::state_tracker::command::Command;
use ash::vk;
use rustc_hash::FxHasher;
use slotmap::{SlotMap, new_key_type};
use std::hash::{Hash, Hasher};
use std::ptr;

/*#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
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
    ty: TypeId,
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
    module: &'a Module,
    push_data: &'a [u8],
}

#[derive(Clone, Debug, Hash)]
pub enum PlaceExprKind {
    PushData,
    FieldOrIndex { parent: Box<PlaceExpr>, index: IndexType },
    Deref(Box<PlaceExpr>),
}

impl PartialEq for PlaceExprKind {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (PlaceExprKind::PushData, PlaceExprKind::PushData) => true,
            (
                PlaceExprKind::FieldOrIndex { parent: p1, index: i1 },
                PlaceExprKind::FieldOrIndex { parent: p2, index: i2 },
            ) => p1 == p2 && i1 == i2,
            (PlaceExprKind::Deref(p1), PlaceExprKind::Deref(p2)) => p1 == p2,
            _ => false,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Hash)]
pub struct PlaceExpr {
    pub ty: TypeId,
    pub kind: PlaceExprKind,
}

impl PlaceExpr {
    fn resolve(&self, ctx: &ResolveCtx) -> PlaceKind {
        match &self.kind {
            PlaceExprKind::PushData => PlaceKind::PushData(0),
            PlaceExprKind::FieldOrIndex { parent, index } => {
                let mut pp = parent.resolve(ctx);
                match index {
                    IndexType::Constant(index) => {
                        let offset = ctx.module[parent.ty].field_or_element_offset(*index).unwrap();
                        //let result_ty = ctx.module[parent.ty].field_or_element_type(*index).unwrap();
                        match pp {
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
                        pp
                    }
                    IndexType::Var(var) => {
                        todo!("index variable resolution");
                    }
                }
            }
            PlaceExprKind::Deref(parent) => {
                match parent.resolve(ctx) {
                    PlaceKind::PushData(push_offset) => {
                        //let result_ty = ctx.module[parent.ty].field_or_element_type(0).unwrap();
                        // read the base address in push data
                        let base =
                            unsafe { (ctx.push_data.as_ptr().add(push_offset) as *const vk::DeviceAddress).read() };
                        PlaceKind::DeviceMemory { base, access_chain: vec![] }
                    }
                    PlaceKind::DeviceMemory { base, mut access_chain } => {
                        //let result_ty = ctx.module[parent.ty].field_or_element_type(0).unwrap();
                        access_chain.push(ResolvedAccess { kind: AccessKind::Indirect, offset_or_stride: 0 });
                        PlaceKind::DeviceMemory { base, access_chain }
                    }
                }
            }
        }
    }
}

/// Returns a pretty-printable wrapper for a PlaceExpr.
pub fn pretty_print_place<'a>(module: &'a Module, place: &'a PlaceExpr) -> PrettyPlaceExpr<'a> {
    PrettyPlaceExpr { module, place }
}

pub struct PrettyPlaceExpr<'a> {
    module: &'a Module,
    place: &'a PlaceExpr,
}

impl<'a> fmt::Display for PrettyPlaceExpr<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.place.kind {
            PlaceExprKind::PushData => {
                write!(f, "(push data)")
            }
            PlaceExprKind::FieldOrIndex { index, parent } => match self.module[parent.ty] {
                TypeInfo::Array { .. } => {
                    write!(
                        f,
                        "{}[{}]",
                        pretty_print_place(self.module, &parent),
                        match index {
                            IndexType::Constant(i) => i.to_string(),
                            IndexType::Var(v) => format!("var({})", v),
                        }
                    )
                }
                TypeInfo::RuntimeArray { .. } => {
                    write!(
                        f,
                        "{}[{}]",
                        pretty_print_place(self.module, &parent),
                        match index {
                            IndexType::Constant(i) => i.to_string(),
                            IndexType::Var(v) => format!("var({})", v),
                        }
                    )
                }
                TypeInfo::Struct(sty) => {
                    let i = match index {
                        IndexType::Constant(i) => i,
                        _ => unreachable!("dynamic field expression"),
                    };
                    write!(f, "{}.{}", pretty_print_place(self.module, &parent), sty.fields[i].name)
                }
                _ => {
                    write!(f, "<unsupported>")
                }
            },
            PlaceExprKind::Deref(parent) => {
                write!(f, "(*{})", pretty_print_place(self.module, &parent))
            }
        }
    }
}

/// Debugger data expression.
#[derive(Clone, Hash)]
pub struct Expr {
    /// Unused for now.
    pub indexvars: Vec<IndexVar>,
    pub place: PlaceExpr,
}

impl Expr {
    /// Creates a new query from the specified expression.
    pub fn new(place: PlaceExpr) -> Expr {
        Expr { place, indexvars: Vec::new() }
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.place)
    }
}*/

/// Represents a sequence of pointer indirections from a base address, (e.g. `base->field->field2 ...`).
#[derive(Clone, Hash, Eq, PartialEq)]
pub struct LoadChain {
    // Chain of offsets for each pointer indirection.
    //
    // This establishes a series of addresses (denoted `address[i]`), defined by the following
    // recurrence relation:
    //
    // - `address[0] = <base> + offsets[0]`
    // - `base[N] = *address[N-1]`
    // - `address[N] = base[N] + offsets[N]`
    pub offsets: Vec<usize>,
}

impl LoadChain {
    pub fn new() -> LoadChain {
        LoadChain { offsets: vec![0] }
    }

    /// Pushes a new indirection on the load chain.
    ///
    /// Concretely, if this load chain represents some address `ADDR`,
    /// then after this function it will point to the address at `*(ADDR + offset)`
    pub fn deref_at(&mut self, offset: usize) {
        let len = self.offsets.len() - 1;
        self.offsets[len] = offset;
        self.offsets.push(0);
    }
}

impl Default for LoadChain {
    fn default() -> Self {
        LoadChain::new()
    }
}

impl fmt::Debug for LoadChain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[base")?;
        for (i,offset) in self.offsets.iter().enumerate() {
            if i > 0 {
                write!(f, "->+0x{:x}", offset)?;
            } else {
                write!(f, "+0x{:x}", offset)?;
            }
        }
        write!(f, "]")?;
        Ok(())
    }
}

//--------------------------------------------------------------------------------------------------

static COPY_1D_SHADER: &[u32] = include_bytes_as_u32!("copy.spv");
const COPY_WORKGROUP_SIZE: u32 = 32;
const MAX_INDIRECTIONS: usize = 8;

#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
struct CopyIndirect1DParams {
    base: vk::DeviceAddress,
    dst: vk::DeviceAddress,
    byte_size: u32,
    count: u32,
    offset: [u32; MAX_INDIRECTIONS],
}

pub struct DebuggerResources {
    copy_indirect_1d: Pipeline,
}

impl DebuggerResources {
    pub unsafe fn new(device_helper: &DeviceHelper) -> DebuggerResources {
        let copy_indirect_1d = device_helper.create_compute_pipeline_helper(
            COPY_1D_SHADER,
            c"copy_indirect_1d",
            &[],
            size_of::<CopyIndirect1DParams>(),
        );
        DebuggerResources { copy_indirect_1d }
    }
}

/// Debugger watch: a query that should run before/after a specified command.
pub struct Watch {
    hash: u64,                   // Hash of (event_id, stage, query)
    eid: EId,                    // Event ID
    stage: vk::ShaderStageFlags, // Which shader stage we are looking at
    load_chain: LoadChain,       // Load chain to the data being inspected
    size: usize, // Size in bytes of the data to record. Not necessarily sizeof(ty) because we can request part of an array
    result: Option<Buffer>, // Result buffer for holding the result of the query
    transient: bool, // Whether this watch is temporary (removed if not read in the last frame)
    last_request: u64, // Last frame in which the watch was requested, for transient watches.
    last_match: u64, // Last frame in which the watch matched an existing command
}

new_key_type! {
    pub struct WatchId;
    pub struct ModuleId;
}

pub struct Debugger {
    pub watches: SlotMap<WatchId, Watch>,
}

impl Debugger {
    pub fn new() -> Debugger {
        Debugger { watches: SlotMap::with_key() }
    }
}

/// Helper context to add watches scoped to a specific command and shader stage.
pub struct DebuggerContext<'a> {
    pub dbg: &'a mut Debugger,
    pub eid: EId,
    pub stage: vk::ShaderStageFlags,
    pub frame_index: u64,
}

impl<'a> DebuggerContext<'a> {
    pub fn new(
        device: &'a Device,
        dbg: &'a mut Debugger,
        eid: EId,
        stage: vk::ShaderStageFlags,
    ) -> DebuggerContext<'a> {
        DebuggerContext { dbg, eid, stage, frame_index: device.get_frame_index() }
    }

    /// Adds a debugger watch on command parameters (push data, bindings).
    ///
    /// This tells the debugger to capture the specified `query` just after the command specified by `eid`.
    ///
    ///
    /// # Arguments
    /// - `d` device
    /// - `eid` event ID of the command to watch
    /// - `stage` shader stage to watch
    pub fn add_watch(&mut self, load_chain: &LoadChain, byte_size: usize, transient: bool) -> WatchId {
        // compute query hash
        let mut h = FxHasher::default();
        (self.eid, self.stage, &load_chain, byte_size).hash(&mut h);
        let hash = h.finish();

        // find a matching, existing watch by query hash
        for (_id, watch) in self.dbg.watches.iter_mut() {
            if watch.hash == hash {
                // found a matching watch, mark it as live, return its id
                watch.last_request = self.frame_index;
                return _id;
            }
        }

        eprintln!("add watch: {}, stage={:?}, load_chain={:?}, byte_size={} ", self.eid, self.stage, load_chain, byte_size);

        self.dbg.watches.insert(Watch {
            hash,
            eid: self.eid,
            stage: self.stage,
            load_chain: load_chain.clone(),
            size: byte_size,
            result: None,
            transient,
            last_match: 0,
            last_request: self.frame_index,
        })
    }

    pub fn request_data(&mut self, load_chain: &LoadChain, byte_size: usize) -> Option<Vec<u8>> {
        let watch_id = self.add_watch(load_chain, byte_size, true);
        let watch = &self.dbg.watches[watch_id];
        if let Some(ref result_buffer) = watch.result
            && watch.last_match == self.frame_index
        {
            // the watch contains valid data
            let result_slice = unsafe { std::slice::from_raw_parts(result_buffer.ptr as *const u8, watch.size) };
            Some(result_slice.to_vec())
        } else {
            None
        }
    }
}

impl Device {
    unsafe fn read_load_chain(
        &self,
        load_chain: &LoadChain,
        cmd_buf: vk::CommandBuffer,
        push_data: &[u8],
        size: usize,
        result_buffer: &Buffer,
        buffer_offset: usize,
    ) {
        let addr0 = push_data.as_ptr().add(load_chain.offsets[0]);

        if load_chain.offsets.len() == 1 {
            // there are no indirections into device memory, we can just copy the data from the push
            // data buffer
            ptr::copy_nonoverlapping(addr0, result_buffer.ptr.add(buffer_offset) as *mut u8, size);
            return;
        }

        // The data spills in device memory, so dispatch a shader to copy from the rest of the load chain
        // into the provided buffer.

        let base1 = *(addr0 as *const u64);
        let addr1 = base1 + load_chain.offsets[1] as u64;
        let count = load_chain.offsets.len() - 2;

        let mut params = CopyIndirect1DParams {
            base: addr1,
            dst: result_buffer.device_address + buffer_offset as u64,
            byte_size: size as u32,
            count: count as u32,
            offset: [Default::default(); MAX_INDIRECTIONS],
        };
        for i in 0..count {
            params.offset[i] = load_chain.offsets[i + 2] as u32;
        }

        eprintln!("read_load_chain:");
        eprintln!("   offset[0]={}", load_chain.offsets[0]);
        eprintln!("   base1=0x{:016x}", base1);
        eprintln!("   addr1=0x{:016x}", addr1);
        eprintln!("   offsets={:?}", &load_chain.offsets[..]);
        eprintln!("   count={}", count);
        eprintln!("   device offsets={:?}", &params.offset[..]);

        let n_workgroups = size.div_ceil(COPY_WORKGROUP_SIZE as usize) as u32;
        self.cmd_bind_pipeline(
            cmd_buf,
            vk::PipelineBindPoint::COMPUTE,
            self.debugger_resources.copy_indirect_1d.pipeline,
        );
        self.push_constants_helper(
            cmd_buf,
            self.debugger_resources.copy_indirect_1d.pipeline_layout,
            vk::ShaderStageFlags::COMPUTE,
            &params,
        );
        self.cmd_dispatch(cmd_buf, n_workgroups, 1, 1);
    }

    pub fn retire_transient_watches(&self) {
        let frame_index = self.get_frame_index();
        let mut dbg = self.debugger.lock();
        dbg.watches.retain(|_id, watch| {
            if watch.transient && watch.last_request < frame_index {
                // no request for this transient watch in this frame, delete it
                if let Some(result_buffer) = watch.result.take() {
                    unsafe { self.destroy_buffer_helper(result_buffer) };
                }
                false
            } else {
                true
            }
        });
    }

    fn update_watch(&self, cmd_buf: vk::CommandBuffer, watch: &mut Watch, push_data: &[u8]) {
        if watch.size == 0 {
            // nothing to copy.
            return;
        }

        unsafe {
            // Allocate the return buffer if necessary
            let result_buffer = watch.result.get_or_insert_with(|| {
                self.create_buffer_helper(
                    vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::STORAGE_BUFFER,
                    watch.size,
                    None,
                )
            });

            // Allocate the return buffer
            self.read_load_chain(&watch.load_chain, cmd_buf, push_data, watch.size, result_buffer, 0);
        }
    }

    /// Called after every command to update relevant watches.
    pub fn update_watches_for_command(&self, command: &Command) {
        let mut dbg = self.debugger.lock();
        for watch in dbg.watches.values_mut() {
            if watch.eid == command.eid {
                eprintln!("debugger: ({}) updating watch stage={:?} load_chain={:?}", command.eid, watch.stage, watch.load_chain);
                watch.last_match = self.get_frame_index();
                self.update_watch(command.cmd_buf, watch, command.push.as_slice());
            }
        }
    }
}
