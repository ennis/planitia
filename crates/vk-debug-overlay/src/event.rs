//! Device events
use crate::Device;
use crate::state_tracker::command::CmdKind;
use ash::vk;
use rustc_hash::FxHasher;
use std::ffi::CString;
use std::fmt;
use std::hash::{Hash, Hasher};

#[derive(Clone, Eq, PartialEq, Hash, Debug)]
enum DrawCommandKind {
    Draw { vertex_count: u32, instance_count: u32, first_vertex: u32, first_instance: u32 },
    DrawIndexed { index_count: u32, instance_count: u32, first_index: u32, vertex_offset: i32, first_instance: u32 },
}

fn hash_draw_command(
    regions: &[CString],
    color_attachments: &[vk::Format],
    depth_format: vk::Format,
    command_kind: &DrawCommandKind,
) -> u64 {
    let mut h = FxHasher::default();
    h.write_u8(0);
    regions.hash(&mut h);
    color_attachments.hash(&mut h);
    depth_format.hash(&mut h);
    command_kind.hash(&mut h);
    h.finish()
}

fn hash_dispatch_command(regions: &[CString], group_count_x: u32, group_count_y: u32, group_count_z: u32) -> u64 {
    let mut h = FxHasher::default();
    h.write_u8(1);
    regions.hash(&mut h);
    h.write_u32(group_count_x);
    h.write_u32(group_count_y);
    h.write_u32(group_count_z);
    h.finish()
}

/// Uniquely identifies a device event within a frame, and (as much as possible) across frames.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct EId(pub u64);

impl fmt::Display for EId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Write only the last 4 hex digits for the display impl. EIDs are supposed to be hashes,
        // so the risk of collision is relatively small as long as the number of displayed events stay small.
        write!(f, "{:04X}", self.0 & 0xffff)
    }
}

pub struct EventTimeline {
    events: Vec<EventInfo>,
    idx: usize,
}

impl EventTimeline {
    pub fn new() -> EventTimeline {
        EventTimeline { events: vec![], idx: 0 }
    }

    pub fn reset(&mut self) {
        // trim list of events
        self.events.truncate(self.idx);
        self.idx = 0;
    }

    fn find_and_move_eid(&mut self, hash: u64) -> Option<EId> {
        for i in self.idx..self.events.len() {
            // we match events between frames if:
            // - the hash matches (same command parameters, pipelines, and debug region markers)
            if self.events[i].hash == hash {
                // rotate event in place
                self.events[self.idx..].rotate_left(i - self.idx);
                let id = self.events[self.idx].id;
                self.idx += 1;
                return Some(id);
            }
        }
        None
    }

    fn insert(&mut self, hash: u64) -> EId {
        if let Some(eid) = self.find_and_move_eid(hash) {
            return eid;
        }
        // For now EID == hash, but we may want to change this in the future so that e.g. EIDs
        // are ordered.
        let eid = EId(hash);
        self.events.insert(self.idx, EventInfo { hash, id: eid });
        self.idx += 1;
        eid
    }
}

struct EventInfo {
    hash: u64,
    // NOTE: for now the EID == hash, but this may change in the future
    id: EId,
}

//--------------------------------------------------------------------------------------------------

impl Device {
    /// Returns an event ID (EID) for the specified command.
    ///
    /// The EID is designed to uniquely identify a command within a frame, and be stable for the same command
    /// across frames. Currently, it is a hash of:
    /// - the debug region markers
    /// - the currently bound pipeline
    /// - command parameters
    pub fn get_command_eid(&self, debug_region_markers: &[CString], pipeline: vk::Pipeline, cmd: &CmdKind) -> EId {
        let mut h = FxHasher::default();
        debug_region_markers.hash(&mut h);
        pipeline.hash(&mut h);
        cmd.hash(&mut h);
        let hash = h.finish();
        let mut evt = self.event_timeline.lock();
        evt.insert(hash)
    }
}
