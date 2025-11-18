use slotmap::SlotMap;
use std::sync::{LazyLock, Mutex};
use std::time::Instant;

slotmap::new_key_type! {
    /// Uniquely identifies a timer.
    pub struct TimerToken;
}

struct Timer {
    deadline: Instant,
}

impl TimerToken {
    pub fn new(deadline: Instant) -> Self {
        TIMERS.lock().unwrap().insert(Timer { deadline })
    }

    pub fn cancel(self) {
        TIMERS.lock().unwrap().remove(self);
    }

    pub fn reset(&mut self, new_deadline: Instant) {}
}

/// Pending timers.
static TIMERS: LazyLock<Mutex<SlotMap<TimerToken, Timer>>> = LazyLock::new(|| Mutex::new(SlotMap::with_key()));
