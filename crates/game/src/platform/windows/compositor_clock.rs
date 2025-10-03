use crate::platform::windows::WakeReason;
use crate::platform::windows::event_loop::EVENT_LOOP_PROXY;
use log::{error, info, trace};
use std::ffi::c_void;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering::{Relaxed, SeqCst};
use windows::Win32::Foundation::{HANDLE, WAIT_OBJECT_0};
use windows::Win32::Graphics::DirectComposition::DCompositionWaitForCompositorClock;
use windows::Win32::System::Threading::{CreateEventW, INFINITE, SetEvent};
use windows::core::Owned;

pub(super) struct CompositorClock {
    abort_event: Owned<HANDLE>,
    active: Arc<AtomicBool>,
}

impl CompositorClock {
    pub(super) fn new() -> CompositorClock {
        // create an event that can be used to abort the clock thread
        let abort_event = unsafe { CreateEventW(None, false, false, None).unwrap() };
        let abort_event = unsafe { Owned::new(abort_event) };
        CompositorClock {
            abort_event,
            active: Default::default(),
        }
    }

    /// Marks that a VSync event should be sent to the event loop on the next compositor clock tick.
    pub(super) fn trigger(&self) {
        self.active.store(true, Relaxed);
    }

    /// Starts the compositor clock thread.
    pub(super) fn start(&self) {
        // Some HANDLEs aren't thread safe, but some are meant to be used across threads, like events.
        // Annoyingly windows-rs sided with making all HANDLEs !Send,
        // which means that we need some ugliness to get around that.
        //
        // See https://github.com/microsoft/windows-rs/issues/3169 for some discussion.
        //
        // I don't think that's a good choice: first, HANDLEs are not necessarily
        // pointers, and most importantly all functions taking HANDLEs are FFI-unsafe, so
        // thread-safety requirements can be added to the safety contracts of the APIs
        // that use HANDLEs, instead of being on the HANDLE type itself.
        let abort_raw = self.abort_event.0 as isize;
        let active = self.active.clone();
        std::thread::spawn(move || {
            let abort_event = HANDLE(abort_raw as *mut c_void);
            loop {
                unsafe {
                    let wait_result = DCompositionWaitForCompositorClock(Some(&[abort_event]), INFINITE);
                    if wait_result == WAIT_OBJECT_0.0 {
                        trace!("Compositor clock stopping");
                        break;
                    } else if wait_result == WAIT_OBJECT_0.0 + 1 {
                        // Ignore delivery errors if the event loop is not running
                        if let Some(proxy) = EVENT_LOOP_PROXY.get() {
                            if active.swap(false, Relaxed) {
                                let _ = proxy.send_event(WakeReason::VSync);
                            }
                        }
                    } else {
                        error!("DCompositionWaitForCompositorClock failed: returned {wait_result:08x}");
                    }
                }
            }
        });
        info!("Compositor clock started");
    }

    /// Stops the compositor clock thread.
    pub(super) fn stop(&self) {
        unsafe {
            SetEvent(*self.abort_event).unwrap();
        }
    }
}

impl Drop for CompositorClock {
    fn drop(&mut self) {
        self.stop();
    }
}
