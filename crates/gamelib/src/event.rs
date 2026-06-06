use std::fmt;

/// A token that uniquely identifies a non-input event.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct EventToken(pub u64);


pub enum UserEvent {
    Timeout(EventToken),
    Callback(Box<dyn FnOnce() + Send>),
}

impl fmt::Debug for UserEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UserEvent::Timeout(token) => write!(f, "UserEvent::Timeout({:?})", token),
            UserEvent::Callback(_) => write!(f, "UserEvent::Callback(<fn>)"),
        }
    }
}
