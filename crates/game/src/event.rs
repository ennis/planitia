use event_listener::{EventListener, IntoNotification, Listener};

pub struct Event<T> {
    event: event_listener::Event<T>,
}

impl<T> Event<T> {
    pub const fn new() -> Self {
        Self {
            event: event_listener::Event::with_tag(),
        }
    }
}

impl<T: Clone> Event<T> {
    pub fn notify(&self, data: T) {
        let count = self.event.total_listeners();
        // takes "IntoNotification" which conceptually represents a number of listeners + a payload
        // instead of having a method with two parameters
        self.event.notify(count.tag(data));
    }

    pub fn listen(&self) -> EventListener<T> {
        self.event.listen()
    }
}
