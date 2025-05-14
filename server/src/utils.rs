use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};
use uuid::Uuid;

pub struct Counter {
    count: AtomicUsize,
}

impl Counter {
    pub fn new() -> Counter {
        Counter {
            count: AtomicUsize::new(0),
        }
    }

    pub fn next(&mut self) -> usize {
        self.count.fetch_add(1, Ordering::SeqCst)
    }
}

static GLOBAL_COUNTER: AtomicUsize = AtomicUsize::new(0);

pub fn generate_session_id() -> String {
    Uuid::new_v4().to_string()
}

pub fn generate_seq_id() -> u64 {
    GLOBAL_COUNTER.fetch_add(1, Ordering::SeqCst) as u64
}

pub fn now() -> u64 {
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(duration) => duration.as_nanos() as u64,
        Err(_) => panic!("SystemTime before UNIX EPOCH!"),
    }
}
