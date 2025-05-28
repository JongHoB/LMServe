use std::sync::atomic::{AtomicUsize, Ordering};

use uuid::Uuid;

static GLOBAL_COUNTER: AtomicUsize = AtomicUsize::new(0);

pub fn generate_seq_id() -> u64 {
    GLOBAL_COUNTER.fetch_add(1, Ordering::SeqCst) as u64
}

pub fn generate_session_id() -> String {
    Uuid::new_v4().to_string()
}
