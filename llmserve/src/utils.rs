use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};
use uuid::Uuid;

pub const KB: u64 = 1024;
pub const MB: u64 = 1024 * KB;
pub const GB: u64 = 1024 * MB;

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

pub fn norm_log_probs(probs: &[f32]) -> f32 {
    let log_sum: f32 = probs
        .iter()
        .map(|&p| if p > 0.0 { p.ln() } else { f32::NEG_INFINITY })
        .sum();

    log_sum / probs.len() as f32
}
