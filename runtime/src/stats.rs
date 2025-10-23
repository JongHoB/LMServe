use std::fmt;

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Stats {
    pub num_running_reqs: usize,
    pub num_allocated_reqs: usize,
    pub num_waiting_reqs: usize,
    pub num_pending_reqs: usize,
    pub num_promoted_reqs: usize,
    pub gpu_kv_block_usage: f32,
    pub host_kv_block_usage: f32,
}

impl fmt::Display for Stats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Running: {} seqs, Allocated: {} reqs, Waiting: {} reqs, Pending: {} reqs, Promoted: {} reqs, GPU KV usage: {:.2} %, Host KV usage: {:.2} %",
            self.num_running_reqs,
            self.num_allocated_reqs,
            self.num_waiting_reqs,
            self.num_pending_reqs,
            self.num_promoted_reqs,
            self.gpu_kv_block_usage * 100.0,
            self.host_kv_block_usage * 100.0,
        )
    }
}
