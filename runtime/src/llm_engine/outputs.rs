use super::Bytes;
use super::infer_task::InferTask;
use super::sequence::SeqStatus;

pub fn norm_log_probs(probs: &[f32]) -> f32 {
    let log_sum: f32 = probs
        .iter()
        .map(|&p| if p > 0.0 { p.ln() } else { f32::NEG_INFINITY })
        .sum();

    log_sum / probs.len() as f32
}

pub struct GenerateOutput {
    pub session_id: String,
    pub output_ids: Vec<u32>,
    /// Token latencies in seconds.
    pub token_latencies: Vec<f32>,
}

impl GenerateOutput {
    pub fn from_task(task: &InferTask) -> Self {
        let seqs = task.get_seqs(SeqStatus::Finished);
        let selected_seq = seqs
            .iter()
            .max_by(|a, b| {
                norm_log_probs(a.get_token_probs().as_ref())
                    .partial_cmp(&norm_log_probs(b.get_token_probs().as_ref()))
                    .unwrap()
            })
            .expect("Failed to select sequence: max_by() returned None.");

        let output_ids = selected_seq.get_output_ids();

        let mut token_times = Vec::with_capacity(selected_seq.append_token_times.len() + 1);
        token_times.push(task.get_arrival_time());
        token_times.extend_from_slice(&selected_seq.append_token_times);

        let token_latencies: Vec<f32> = token_times
            .windows(2)
            .map(|pair| pair[1].saturating_sub(pair[0]) as f32 / 1e9)
            .collect();

        Self {
            session_id: task.get_session_id(),
            output_ids: output_ids.to_vec(),
            token_latencies,
        }
    }
}

pub struct ReserveOutput {
    pub kv_descs: Vec<Bytes>,
    pub hash_values: Vec<u64>,
}

pub struct TransferOutput {
    pub hash_values: Vec<u64>,
}

pub struct EngineStatus {
    pub num_running_reqs: usize,
    pub num_allocated_reqs: usize,
    pub num_waiting_reqs: usize,
    pub num_pendding_reqs: usize,
    pub gpu_kv_block_usage: f32,
    pub host_kv_block_usage: f32,
}
