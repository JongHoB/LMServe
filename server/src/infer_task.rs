use crate::sequence::{SeqStatus, Sequence};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum InferTaskStatus {
    WAITING,
    DECODE,
    PREFILL,
}

pub struct InferTask {
    session_id: String,
    seqs: Vec<Sequence>,
    status: InferTaskStatus,
    arrival_time: u64,
    num_samples: u16,
}

impl InferTask {
    pub fn new(session_id: String, seqs: Vec<Sequence>, arrival_time: u64) -> InferTask {
        let num_samples = seqs.len() as u16;
        InferTask {
            session_id,
            seqs,
            status: InferTaskStatus::WAITING,
            arrival_time,
            num_samples,
        }
    }

    pub fn get_active_seqs(&self) -> Vec<&Sequence> {
        self.seqs.iter().filter(|seq| seq.is_active()).collect()
    }

    pub fn get_active_seqs_mut(&mut self) -> Vec<&mut Sequence> {
        self.seqs.iter_mut().filter(|seq| seq.is_active()).collect()
    }

    pub fn get_seqs(&self, status: SeqStatus) -> Vec<&Sequence> {
        self.seqs
            .iter()
            .filter(|seq| seq.status == status)
            .collect()
    }

    pub fn get_status(&self) -> InferTaskStatus {
        self.status
    }

    pub fn set_prefill(&mut self) {
        self.status = InferTaskStatus::PREFILL;
    }

    pub fn set_decode(&mut self) {
        self.status = InferTaskStatus::DECODE;
    }

    pub fn is_finished(&mut self) -> bool{
        self.seqs.iter().all(|seq| seq.is_finished())
    }
}

pub struct InferInput {
    pub seq_id: u64,
    input_ids: Vec<u32>,
    input_len: usize,
    filled_token_len: usize,
    context_len: usize,
    block_ids: Vec<u32>,
}

impl InferInput {
    pub fn new(
        seq_id: u64,
        input_ids: Vec<u32>,
        filled_token_len: usize,
        block_ids: Vec<u32>,
    ) -> InferInput {
        let input_len = input_ids.len();
        InferInput {
            seq_id,
            input_ids,
            input_len,
            filled_token_len,
            context_len: filled_token_len + input_len,
            block_ids,
        }
    }
}

pub struct InferOutput {
    pub output_id: u32,
}

impl InferOutput {
    pub fn new(output_id: u32) -> InferOutput {
        InferOutput { output_id }
    }
}
