use crate::utils;

#[derive(PartialEq, Eq)]
pub enum SeqStatus {
    WAITING,
    ALLOCATED,
    FINISHED,
    CANCELLED,
}

#[derive(PartialEq, Eq)]
pub enum StopReason {
    EOS,
    MAXLENGTH,
}

pub struct Sequence {
    pub seq_id: u64,
    pub session_id: String,

    pub prompt: String,
    pub token_ids: Vec<u32>,
    pub max_output_len: Option<u32>,

    pub status: SeqStatus,
    pub stop_reason: Option<StopReason>,
    pub output_len: u32,
    output_words: Vec<String>,
    output_token_probs: Vec<f32>,
    filled_token_ids: Vec<u32>,

    append_token_times: Vec<f32>,
}

impl Sequence {
    pub fn new(
        session_id: String,
        prompt: String,
        token_ids: Vec<u32>,
        max_output_len: Option<u32>,
    ) -> Sequence {
        let seq_id = utils::generate_seq_id();

        Sequence {
            seq_id,
            session_id,
            prompt,
            token_ids,
            max_output_len,
            status: SeqStatus::WAITING,
            stop_reason: None,
            output_len: 0,
            output_words: Vec::new(),
            output_token_probs: Vec::new(),
            filled_token_ids: Vec::new(),
            append_token_times: Vec::new(),
        }
    }

    pub fn is_active(&self) -> bool{
        matches!(self.status, SeqStatus::WAITING | SeqStatus::ALLOCATED)
    }

    pub fn is_finished(&self) -> bool {
        matches!(self.status, SeqStatus::FINISHED | SeqStatus::CANCELLED)
    }

    pub fn append_output_id(&mut self, output_id: u32) {
        self.token_ids.push(output_id);
        self.output_len += 1;
        // TODO(jinu): Add prob & word.
    }
}
