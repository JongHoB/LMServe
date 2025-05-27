use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about)]
pub struct LLMServeArgs {
    #[arg(long, default_value = "Qwen/Qwen2.5-0.5B")]
    pub model_name: String,

    #[arg(long, default_value_t = 8)]
    pub block_size: usize,

    /// Fraction of total GPU memory to use (0.0 ~ 1.0)
    #[arg(long, default_value_t = 0.9)]
    pub gpu_memory_fraction: f32,

    #[arg(long, default_value_t = 256)]
    pub max_batch_size: usize,

    #[arg(long, default_value_t = 4096)]
    pub max_seq_len: usize,

    #[arg(long, default_value_t = 5120)]
    pub max_num_batched_tokens: usize,

    #[arg(long, default_value_t = 1)]
    pub tp_size: u8,

    #[arg(long, default_value = "localhost")]
    pub address: String,

    #[arg(short, long, default_value_t = 8000)]
    pub port: u16,
}
