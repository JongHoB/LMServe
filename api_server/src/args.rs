use clap::Parser;
use serde::Serialize;

#[derive(Parser, Debug, Serialize)]
#[command(author, version, about)]
pub struct APIServerArgs {
    #[arg(long, default_value = "Qwen/Qwen2.5-0.5B")]
    pub tokenier_name: String,

    #[arg(long, default_values_t = vec!["http://[::1]:7000".to_string()])]
    pub engine_urls: Vec<String>,

    #[arg(long, default_value = "localhost")]
    pub address: String,

    #[arg(short, long, default_value_t = 8000)]
    pub port: u32,
}
