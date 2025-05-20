use std::collections::HashMap;
use std::os::unix::process::CommandExt;
use std::process::Command;
use std::time::Duration;

use tokio::time::sleep;
use tonic::transport::Channel;
use tracing::info;

use crate::infer_task::{InferInput, InferOutput};
use crate::pb::worker::worker_client::WorkerClient;
use crate::pb::worker::{InferRequest, InitCacheRequest, WarmupRequest};

pub struct Worker {
    pid: u32,
    client: WorkerClient<Channel>,
}

impl Worker {
    pub async fn new(
        model_name: String,
        block_size: usize,
    ) -> Result<Worker, Box<dyn std::error::Error>> {
        let args = vec![model_name, block_size.to_string()];
        let worker = Command::new("llmserve-worker")
            .args(args)
            .process_group(0)
            .spawn()?;

        info!("Waiting for model worker to be ready...");
        let client = loop {
            match WorkerClient::connect("http://[::1]:5000").await {
                Ok(client) => {
                    info!("Successfully connected to the worker.");
                    break client;
                }
                Err(_) => {}
            };

            sleep(Duration::from_millis(500)).await;
        };

        Ok(Worker {
            pid: worker.id(),
            client,
        })
    }

    pub async fn warmup(
        &mut self,
        max_batch_size: u64,
        max_seq_len: u64,
        max_num_batched_tokens: u64,
    ) -> Result<(u64, u64), Box<dyn std::error::Error>> {
        let response = self
            .client
            .warmup(WarmupRequest {
                max_batch_size,
                max_seq_len,
                max_num_batched_tokens,
            })
            .await?;
        let response = response.into_inner();
        Ok((response.gpu_total_mem_size, response.gpu_peak_mem_size))
    }

    pub async fn infer(
        &mut self,
        inputs: Vec<InferInput>,
    ) -> Result<HashMap<u64, InferOutput>, Box<dyn std::error::Error>> {
        let response = self
            .client
            .infer(InferRequest {
                inputs,
                use_cache: true,
            })
            .await?;
        let outputs = response.into_inner().outputs;
        Ok(outputs)
    }

    pub async fn init_cache(&mut self, cache_size: u64) -> Result<u64, Box<dyn std::error::Error>> {
        let response = self
            .client
            .init_cache(InitCacheRequest { cache_size })
            .await?;
        let num_blocks = response.into_inner().num_blocks;
        Ok(num_blocks)
    }

    pub fn get_pid(&self) -> u32 {
        self.pid
    }
}
