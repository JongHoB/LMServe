use std::{
    collections::HashMap,
    os::unix::process::CommandExt,
    process::{Child, Command},
    sync::Arc,
    time::Duration,
};

use nix::{
    sys::signal::{Signal, killpg},
    unistd::Pid,
};

use futures::future::join_all;

use tokio::{sync::Mutex, time::sleep};

use tonic::transport::Channel;

use tracing::{error, info};

use crate::infer_task::{InferInput, InferOutput};
use crate::pb::worker::worker_client::WorkerClient;
use crate::pb::worker::{InferRequest, InitCacheRequest, WarmupRequest};

type StdError = Box<dyn std::error::Error + Send + Sync + 'static>;
type Result<T, E = StdError> = std::result::Result<T, E>;

const MASTER_ADDR: &str = "localhost";
const MASTER_PORT: u32 = 6000;

fn extract_if_all_equal<T>(items: &[T]) -> Result<T, &'static str>
where
    T: PartialEq + Clone,
{
    if let Some(first) = items.first() {
        if items.iter().all(|x| x == first) {
            Ok(first.clone())
        } else {
            Err("Values are not all equal")
        }
    } else {
        Err("Vector is empty")
    }
}

pub struct Worker {
    client: Arc<Mutex<WorkerClient<Channel>>>,
}

impl Worker {
    pub async fn connect(port: u32) -> Result<Worker> {
        info!("Waiting for model worker to be ready...");
        let client = loop {
            match WorkerClient::connect(format!("http://[::1]:{port}")).await {
                Ok(client) => {
                    info!("Successfully connected to the worker.");
                    break client;
                }
                Err(_) => {}
            };

            sleep(Duration::from_millis(500)).await;
        };

        Ok(Worker {
            client: Arc::new(Mutex::new(client)),
        })
    }

    pub async fn warmup(&self, request: WarmupRequest) -> Result<(u64, u64)> {
        let response = self.client.lock().await.warmup(request).await?;
        let response = response.into_inner();
        Ok((response.gpu_total_mem_size, response.gpu_peak_mem_size))
    }

    pub async fn infer(&self, request: InferRequest) -> Result<HashMap<u64, InferOutput>> {
        let response = self.client.lock().await.infer(request).await?;
        let outputs = response.into_inner().outputs;
        Ok(outputs)
    }

    pub async fn init_cache(&self, request: InitCacheRequest) -> Result<u64> {
        let response = self.client.lock().await.init_cache(request).await?;
        let num_blocks = response.into_inner().num_blocks;
        Ok(num_blocks)
    }
}

pub struct WorkerHandle {
    child: Child,
}

impl Drop for WorkerHandle {
    fn drop(&mut self) {
        let pid = self.child.id();
        let result = killpg(Pid::from_raw(pid as i32), Signal::SIGTERM);
        match result {
            Ok(_) => {
                let _ = self.child.wait();
                info!("Successfully terminated worker (pid={pid}).")
            }
            Err(err) => error!("Failed to terminate worker (pid={pid}): {err}"),
        }
    }
}

#[allow(dead_code)]
pub struct WorkerGroup {
    worker_handles: Vec<WorkerHandle>,
    workers: Vec<Arc<Worker>>,
}

impl WorkerGroup {
    pub async fn init(
        model_name: String,
        block_size: usize,
        num_workers: u8,
        base_port: u32,
    ) -> Result<WorkerGroup> {
        let worker_handles = WorkerGroup::launch(model_name, block_size, num_workers, base_port)?;
        let mut workers: Vec<_> = Vec::with_capacity(num_workers as usize);
        for rank in 0..num_workers {
            let port = base_port + rank as u32;
            let worker = Worker::connect(port)
                .await
                .unwrap_or_else(|e| panic!("Failed to connect worker: {e}"));
            workers.push(Arc::new(worker));
        }

        Ok(WorkerGroup {
            worker_handles,
            workers,
        })
    }

    fn launch(
        model_name: String,
        block_size: usize,
        num_workers: u8,
        base_port: u32,
    ) -> Result<Vec<WorkerHandle>> {
        let mut worker_handles: Vec<WorkerHandle> = Vec::with_capacity(num_workers as usize);
        for rank in 0..num_workers {
            let port = base_port + rank as u32;
            let args = vec![
                model_name.clone(),
                block_size.to_string(),
                rank.to_string(),
                port.to_string(),
            ];
            let mut envs = HashMap::new();
            envs.insert("RANK", rank.to_string());
            envs.insert("WORLD_SIZE", num_workers.to_string());
            envs.insert("MASTER_ADDR", MASTER_ADDR.to_string());
            envs.insert("MASTER_PORT", MASTER_PORT.to_string());
            let worker_handle = WorkerHandle {
                child: Command::new("llmserve-worker")
                    .args(args)
                    .process_group(0)
                    .envs(envs)
                    .spawn()?,
            };
            worker_handles.push(worker_handle);
        }
        Ok(worker_handles)
    }

    pub async fn warmup(
        &self,
        max_batch_size: usize,
        max_seq_len: usize,
        max_num_batched_tokens: usize,
    ) -> Result<(u64, u64)> {
        let request = WarmupRequest {
            max_batch_size: max_batch_size as u64,
            max_seq_len: max_seq_len as u64,
            max_num_batched_tokens: max_num_batched_tokens as u64,
        };
        let outputs = self
            .run_workers_gather(request, move |worker, request| async move {
                worker.warmup(request).await
            })
            .await?;

        let (total, peak) = match outputs.into_iter().min_by_key(|x| x.0) {
            Some(min_values) => min_values,
            None => panic!("Outputs is empty"),
        };

        Ok((total, peak))
    }

    pub async fn init_cache(&self, cache_size: usize) -> Result<u64> {
        let request = InitCacheRequest {
            cache_size: cache_size as u64,
        };
        let outputs = self
            .run_workers_gather(request, move |worker, request| async move {
                worker.init_cache(request).await
            })
            .await?;

        let num_blocks = extract_if_all_equal(&outputs).expect("Error: not all values are equal");

        Ok(num_blocks)
    }

    pub async fn infer(&self, inputs: Vec<InferInput>) -> Result<HashMap<u64, InferOutput>> {
        let request = InferRequest {
            inputs,
            use_cache: true,
        };
        let outputs = self
            .run_workers_gather(request, move |worker, request| async move {
                worker.infer(request).await
            })
            .await?;

        let infer_outputs =
            extract_if_all_equal(&outputs).expect("Error: not all values are equal");

        Ok(infer_outputs)
    }

    async fn run_workers_gather<T, P, E, Fut, F>(&self, params: P, task_fn: F) -> Result<Vec<T>, E>
    where
        T: PartialEq + Send + 'static,
        P: Send + Sync + Clone + 'static,
        E: Send + 'static,
        Fut: std::future::Future<Output = Result<T, E>> + Send + 'static,
        F: Fn(Arc<Worker>, P) -> Fut + Copy + Send + Sync + 'static,
    {
        let mut handles = Vec::with_capacity(self.workers.len());

        for worker in self.workers.iter() {
            let w = Arc::clone(worker);
            let p = params.clone();

            let handle = tokio::spawn(async move { task_fn(w, p).await });

            handles.push(handle);
        }

        let results = join_all(handles).await;

        let mut outputs = Vec::with_capacity(results.len());
        for res in results {
            match res {
                Ok(Ok(val)) => outputs.push(val),
                Ok(Err(_)) => panic!("Failed to run worker"),
                Err(join_err) => {
                    panic!("tokio::spawn task panicked: {:?}", join_err);
                }
            }
        }

        Ok(outputs)
    }
}
