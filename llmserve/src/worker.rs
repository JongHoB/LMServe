use std::{
    collections::HashMap,
    sync::Arc,
    time::Duration,
    env,
};

use futures::future::join_all;

use tokio::{sync::Mutex, time::sleep};

use tonic::transport::Channel;

use tracing::{debug, info};

use crate::infer_task::{InferInput, InferOutput};
use crate::pb::worker::worker_client::WorkerClient;
use crate::pb::worker::{InferRequest, InitCacheRequest, WarmupRequest};

type StdError = Box<dyn std::error::Error + Send + Sync + 'static>;
type Result<T, E = StdError> = std::result::Result<T, E>;

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
    pub async fn connect(address: String) -> Result<Worker> {
        // TODO(jinu): Add timeout.
        let client = loop {
            match WorkerClient::connect(address.clone()).await {
                Ok(client) => {
                    info!("Successfully connected to the worker.");
                    break client;
                }
                Err(error) => {
                    debug!("Trying to connect to worker ({}): {:?}", address, error);
                }
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

#[allow(dead_code)]
pub struct WorkerGroup {
    workers: Vec<Arc<Worker>>,
}

impl WorkerGroup {
    pub async fn init(
        num_workers: u8,
    ) -> Result<WorkerGroup> {
        let mut workers: Vec<_> = Vec::with_capacity(num_workers as usize);
        let uds_path_prefix = env::var("WORKER_UDS_PATH_PREFIX")?;
        for rank in 0..num_workers {
            let address = format!("unix://{}-{}", uds_path_prefix, rank);
            let worker = Worker::connect(address.parse().unwrap())
                .await
                .unwrap_or_else(|e| panic!("Failed to connect worker: {e}"));
            workers.push(Arc::new(worker));
        }

        Ok(WorkerGroup { workers })
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
