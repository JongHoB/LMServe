use std::{collections::HashMap, marker::PhantomData, path::PathBuf, sync::Arc, time::Duration};

use futures::future::join_all;

use hyper_util::rt::TokioIo;
use tokio::net::UnixStream;
use tokio::time::sleep;
use tonic::transport::{Channel, Endpoint, Uri};
use tower::service_fn;

use tracing::debug;

use crate::pb::worker::kv_worker_client::KvWorkerClient;
use crate::pb::worker::worker_client::WorkerClient;
use crate::pb::worker::{
    AgentMetadata, BlockMapping, CopyKvRequest, GetDescriptorsRequest, InferRequest,
    InitCacheRequest, PullKvRequest, PushKvRequest, WarmupRequest,
};

use super::infer_task::{InferInput, InferOutput};
use super::{Bytes, Device};

type StdError = Box<dyn std::error::Error + Send + Sync + 'static>;
type Result<T, E = StdError> = std::result::Result<T, E>;

#[tonic::async_trait]
pub trait GrpcClient: Send + Sync + Sized + 'static {
    fn new(channel: Channel) -> Self;
}

#[tonic::async_trait]
impl GrpcClient for WorkerClient<Channel> {
    fn new(channel: Channel) -> Self {
        WorkerClient::new(channel)
    }
}

#[tonic::async_trait]
impl GrpcClient for KvWorkerClient<Channel> {
    fn new(channel: Channel) -> Self {
        KvWorkerClient::new(channel)
    }
}

#[tonic::async_trait]
trait Worker: Send + Sync + Sized + 'static {
    type Client: GrpcClient;

    async fn new(address: String) -> Result<Self>;

    fn connect(&self) -> Self::Client;
}

pub struct WorkerImpl<T: GrpcClient> {
    channel: Channel,
    _marker: PhantomData<T>,
}

#[tonic::async_trait]
impl<T: GrpcClient> Worker for WorkerImpl<T> {
    type Client = T;

    async fn new(address: String) -> Result<Self> {
        // TODO(jinu): Add timeout
        let channel = if let Some(path) = address.strip_prefix("unix://") {
            let sock: PathBuf = PathBuf::from(path);

            loop {
                let sock = sock.clone();

                // Open a gRPC channel over a Unix domain socket using a custom connector.
                // The URI below is just a placeholder.
                // Ref: https://github.com/hyperium/tonic/blob/master/examples/src/uds/client_with_connector.rs
                match Endpoint::try_from("http://[::]:50051")? // Dummy address, not used
                    .connect_with_connector(service_fn(move |_: Uri| {
                        let sock = sock.clone();

                        async move {
                            // Connect to a UDS socket
                            Ok::<_, std::io::Error>(TokioIo::new(UnixStream::connect(sock).await?))
                        }
                    }))
                    .await
                {
                    Ok(channel) => {
                        debug!("Successfully connected to {:?}", address);
                        break channel;
                    }
                    Err(error) => {
                        debug!("Trying to connect to worker ({:?}): {:?}", address, error);
                        sleep(Duration::from_millis(500)).await;
                    }
                }
            }
        } else {
            let uri: String = address.parse().unwrap();
            let endpoint = Channel::from_shared(uri.clone().into_bytes())?;

            loop {
                match endpoint.connect().await {
                    Ok(channel) => {
                        debug!("Successfully connected to {address}");
                        break channel;
                    }
                    Err(error) => {
                        debug!("Trying to connect to worker ({}): {:?}", address, error);
                        sleep(Duration::from_millis(500)).await;
                    }
                };
            }
        };

        Ok(Self {
            channel,
            _marker: PhantomData,
        })
    }

    fn connect(&self) -> T {
        T::new(self.channel.clone())
    }
}

type ModelWorker = WorkerImpl<WorkerClient<Channel>>;
type KVWorker = WorkerImpl<KvWorkerClient<Channel>>;

impl ModelWorker {
    async fn warmup(&self, request: WarmupRequest) -> Result<(u64, u64)> {
        let mut client = self.connect();
        let response = client.warmup(request).await?;
        let response = response.into_inner();
        Ok((response.gpu_total_mem_size, response.gpu_peak_mem_size))
    }

    async fn infer(&self, request: InferRequest) -> Result<HashMap<u64, InferOutput>> {
        let mut client = self.connect();
        let response = client.infer(request).await?;
        let outputs = response.into_inner().outputs;
        Ok(outputs)
    }

    async fn init_cache(&self, request: InitCacheRequest) -> Result<(u64, u64, u64)> {
        let mut client = self.connect();
        let response = client.init_cache(request).await?;
        let response = response.into_inner();

        let num_gpu_blocks = response.num_gpu_blocks;
        let num_host_blocks = response.num_host_blocks;
        let num_disk_blocks = response.num_disk_blocks;

        Ok((num_gpu_blocks, num_host_blocks, num_disk_blocks))
    }
}

impl KVWorker {
    async fn copy_kv(&self, request: CopyKvRequest) -> Result<()> {
        let mut client = self.connect();
        let _ = client.copy_kv(request).await?;
        Ok(())
    }

    async fn get_local_agent_metadata(&self) -> Result<(String, Vec<u8>)> {
        let mut client = self.connect();
        let response = client.get_local_agent_metadata(()).await?;

        let response = response.into_inner();
        let agent_name = response.agent_name;
        let metadata = response.data;
        Ok((agent_name, metadata))
    }

    async fn add_remote_agent_metadata(&self, request: AgentMetadata) -> Result<String> {
        let mut client = self.connect();
        let response = client.add_remote_agent_metadata(request).await?;
        let peer_name = response.into_inner().peer_name;
        Ok(peer_name)
    }

    async fn get_descriptors(&self, request: GetDescriptorsRequest) -> Result<Bytes> {
        let mut client = self.connect();
        let response = client.get_descriptors(request).await?.into_inner();

        let descs = response.descs;
        Ok(descs)
    }

    async fn push_kv(&self, request: PushKvRequest) -> Result<bool> {
        let mut client = self.connect();
        let response = client.push_kv(request).await?.into_inner();

        Ok(response.success)
    }

    async fn pull_kv(&self, request: PullKvRequest) -> Result<bool> {
        let mut client = self.connect();
        let response = client.pull_kv(request).await?.into_inner();

        Ok(response.success)
    }
}

#[tonic::async_trait]
trait WorkerGroup: Send + Sync + Sized + 'static {
    type Worker: Worker;

    async fn run_workers_gather<V, P, E, Fut, F>(&self, params: P, task_fn: F) -> Result<Vec<V>, E>
    where
        V: PartialEq + Send + 'static,
        P: Send + Sync + Clone + 'static,
        E: Send + 'static,
        Fut: std::future::Future<Output = Result<V, E>> + Send + 'static,
        F: Fn(Arc<Self::Worker>, P) -> Fut + Copy + Send + Sync + 'static;

    async fn run_workers<V, P, E, Fut, F>(
        &self,
        params_list: &[P],
        task_fn: F,
    ) -> Result<Vec<V>, E>
    where
        V: PartialEq + Send + 'static,
        P: Send + Sync + Clone + 'static,
        E: Send + 'static,
        Fut: std::future::Future<Output = Result<V, E>> + Send + 'static,
        F: Fn(Arc<Self::Worker>, P) -> Fut + Copy + Send + Sync + 'static;
}

pub struct WorkerGroupImpl<T> {
    workers: Vec<Arc<T>>,
}

#[tonic::async_trait]
impl<T: Worker> WorkerGroup for WorkerGroupImpl<T> {
    type Worker = T;

    async fn run_workers_gather<V, P, E, Fut, F>(&self, params: P, task_fn: F) -> Result<Vec<V>, E>
    where
        V: PartialEq + Send + 'static,
        P: Send + Sync + Clone + 'static,
        E: Send + 'static,
        Fut: std::future::Future<Output = Result<V, E>> + Send + 'static,
        F: Fn(Arc<T>, P) -> Fut + Copy + Send + Sync + 'static,
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

    async fn run_workers<V, P, E, Fut, F>(&self, params_list: &[P], task_fn: F) -> Result<Vec<V>, E>
    where
        V: PartialEq + Send + 'static,
        P: Send + Sync + Clone + 'static,
        E: Send + 'static,
        Fut: std::future::Future<Output = Result<V, E>> + Send + 'static,
        F: Fn(Arc<T>, P) -> Fut + Copy + Send + Sync + 'static,
    {
        assert!(
            self.workers.len() == params_list.len(),
            "Mismatch in number of workers and number of requests: workers = {}, requests = {}",
            self.workers.len(),
            params_list.len(),
        );

        let mut handles = Vec::with_capacity(self.workers.len());

        for (i, worker) in self.workers.iter().enumerate() {
            let w = Arc::clone(worker);
            let p = params_list[i].clone();

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

pub type ModelWorkerGroup = WorkerGroupImpl<ModelWorker>;
pub type KVWorkerGroup = WorkerGroupImpl<KVWorker>;

impl ModelWorkerGroup {
    pub async fn init(num_workers: u8, worker_group_uds_path: String) -> Result<Self> {
        let mut workers: Vec<_> = Vec::with_capacity(num_workers as usize);
        for rank in 0..num_workers {
            let address = format!("unix://{}/model-{}", worker_group_uds_path, rank);
            let worker = ModelWorker::new(address.parse().unwrap())
                .await
                .unwrap_or_else(|e| panic!("Failed to connect worker: {e}"));
            workers.push(Arc::new(worker));
        }

        Ok(Self { workers })
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

    pub async fn init_cache(
        &self,
        gpu_cache_size: usize,
        host_cache_size: usize,
        disk_cache_size: usize,
        disk_cache_path: String,
    ) -> Result<(u64, u64, u64)> {
        let request = InitCacheRequest {
            gpu_cache_size: gpu_cache_size as u64,
            host_cache_size: host_cache_size as u64,
            disk_cache_size: disk_cache_size as u64,
            disk_cache_path,
        };
        let outputs = self
            .run_workers_gather(request, move |worker, request| async move {
                worker.init_cache(request).await
            })
            .await?;

        let (num_gpu_blocks, num_host_blocks, num_disk_blocks) =
            extract_if_all_equal(&outputs).expect("Error: not all values are equal");

        Ok((num_gpu_blocks, num_host_blocks, num_disk_blocks))
    }

    pub async fn infer(
        &self,
        inputs: Vec<InferInput>,
        wait_before_execute: bool,
        record_after_execute: bool,
    ) -> Result<HashMap<u64, InferOutput>> {
        let request = InferRequest {
            inputs,
            use_cache: true,
            wait_before_execute,
            record_after_execute,
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
}

impl KVWorkerGroup {
    pub async fn init(num_workers: u8, worker_group_uds_path: String) -> Result<Self> {
        let mut workers: Vec<_> = Vec::with_capacity(num_workers as usize);
        for rank in 0..num_workers {
            let address = format!("unix://{}/model-{}-kv", worker_group_uds_path, rank);
            let worker = KVWorker::new(address.parse().unwrap())
                .await
                .unwrap_or_else(|e| panic!("Failed to connect worker: {e}"));
            workers.push(Arc::new(worker));
        }

        Ok(Self { workers })
    }

    pub async fn copy_kv(
        &self,
        block_mappings: Vec<BlockMapping>,
        src_device: Device,
        dst_device: Device,
    ) -> Result<()> {
        let request = CopyKvRequest {
            block_mappings,
            src_device: src_device.into(),
            dst_device: dst_device.into(),
        };

        let _ = self
            .run_workers_gather(request, move |worker, request| async move {
                worker.copy_kv(request).await
            })
            .await?;

        Ok(())
    }

    pub async fn get_local_agent_metadata(&self) -> Result<Vec<(String, Vec<u8>)>> {
        let outputs = self
            .run_workers_gather((), move |worker, _| async move {
                worker.get_local_agent_metadata().await
            })
            .await?;

        Ok(outputs)
    }

    pub async fn add_remote_agent_metadata(
        &self,
        agent_metadata: Vec<(String, Vec<u8>)>,
    ) -> Result<Vec<String>> {
        let mut requests: Vec<_> = Vec::with_capacity(self.workers.len());
        for (agent_name, data) in agent_metadata {
            requests.push(AgentMetadata { agent_name, data })
        }

        let peer_names = self
            .run_workers(&requests, move |worker, request| async move {
                worker.add_remote_agent_metadata(request).await
            })
            .await?;

        Ok(peer_names)
    }

    pub async fn get_descriptors(&self, block_ids: Vec<u32>) -> Result<Vec<Bytes>> {
        let request = GetDescriptorsRequest { block_ids };
        let descs = self
            .run_workers_gather(request, move |worker, request| async move {
                worker.get_descriptors(request).await
            })
            .await?;

        Ok(descs)
    }

    pub async fn push_kv(
        &self,
        peer_names: Vec<String>,
        kv_descs: Vec<Bytes>,
        block_ids: Vec<u32>,
    ) -> Result<bool> {
        assert!(kv_descs.len() == self.workers.len());

        let mut requests: Vec<_> = Vec::with_capacity(self.workers.len());
        for (peer_name, descs) in peer_names.into_iter().zip(kv_descs) {
            requests.push(PushKvRequest {
                peer_name,
                kv_descs: descs,
                block_ids: block_ids.clone(),
            })
        }

        let rets = self
            .run_workers(&requests, move |worker, request| async move {
                worker.push_kv(request).await
            })
            .await?;

        let all_success = rets.iter().all(|&x| x);
        Ok(all_success)
    }

    #[allow(dead_code)]
    pub async fn pull_kv(
        &self,
        peer_names: Vec<String>,
        kv_descs: Vec<Bytes>,
        block_ids: Vec<u32>,
    ) -> Result<bool> {
        assert!(kv_descs.len() == self.workers.len());

        let mut requests: Vec<_> = Vec::with_capacity(self.workers.len());
        for (peer_name, descs) in peer_names.into_iter().zip(kv_descs) {
            requests.push(PullKvRequest {
                peer_name,
                descs,
                block_ids: block_ids.clone(),
            })
        }

        let rets = self
            .run_workers(&requests, move |worker, request| async move {
                worker.pull_kv(request).await
            })
            .await?;

        let all_success = rets.iter().all(|&x| x);
        Ok(all_success)
    }
}
