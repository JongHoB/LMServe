use std::sync::Arc;

use anyhow::Result;
use clap::Parser;
use tokio::sync::Notify;
use tonic::transport::Server;
use tonic::{Request, Response, Status};
use tracing::debug;

use llmserve::args::LLMEngineArgs;
use llmserve::engine::LLMEngineWrapper;
use llmserve::pb::llm::llm_server::{Llm, LlmServer};
use llmserve::pb::llm::{GenerateRequest, GenerateResponse};

pub struct LLMService {
    engine: Arc<LLMEngineWrapper>,
}

#[tonic::async_trait]
impl Llm for LLMService {
    async fn generate(
        &self,
        request: Request<GenerateRequest>,
    ) -> Result<Response<GenerateResponse>, Status> {
        let generate_request = request.into_inner();

        let session_id = generate_request.session_id;
        let input_ids = generate_request.input_ids;
        let num_samples = generate_request.num_samples;
        let max_output_len = generate_request.max_output_len;
        let ignore_eos = generate_request.ignore_eos;

        let output = self
            .engine
            .generate(
                input_ids,
                num_samples as u16,
                session_id,
                max_output_len.map(|x| x as usize),
                ignore_eos,
            )
            .await
            .expect("Failed to generate");

        Ok(Response::new(GenerateResponse {
            session_id: output.session_id,
            output_ids: output.output_ids,
        }))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    utils::logging::init_tracing();

    let shutdown_notify = Arc::new(Notify::new());
    let shutdown_signal = shutdown_notify.clone();

    tokio::spawn(async move {
        let shutdown_notify = shutdown_notify.clone();
        utils::signal_handler::wait_shutdown_signal().await;
        shutdown_notify.notify_waiters();
    });

    let args = LLMEngineArgs::parse();

    let address = format!("[::1]:{}", args.port).parse().unwrap();

    let engine = Arc::new(
        LLMEngineWrapper::new(
            args.model_name,
            args.block_size,
            args.gpu_memory_fraction,
            args.max_batch_size,
            args.max_seq_len,
            args.max_num_batched_tokens,
            args.tp_size,
        )
        .await
        .expect("Failed to start API Server"),
    );

    let llm_service = LLMService {
        engine: engine.clone(),
    };

    // Run the engine asynchronously in the background
    let engine_clone = engine.clone();
    tokio::spawn(async move {
        engine_clone.run_engine().await.unwrap();
    });

    let svc = LlmServer::new(llm_service);

    debug!("LLMServer listening on: {address}");

    Server::builder()
        .add_service(svc)
        .serve_with_shutdown(address, async move { shutdown_signal.notified().await })
        .await?;

    Ok(())
}
