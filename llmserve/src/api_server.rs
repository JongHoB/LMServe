use std::collections::HashMap;
use std::sync::Arc;

use axum::{Json, Router, http::StatusCode, response::IntoResponse, routing::post};
use clap::Parser;
use serde::{Deserialize, Serialize};
use tokenizers::tokenizer::{Result, Tokenizer};
use tokio::signal;
use tokio::sync::{Mutex, Notify};
use tracing::info;

use llmserve::args::LLMServeArgs;
use llmserve::infer_task::InferTask;
use llmserve::llmserve::LLMServe;
use llmserve::sequence::SeqStatus;
use llmserve::utils::{generate_session_id, norm_log_probs};

#[derive(Serialize)]
struct GenerateOutput {
    token_ids: Vec<u32>,
    output_text: String,
    output_len: usize,
}

#[derive(Deserialize)]
struct GenerateParams {
    prompt: String,
    num_samples: u16,
    session_id: Option<String>,
    max_output_len: Option<usize>,
    ignore_eos: bool,
}

struct APIServer {
    engine: Arc<LLMServe>,
    tokenizer: Tokenizer,

    request_events: Mutex<HashMap<String, Arc<Notify>>>,
    request_outputs: Mutex<HashMap<String, InferTask>>,
}

impl APIServer {
    fn new(engine: LLMServe, tokenizer: Tokenizer) -> APIServer {
        APIServer {
            engine: Arc::new(engine),
            tokenizer,
            request_events: Mutex::new(HashMap::new()),
            request_outputs: Mutex::new(HashMap::new()),
        }
    }

    async fn generate(&self, mut params: GenerateParams) -> Result<GenerateOutput> {
        let session_id = generate_session_id();
        params.session_id = Some(session_id.clone());

        let token_ids = self
            .tokenizer
            .encode(params.prompt, false)?
            .get_ids()
            .to_vec();

        let notify = Arc::new(Notify::new());
        {
            self.request_events
                .lock()
                .await
                .insert(session_id.clone(), notify.clone());
        }

        self.engine
            .add_request(
                token_ids,
                params.num_samples,
                Some(session_id.clone()),
                params.max_output_len,
                params.ignore_eos,
            )
            .await?;

        notify.notified().await;

        let request_output = {
            self.request_outputs
                .lock()
                .await
                .remove(&session_id)
                .unwrap()
        };
        let seqs = request_output.get_seqs(SeqStatus::FINISHED);
        let selected_seq = seqs
            .iter()
            .max_by(|a, b| {
                norm_log_probs(a.get_token_probs().as_ref())
                    .partial_cmp(&norm_log_probs(b.get_token_probs().as_ref()))
                    .unwrap()
            })
            .expect("Failed to select sequence: max_by() returned None.");

        let token_ids = selected_seq.get_token_ids();
        let output_ids = selected_seq.get_output_ids();
        let output_text = self.tokenizer.decode(output_ids, false)?;

        Ok(GenerateOutput {
            token_ids: token_ids.to_vec(),
            output_text,
            output_len: selected_seq.output_len,
        })
    }

    async fn run_engine(&self) -> Result<()> {
        loop {
            let infer_tasks = self.engine.iter().await;
            let Some(infer_tasks) = infer_tasks else {
                continue;
            };
            for infer_task in infer_tasks.into_iter() {
                let session_id = infer_task.get_session_id();

                self.request_outputs
                    .lock()
                    .await
                    .insert(session_id.clone(), infer_task);

                let request_event = self
                    .request_events
                    .lock()
                    .await
                    .remove(&session_id)
                    .unwrap();
                request_event.notify_waiters()
            }
        }
    }
}

#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
async fn main() {
    tracing_subscriber::fmt::init();

    let args = LLMServeArgs::parse();
    let socket_addr = format!("{}:{}", args.address, args.port);
    let url = format!("http://{}", socket_addr);

    let tokenizer =
        Tokenizer::from_pretrained(&args.model_name, None).expect("Failed to load tokenizer");

    let llmserve = LLMServe::new(
        args.model_name,
        args.block_size,
        args.gpu_memory_fraction,
        args.max_batch_size,
        args.max_seq_len,
        args.max_num_batched_tokens,
        args.tp_size,
    )
    .await
    .expect("Failed to start API Server");

    let api_server = Arc::new(APIServer::new(llmserve, tokenizer));

    let api_server_clone = api_server.clone();
    tokio::spawn(async move {
        api_server_clone.run_engine().await.unwrap();
    });

    // Build API server with a route.
    let app = Router::new().route(
        "/generate",
        post(move |Json(params): Json<GenerateParams>| async move {
            match api_server.generate(params).await {
                Ok(output) => (StatusCode::OK, Json(output)).into_response(),
                Err(err) => {
                    eprintln!("Failed to generate: {:?}", err);
                    (StatusCode::INTERNAL_SERVER_ERROR, "Generation failed").into_response()
                }
            }
        }),
    );

    let listener = tokio::net::TcpListener::bind(socket_addr).await.unwrap();
    info!("LLMServe started to {url}");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .unwrap();
}

async fn shutdown_signal() {
    // Ctrl+C or termination signal (SIGINT / SIGTERM)
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler.");
    };

    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler.")
            .recv()
            .await;
    };

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    };

    info!("signal received, starting graceful shutdown.");
}
