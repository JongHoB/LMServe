use std::sync::Arc;

use axum::{Json, Router, http::StatusCode, response::IntoResponse, routing::post};
use clap::Parser;
use serde::{Deserialize, Serialize};
use tokenizers::tokenizer::{Result, Tokenizer};
use tracing::info;

use runtime::pb::llm::GenerateRequest;
use runtime::router::Controller;
use utils::tokenizer::get_tokenizer;

use clis::args::APIServerArgs;

#[derive(Serialize)]
pub struct GenerateOutput {
    pub session_id: String,
    pub token_ids: Vec<u32>,
    pub output_text: String,
    pub output_len: usize,
    pub token_latencies: Vec<f32>,
}

#[derive(Deserialize)]
pub struct GenerateParams {
    pub prompt: String,
    pub num_samples: u16,
    pub session_id: Option<String>,
    pub max_output_len: Option<u64>,
    pub ignore_eos: bool,
    pub disable_cache: bool,
}

pub struct APIServer {
    pub controller: Controller,
    pub tokenizer: Tokenizer,
}

impl APIServer {
    pub fn new(controller: Controller, tokenizer: Tokenizer) -> Self {
        Self {
            controller,
            tokenizer,
        }
    }

    pub async fn generate(&self, params: GenerateParams) -> Result<GenerateOutput> {
        // If no session ID is given, generate a new one
        let session_id = params
            .session_id
            .unwrap_or(utils::random::generate_session_id());

        let input_ids = self
            .tokenizer
            .encode(params.prompt, false)?
            .get_ids()
            .to_vec();

        let response = self
            .controller
            .generate(GenerateRequest {
                session_id,
                input_ids: input_ids.clone(),
                num_samples: params.num_samples as u32,
                max_output_len: params.max_output_len,
                ignore_eos: params.ignore_eos,
                disable_cache: params.disable_cache,
            })
            .await?;

        let session_id = response.session_id;
        let output_ids = response.output_ids;
        let output_text = self.tokenizer.decode(&output_ids, false)?;
        let output_len = output_ids.len();

        Ok(GenerateOutput {
            session_id,
            token_ids: [input_ids, output_ids].concat(),
            output_text,
            output_len,
            token_latencies: response.token_latencies,
        })
    }

    pub async fn clear_cache(&self) -> Result<()> {
        self.controller.clear_cache().await?;
        Ok(())
    }
}

#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
async fn main() -> Result<()> {
    utils::logging::init_tracing();

    let args = APIServerArgs::parse();

    let socket_addr = args.address;

    let url = format!("http://{}", socket_addr);

    let tokenizer =
        get_tokenizer(&args.model_name).unwrap_or_else(|e| panic!("Failed to load tokenizer: {e}"));

    let controller: Controller = Controller::new(args.route_policy, args.nats_uri).await;
    for addr in args.llm_server_addresses.iter() {
        let llm_server_url = format!("http://{}", addr);
        controller
            .add_node(&llm_server_url)
            .await
            .unwrap_or_else(|e| panic!("Unable to establish connection to engine ({addr}): {e}"));

        info!("Successfully connected to LLM server at {}", addr);
    }

    let api_server = Arc::new(APIServer::new(controller, tokenizer));

    // Build API server with a route.
    let app = Router::new()
        .route(
            "/generate",
            post({
                let api_server = api_server.clone();
                move |Json(params): Json<GenerateParams>| async move {
                    match api_server.generate(params).await {
                        Ok(output) => (StatusCode::OK, Json(output)).into_response(),
                        Err(err) => {
                            eprintln!("Failed to generate: {:?}", err);
                            (StatusCode::INTERNAL_SERVER_ERROR, "Generation failed").into_response()
                        }
                    }
                }
            }),
        )
        .route(
            "/clear_cache",
            post({
                let api_server = api_server.clone();
                move || async move {
                    match api_server.clear_cache().await {
                        Ok(_) => (StatusCode::OK).into_response(),
                        Err(err) => {
                            eprintln!("Failed to clear cacahe: {:?}", err);
                            (StatusCode::INTERNAL_SERVER_ERROR, "Clear cache failed")
                                .into_response()
                        }
                    }
                }
            }),
        );

    let listener = tokio::net::TcpListener::bind(socket_addr).await.unwrap();
    info!("API Server started to {url}");

    axum::serve(listener, app)
        .with_graceful_shutdown(utils::signal_handler::wait_shutdown_signal())
        .await
        .unwrap();

    info!("API Server terminated.");

    Ok(())
}
