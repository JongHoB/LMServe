use std::sync::Arc;

use axum::{Json, Router, http::StatusCode, response::IntoResponse, routing::post};
use clap::Parser;
use serde::{Deserialize, Serialize};
use tokenizers::tokenizer::{Result, Tokenizer};
use tracing::info;

use api_server::args::APIServerArgs;
use api_server::pb::llm::GenerateRequest;
use api_server::router::EngineRouter;

#[derive(Serialize)]
struct GenerateOutput {
    session_id: String,
    token_ids: Vec<u32>,
    output_text: String,
    output_len: usize,
}

#[derive(Deserialize)]
struct GenerateParams {
    prompt: String,
    num_samples: u16,
    session_id: Option<String>,
    max_output_len: Option<u64>,
    ignore_eos: bool,
}

struct APIServer {
    router: EngineRouter,
    tokenizer: Tokenizer,
}

impl APIServer {
    fn new(router: EngineRouter, tokenizer: Tokenizer) -> Self {
        Self { router, tokenizer }
    }

    async fn generate(&self, params: GenerateParams) -> Result<GenerateOutput> {
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
            .router
            .generate(GenerateRequest {
                session_id: session_id,
                input_ids: input_ids.clone(),
                num_samples: params.num_samples as u32,
                max_output_len: params.max_output_len,
                ignore_eos: params.ignore_eos,
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
        })
    }
}

#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
async fn main() {
    utils::logging::init_tracing();

    let args = APIServerArgs::parse();

    let socket_addr = format!("{}:{}", args.address, args.port);
    let url = format!("http://{}", socket_addr);

    let tokenizer =
        Tokenizer::from_pretrained(&args.tokenier_name, None).expect("Failed to load tokenizer");

    let mut engine_router = EngineRouter::new();
    for engine_url in args.engine_urls.iter() {
        let _ = engine_router
            .add_node(engine_url)
            .await
            .unwrap_or_else(|e| {
                panic!("Unable to establish connection to engine ({engine_url}): {e}")
            });
    }

    let api_server = Arc::new(APIServer::new(engine_router, tokenizer));

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
    info!("API Server started to {url}");

    axum::serve(listener, app)
        .with_graceful_shutdown(utils::signal_handler::wait_shutdown_signal())
        .await
        .unwrap();
}
