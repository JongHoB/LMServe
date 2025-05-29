use std::time::Duration;

use tokio::time::sleep;
use tonic::transport::{Channel, Endpoint};
use tracing::debug;

use crate::pb::llm::llm_client::LlmClient;
use crate::pb::llm::{GenerateRequest, GenerateResponse};

use anyhow::Result;

pub struct EngineRouter {
    engine_endpoints: Vec<Endpoint>,
}

impl EngineRouter {
    pub fn new() -> Self {
        Self {
            engine_endpoints: Vec::new(),
        }
    }

    pub async fn add_node(&mut self, url: &String) -> Result<()> {
        let url: String = url.parse().unwrap();
        let endpoint = Channel::from_shared(url.clone().into_bytes())?;

        // TODO(jinu): Add timeout.
        let _ = loop {
            match LlmClient::connect(endpoint.clone()).await {
                Ok(client) => {
                    break client;
                }
                Err(error) => {
                    debug!("Trying to connect to engine server ({}): {:?}", url, error);
                }
            };

            sleep(Duration::from_millis(500)).await;
        };

        self.engine_endpoints.push(endpoint);
        Ok(())
    }

    // FIXME(jinu)
    pub async fn generate(&self, request: GenerateRequest) -> Result<GenerateResponse> {
        let endpoint = self
            .engine_endpoints
            .first()
            .unwrap_or_else(|| panic!("Not found client matching"));

        let mut client = LlmClient::connect(endpoint.clone()).await?;

        let response = client.generate(request).await?.into_inner();

        Ok(response)
    }
}
