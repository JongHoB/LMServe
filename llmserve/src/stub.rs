use anyhow::Result;

use crate::pb::llm::GetDescriptorsRequest;
use crate::pb::llm::llm_client::LlmClient;

type Bytes = Vec<u8>;

pub struct LLMEngineStub {}

impl LLMEngineStub {
    pub async fn get_descriptors(
        remote_url: String,
        token_ids: Vec<u32>,
        start: usize,
        end: usize,
    ) -> Result<(Vec<Bytes>, usize)> {
        let request = GetDescriptorsRequest {
            token_ids,
            start: start as u64,
            end: end as u64,
        };

        let mut client = LlmClient::connect(remote_url.clone()).await?;
        let response = client.get_descriptors(request).await?.into_inner();

        let descs = response.descs;
        let last_token_idx = response.last_token_idx;

        Ok((descs, last_token_idx as usize))
    }

    pub async fn get_remote_kv_agent_metadata(remote_url: String) -> Result<Vec<Bytes>> {
        let mut client = LlmClient::connect(remote_url.clone()).await?;
        let response = client.get_kv_agent_metadata(()).await?.into_inner();

        let metadata = response.metadata;

        Ok(metadata)
    }
}
