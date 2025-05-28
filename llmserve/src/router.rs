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

use crate::pb::router::router_client::RouterClient;
use crate::pb::router::{ForwardRequest, ForwardResponse};

use anyhow::Result;

pub struct Router {
    clients: HashMap<String, Arc<Mutex<RouterClient<Channel>>>>,
}

impl Router {
    pub fn new() -> Router {
        Router {
            clients: HashMap::new(),
        }
    }

    pub async fn add_node(&mut self, url: String) -> Result<()> {
        let client = RouterClient::connect(url.clone()).await?;
        self.clients.insert(url, Arc::new(Mutex::new(client)));
        Ok(())
    }

    pub async fn forward(&self, url: String, request: ForwardRequest) -> Result<ForwardResponse> {
        let client = self
            .clients
            .get(&url)
            .unwrap_or_else(|| {
                panic!("Not found client matching {}", url)
            });
        let response = client.lock().await.forward(request).await?.into_inner();
        Ok(response)
    }
}
