use std::{collections::HashMap, sync::Arc, thread::JoinHandle};

use anyhow::Result;
use async_nats;
use futures::StreamExt;
use serde_json;
use tokio::{
    runtime::Builder,
    sync::{RwLock, oneshot},
};
use tracing::{info, warn};

use crate::stats::Stats;

type ShareMap = Arc<RwLock<HashMap<String, Stats>>>;

pub struct StatsMonitor {
    shut_tx: Option<oneshot::Sender<()>>,
    handle: Option<JoinHandle<()>>,
}

impl StatsMonitor {
    pub fn start(nats_uri: &str) -> Result<Self> {
        let uri = nats_uri.to_string();

        let (shut_tx, shut_rx) = oneshot::channel();

        let handle = std::thread::spawn(move || {
            let rt = Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("Failed to build monitor thread");
            if let Err(e) = rt.block_on(run_async(&uri, shut_rx)) {
                warn!("monitor exit with error: {e:?}");
            };
        });

        Ok(Self {
            shut_tx: Some(shut_tx),
            handle: Some(handle),
        })
    }

    pub fn shutdown(&mut self) -> Result<()> {
        if let Some(tx) = self.shut_tx.take() {
            let _ = tx.send(());
        }

        if let Some(handle) = self.handle.take() {
            handle.join().map_err(|e| anyhow::anyhow!("{:?}", e))?;
        }
        info!("Stats monitor daemon stopped");
        Ok(())
    }
}

impl Drop for StatsMonitor {
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}

async fn run_async(uri: &str, mut shut_rx: oneshot::Receiver<()>) -> Result<(), async_nats::Error> {
    let nc = async_nats::connect(uri).await?;
    let updates = nc.subscribe("stats.update.*").await?;
    let queries = nc.subscribe("stats.query.*").await?;

    let store: ShareMap = ShareMap::default();

    let s1 = store.clone();
    let mut updates_handle = tokio::spawn(async move {
        let mut updates = updates;
        while let Some(msg) = updates.next().await {
            if let Some(worker_id) = msg.subject.split('.').nth(2) {
                if let Ok(stat) = serde_json::from_slice::<Stats>(&msg.payload) {
                    s1.write().await.insert(worker_id.to_string(), stat);
                }
            }
        }
    });

    let s2 = store.clone();
    let nc_clone = nc.clone();
    let mut queries_handle = tokio::spawn(async move {
        let mut queries = queries;
        while let Some(msg) = queries.next().await {
            let reply_to = match msg.reply {
                Some(r) => r,
                None => continue,
            };

            let parts: Vec<&str> = msg.subject.split('.').collect();
            let resp = match parts.as_slice() {
                ["stats", "query", "all"] => serde_json::to_vec(&*s2.read().await)?,
                ["stats", "query", "list"] => {
                    let ids: Vec<String> = match serde_json::from_slice(&msg.payload) {
                        Ok(v) => v,
                        Err(e) => {
                            warn!("Bad list payload: {e}");
                            Vec::new()
                        }
                    };

                    let map = {
                        let guard = s2.read().await;
                        ids.into_iter()
                            .filter_map(|id| guard.get(&id).cloned().map(|st| (id, st)))
                            .collect::<HashMap<_, _>>()
                    };
                    serde_json::to_vec(&map)?
                }
                ["stats", "query", id] => {
                    let maybe = s2.read().await.get(&id.to_string()).cloned();
                    serde_json::to_vec(&maybe)?
                }
                _ => b"{}".to_vec(),
            };

            nc_clone.publish(reply_to, resp.into()).await.unwrap();
        }
        Ok::<(), async_nats::Error>(())
    });

    info!("Stats monitor daemon started.");

    tokio::select! {
    _ = &mut shut_rx => {info!("shutdown signal received");}
    res = (&mut updates_handle) => {warn!("Update task ended: {res:?}");}
    res = (&mut queries_handle) => {warn!("Query task ended: {res:?}");}
    }

    updates_handle.abort();
    queries_handle.abort();

    nc.flush().await?;
    Ok(())
}
