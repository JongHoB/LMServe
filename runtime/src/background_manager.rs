use std::pin::Pin;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::thread::JoinHandle;

use anyhow::Result;
use tokio::{
    runtime::Builder,
    sync::{Notify, mpsc},
};
use tracing::{debug, warn};

type AsyncResult = Result<(), Box<dyn std::error::Error + Send + Sync>>;
type AsyncFunc = Pin<Box<dyn Future<Output = AsyncResult> + Send + 'static>>;

pub struct AsyncTask {
    fut: AsyncFunc,
    on_complete: AsyncFunc,
}

pub struct BackgroundTaskManager {
    handle: Option<JoinHandle<()>>,
    sender: mpsc::UnboundedSender<AsyncTask>,
    shutdown_tx: tokio::sync::watch::Sender<bool>,
    is_shutdown: Arc<AtomicBool>,
}

impl BackgroundTaskManager {
    pub fn new() -> Arc<Self> {
        let (tx, rx) = mpsc::unbounded_channel::<AsyncTask>();
        let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
        let is_shutdown = Arc::new(AtomicBool::new(false));

        let handle = std::thread::spawn(move || {
            let rt = Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("Failed to build background thread");

            if let Err(e) = rt.block_on(run_async(rx, shutdown_rx)) {
                warn!("Background loop exit with error: {e:?}");
            };
        });

        Arc::new(Self {
            handle: Some(handle),
            sender: tx,
            shutdown_tx,
            is_shutdown: is_shutdown.clone(),
        })
    }

    pub fn submit<F, C>(&self, fut: F, cb: C)
    where
        F: Future<Output = AsyncResult> + Send + 'static,
        C: Future<Output = AsyncResult> + Send + 'static,
    {
        if self.is_shutdown.load(Ordering::SeqCst) {
            eprintln!("Cannot submit task: shuted down");
            return;
        }
        let task = AsyncTask {
            fut: Box::pin(fut),
            on_complete: Box::pin(cb),
        };
        if let Err(e) = self.sender.send(task) {
            eprintln!("Failed to submit task: {e}");
        }
    }

    pub async fn wait(&self) {
        let notify = Arc::new(Notify::new());

        let sync_event = {
            let notify_clone = notify.clone();
            async move {
                notify_clone.notify_one();
                Ok(())
            }
        };

        self.submit(sync_event, async move { Ok(()) });

        notify.notified().await;
    }

    pub fn shutdown(&mut self) -> Result<()> {
        if self.is_shutdown.swap(true, Ordering::SeqCst) {
            return Ok(());
        }

        let _ = self.shutdown_tx.send(true);

        if let Some(handle) = self.handle.take() {
            handle.join().map_err(|e| anyhow::anyhow!("{:?}", e))?;
        }

        debug!("Background thread stopped");
        Ok(())
    }
}

impl Drop for BackgroundTaskManager {
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}

async fn run_async(
    mut reciever: mpsc::UnboundedReceiver<AsyncTask>,
    mut shutdown_rx: tokio::sync::watch::Receiver<bool>,
) -> Result<()> {
    loop {
        tokio::select! {
            _ = shutdown_rx.changed() => { break; }
            maybe_task = reciever.recv() => {
                match maybe_task {
                    Some(task) => {
                        if let Err(e) = task.fut.await {
                            eprintln!("Failed to future: {e}");
                            continue;
                        }

                        if let Err(e) = task.on_complete.await {
                            eprintln!("Failed to callback: {e}");
                            continue;
                        }
                    },

                    None => {
                        eprintln!("Background task receiver is closed so all taskss are dropped.");
                        break;
                    }
                }
            }
        }
    }

    Ok(())
}
