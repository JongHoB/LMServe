use std::fmt;

use chrono::Local;
use tracing_subscriber::fmt::format::Writer;
use tracing_subscriber::fmt::time::FormatTime;

struct Date;

impl FormatTime for Date {
    fn format_time(&self, w: &mut Writer<'_>) -> fmt::Result {
        write!(w, "{}", Local::now().format("%Y-%m-%dT%H:%M:%SZ"))
    }
}

pub fn init_tracing() {
    tracing_subscriber::fmt::Subscriber::builder()
        .with_max_level(tracing::Level::INFO)
        .with_timer(Date)
        .with_target(true)
        .init();
}
