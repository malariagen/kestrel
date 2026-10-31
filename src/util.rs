use std::time::{Duration, Instant};

use bytesize::ByteSize;
use nix::sys::{resource::{Usage, UsageWho, getrusage}, time::TimeVal};

pub struct ResourceTracker {
    time: Instant,
    usage: Usage
}

impl ResourceTracker {
    pub fn new() -> Self {
        let time = Instant::now();
        let usage = getrusage(UsageWho::RUSAGE_SELF).expect("Failed to get usage");

        ResourceTracker { time, usage }
    }

    pub fn checkpoint(&self) {
        let now = Instant::now();
        let duration = now.duration_since(self.time);

        let usage = getrusage(UsageWho::RUSAGE_SELF).expect("Failed to get usage");

        let user_duration = timeval_duration(self.usage.user_time(), usage.user_time());
        let system_duration = timeval_duration(self.usage.system_time(), usage.system_time());

        let memory_kib = max_rss_to_kib(usage.max_rss());
        let memory = ByteSize::kib(memory_kib);

        log::info!("Wall Time {duration:.2?}, User Time {user_duration:.2?}, Kernel Time {system_duration:.2?}, Peak Memory {memory:.2}");
    }
}

fn max_rss_to_kib(maxrss: i64) -> u64 {
    // Macs use bytes
    // https://stackoverflow.com/questions/59913657/strange-values-of-get-rusage-maxrss-on-macos-and-linux
    #[cfg(target_os = "macos")]
    {
        u64::try_from(maxrss).unwrap() / 1024
    }
    // Everyone else uses kibibytes
    // https://stackoverflow.com/questions/12050913/whats-the-unit-of-ru-maxrss-on-linux
    #[cfg(not(target_os = "macos"))]
    {
        u64::try_from(maxrss).unwrap()
    }
}

fn timeval_duration(start: TimeVal, end: TimeVal) -> Duration {
    let s_to_us = 1_000_000;
    let start_micros = s_to_us * start.tv_sec() + start.tv_usec();
    let end_micros = s_to_us * end.tv_sec() + end.tv_usec();

    let micros_diff = u64::try_from(end_micros - start_micros).unwrap();

    Duration::from_micros(micros_diff)
}