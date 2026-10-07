use anyhow::Result;
use kestrel::cli::run_cli;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    run_cli(&args)
}
