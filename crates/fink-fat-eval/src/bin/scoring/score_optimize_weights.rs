use anyhow::Result;
use clap::Parser;
use fink_fat_eval::scoring::optimizer_pipeline::{Cli, run};

// (garde ici ta struct Cli telle quelle)

fn main() -> Result<()> {
    let cli = Cli::parse();
    run(&cli)
}
