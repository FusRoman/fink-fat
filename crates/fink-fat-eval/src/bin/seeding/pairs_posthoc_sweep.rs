//! Post-hoc sweep for pair-seeding thresholds (diagnostic plots).
//!
//! See also
//! --------
//! * [`fink_fat_eval::seeding::pairs_sweep`] – reusable post-hoc sweep pipeline.

use anyhow::Result;
use clap::{ArgAction, Parser};

use fink_fat_engine::night_id::NightId;
use fink_fat_eval::{
    angular_speed::AngularSpeed,
    cli::common::{CommonBinningArgs, CommonPairGenArgs, CommonPlotArgs, CommonScanArgs},
    dataset::{ingest_config::AlertIngestConfig, ztf_alerts::ZtfAlertScan},
    seeding::pairs_sweep::{PairsPosthocSweepConfig, default_plot_config, run_pairs_posthoc_sweep},
};

/// Command-line interface for the post-hoc pair sweep tool.
///
/// This CLI composes common scan + generation + binning + plot arguments,
/// and adds sweep-specific options (min ω, number of steps, log spacing).
#[derive(Parser, Debug)]
#[command(
    name = "pairs-posthoc-sweep",
    version,
    about = "Generate pairs with a large max_omega, then post-hoc sweep omega thresholds and plot metrics.",
    disable_help_subcommand = true,
    verbatim_doc_comment
)]
struct Cli {
    #[command(flatten)]
    scan: CommonScanArgs,

    #[command(flatten)]
    pair_gen: CommonPairGenArgs,

    #[command(flatten)]
    binning: CommonBinningArgs,

    #[command(flatten)]
    plot: CommonPlotArgs,

    /// Minimum angular-speed threshold for the post-hoc sweep.
    ///
    /// The sweep evaluates cuts of the form `sep <= omega * dt` starting from
    /// this minimum value up to `--gen-max-omega`.
    #[arg(
        long,
        default_value = "1 arcsec/min",
        value_name = "OMEGA",
        help_heading = "Sweep"
    )]
    sweep_min_omega: AngularSpeed,

    /// Number of omega thresholds evaluated in the sweep.
    ///
    /// This controls the resolution of the tradeoff curve:
    /// - too small → coarse diagnostics,
    /// - too large → diminishing returns and longer runtime.
    #[arg(long, default_value_t = 40, value_name = "N", help_heading = "Sweep")]
    sweep_steps: usize,

    /// Use logarithmic spacing for the omega threshold sweep.
    ///
    /// Log spacing is strongly recommended when thresholds span several
    /// orders of magnitude, as it provides better resolution at small
    /// omega where performance often changes rapidly.
    ///
    /// Disable with `--no-logspace`.
    #[arg(
        long,
        default_value_t = true,
        action = ArgAction::Set,
        help_heading = "Sweep"
    )]
    logspace: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let cfg = PairsPosthocSweepConfig {
        parquet: cli.scan.parquet,
        out_dir: cli.scan.out_dir,

        scan: ZtfAlertScan {
            nid: cli.scan.nid.map(NightId),
            mode: cli.scan.mode.into(),
            minimal: cli.scan.minimal,
        },
        ingest: AlertIngestConfig::default(),

        max_dt_days: cli.pair_gen.max_dt,

        gen_max_angular_speed_rad_per_day: cli.pair_gen.gen_max_omega.as_rad_per_day(),
        sweep_min_angular_speed_rad_per_day: cli.sweep_min_omega.as_rad_per_day(),
        sweep_steps: cli.sweep_steps,
        logspace: cli.logspace,

        max_flux_difference: cli.pair_gen.max_flux_difference,
        allow_same_timebin: cli.pair_gen.allow_same_timebin,

        healpix_depth: cli.binning.healpix_depth,
        time_bin_days: cli.binning.time_bin_days,

        plot: default_plot_config(cli.plot.width, cli.plot.height, cli.plot.angular_unit),
    };

    run_pairs_posthoc_sweep(&cfg)?;
    eprintln!("Wrote plots to {}", cfg.out_dir);
    Ok(())
}
