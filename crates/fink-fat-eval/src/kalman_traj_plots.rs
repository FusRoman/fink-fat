//! Per-trajectory NIS/NEES time-series charts, reproducing the style of
//! <https://kalman-filter.com/normalized-estimation-error-squared/>: the
//! metric plotted against step index, with the χ² 95% confidence band drawn
//! as horizontal reference lines.
//!
//! Reuses the generic [`draw_line_chart`] primitive from
//! [`crate::seed_bank_report::plots`] instead of duplicating `plotters`
//! boilerplate.

use anyhow::Result;
use camino::Utf8Path;
use plotters::prelude::*;

use crate::kalman_traj::{
    KFStudyResult, NEES_CHI2_2DOF_HIGH, NEES_CHI2_2DOF_LOW, NEES_CHI2_6DOF_HIGH,
    NEES_CHI2_6DOF_LOW, NIS_CHI2_2DOF_HIGH, NIS_CHI2_2DOF_LOW,
};
use crate::seed_bank_report::plots::{Series, draw_line_chart};

/// Plot NIS per step, with the χ²(2) 95% confidence band.
pub fn plot_nis_chart(results: &[KFStudyResult], output_path: &Utf8Path) -> Result<()> {
    let points = results
        .iter()
        .enumerate()
        .map(|(i, r)| ((i + 1) as f64, r.nis))
        .filter(|(_, y)| y.is_finite())
        .collect();

    draw_line_chart(
        output_path,
        "NIS per step (χ²(2) 95% band)",
        "Step",
        "NIS",
        &[Series {
            label: "NIS",
            color: BLUE,
            points,
        }],
        &[
            (NIS_CHI2_2DOF_LOW, RED, "χ²(2) 2.5%"),
            (NIS_CHI2_2DOF_HIGH, RED, "χ²(2) 97.5%"),
        ],
    )
}

/// Plot the 2-D sky-plane NEES per step, with the χ²(2) 95% confidence band.
/// Steps without ground truth (`nees_sky == None`) are skipped.
pub fn plot_nees_sky_chart(results: &[KFStudyResult], output_path: &Utf8Path) -> Result<()> {
    let points = results
        .iter()
        .enumerate()
        .filter_map(|(i, r)| r.nees_sky.map(|v| ((i + 1) as f64, v)))
        .filter(|(_, y)| y.is_finite())
        .collect();

    draw_line_chart(
        output_path,
        "NEES sky-plane per step (χ²(2) 95% band)",
        "Step",
        "NEES",
        &[Series {
            label: "NEES (sky)",
            color: BLUE,
            points,
        }],
        &[
            (NEES_CHI2_2DOF_LOW, RED, "χ²(2) 2.5%"),
            (NEES_CHI2_2DOF_HIGH, RED, "χ²(2) 97.5%"),
        ],
    )
}

/// Plot the full 6-DOF Cartesian NEES per step, with the χ²(6) 95%
/// confidence band. Steps without ground truth (`nees_cart == None`) are
/// skipped.
pub fn plot_nees_cart_chart(results: &[KFStudyResult], output_path: &Utf8Path) -> Result<()> {
    let points = results
        .iter()
        .enumerate()
        .filter_map(|(i, r)| r.nees_cart.map(|v| ((i + 1) as f64, v)))
        .filter(|(_, y)| y.is_finite())
        .collect();

    draw_line_chart(
        output_path,
        "NEES Cartesian per step (χ²(6) 95% band)",
        "Step",
        "NEES",
        &[Series {
            label: "NEES (cartesian)",
            color: BLUE,
            points,
        }],
        &[
            (NEES_CHI2_6DOF_LOW, RED, "χ²(6) 2.5%"),
            (NEES_CHI2_6DOF_HIGH, RED, "χ²(6) 97.5%"),
        ],
    )
}
