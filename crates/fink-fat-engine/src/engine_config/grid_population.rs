use serde::{Deserialize, Serialize};

/// A small-body population, modeled as a Gaussian prior over semi-major axis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Population {
    /// Human-readable label (NEO, MBA, …).
    pub name: String,
    /// Peak semi-major axis of the population (AU).
    pub a_center: f64,
    /// Spread of the population in semi-major axis (AU).
    pub a_sigma: f64,
    /// Relative abundance (un-normalized; only ratios matter).
    pub weight: f64,
}

/// Rough default population priors covering the main dynamical families
/// of the Solar System, from Near-Earth Objects out to the distant Kuiper Belt.
///
/// Population model
/// ----------------
/// Each [`Population`] is a Gaussian kernel $\mathcal{N}(a_\text{center},\,
/// a_\sigma^2)$ over semi-major axis $a$ (AU). The `weight` field encodes
/// relative abundance; only ratios between weights matter — the sampler
/// normalises them internally.
///
/// Populations included
/// --------------------
///
/// | Name            | $a_\text{center}$ (AU) | $a_\sigma$ (AU) | Notes                                      |
/// |-----------------|------------------------|------------------|--------------------------------------------|
/// | NEO             | 1.50                   | 0.60             | Amors, Apollos, Atens, Atiras               |
/// | MBA Inner       | 2.20                   | 0.15             | Inside the 3:1 Kirkwood gap                |
/// | MBA Middle      | 2.70                   | 0.15             | Between 3:1 and 5:2 gaps                   |
/// | MBA Outer       | 3.10                   | 0.15             | Between 5:2 and 2:1 gaps                   |
/// | Cybele          | 3.42                   | 0.10             | Just outside the 2:1 resonance             |
/// | Hilda           | 3.95                   | 0.15             | 3:2 mean-motion resonance with Jupiter     |
/// | Trojan          | 5.20                   | 0.30             | Jupiter L4/L5 clouds                       |
/// | Centaur         | 15.00                  | 5.00             | Scattered between Jupiter and Neptune      |
/// | KBO Cold        | 44.00                  | 2.00             | Cold classical Kuiper Belt                 |
/// | KBO Hot         | 45.00                  | 8.00             | Hot classical Kuiper Belt                  |
/// | KBO Resonant    | 39.40                  | 1.00             | Plutinos — 3:2 resonance with Neptune      |
/// | SDO             | 70.00                  | 20.00            | Scattered-disc objects                     |
/// | Detached / ETNO | 90.00                  | 10.00            | Detached / extreme TNOs up to ~100 AU      |
///
/// Design intent
/// -------------
/// These priors are deliberately broad. Within a few nights of observations
/// the data resolves the semi-major axis tightly enough that the prior only
/// needs to seed plausible modes and prevent the hypothesis bank from wasting
/// samples on dynamically empty regions of element space.
///
/// The three MBA sub-populations replace the single broad kernel used
/// previously. This avoids over-smoothing across the major Kirkwood gaps
/// at 2.50 AU (3:1), 2.82 AU (5:2) and 2.95 AU (7:3).
pub fn default_populations() -> Vec<Population> {
    vec![
        // --- Near-Earth Objects ---
        Population {
            name: "NEO".to_string(),
            a_center: 1.50,
            a_sigma: 0.60,
            weight: 0.02,
        },
        // --- Main Belt (three sub-populations separated by Kirkwood gaps) ---
        Population {
            name: "MBA Inner".to_string(),
            a_center: 2.20,
            a_sigma: 0.15,
            weight: 0.60,
        },
        Population {
            name: "MBA Middle".to_string(),
            a_center: 2.70,
            a_sigma: 0.15,
            weight: 1.00,
        },
        Population {
            name: "MBA Outer".to_string(),
            a_center: 3.10,
            a_sigma: 0.15,
            weight: 0.70,
        },
        // --- Cybele group (just outside the 2:1 resonance at 3.28 AU) ---
        Population {
            name: "Cybele".to_string(),
            a_center: 3.42,
            a_sigma: 0.10,
            weight: 0.04,
        },
        // --- Hilda group (3:2 resonance with Jupiter) ---
        Population {
            name: "Hilda".to_string(),
            a_center: 3.95,
            a_sigma: 0.15,
            weight: 0.05,
        },
        // --- Jupiter Trojans (L4 / L5) ---
        Population {
            name: "Trojan".to_string(),
            a_center: 5.20,
            a_sigma: 0.30,
            weight: 0.10,
        },
        // --- Centaurs ---
        Population {
            name: "Centaur".to_string(),
            a_center: 15.00,
            a_sigma: 5.00,
            weight: 0.03,
        },
        // --- Kuiper Belt — cold classical disk ---
        Population {
            name: "KBO Cold".to_string(),
            a_center: 44.00,
            a_sigma: 2.00,
            weight: 0.04,
        },
        // --- Kuiper Belt — hot classical disk ---
        Population {
            name: "KBO Hot".to_string(),
            a_center: 45.00,
            a_sigma: 8.00,
            weight: 0.03,
        },
        // --- Plutinos (3:2 resonance with Neptune at ~39.4 AU) ---
        Population {
            name: "KBO Resonant".to_string(),
            a_center: 39.40,
            a_sigma: 1.00,
            weight: 0.02,
        },
        // --- Scattered-disc objects ---
        Population {
            name: "SDO".to_string(),
            a_center: 70.00,
            a_sigma: 20.00,
            weight: 0.02,
        },
        // --- Detached / extreme TNOs (up to ~100 AU) ---
        Population {
            name: "Detached".to_string(),
            a_center: 90.00,
            a_sigma: 10.00,
            weight: 0.01,
        },
    ]
}

/// Configuration for [`admissible_region_grid`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridConfig {
    /// Minimum topocentric range (AU). Excludes near-Earth/geocentric objects.
    pub rho_min: f64,
    /// Maximum topocentric range (AU).
    pub rho_max: f64,
    /// Number of range nodes (log-spaced between `rho_min` and `rho_max`).
    pub n_rho: usize,
    /// Number of range-rate nodes per range (linear, inside the bound interval).
    pub n_rho_dot: usize,
    /// Population priors used to weight each node.
    pub populations: Vec<Population>,
    /// Floor on the range σ (AU), in case a cell is tiny.
    pub sigma_pos_rad_floor: f64,
    /// Floor on the range-rate σ (AU/day).
    pub sigma_rho_dot_floor: f64,
    /// Nodes whose population weight falls below this are dropped.
    pub weight_floor: f64,
}

impl Default for GridConfig {
    fn default() -> Self {
        Self {
            rho_min: 0.1,
            rho_max: 60.0,
            n_rho: 28,
            n_rho_dot: 11,
            populations: default_populations(),
            sigma_pos_rad_floor: 1e-3,
            sigma_rho_dot_floor: 1e-4,
            weight_floor: 1e-6,
        }
    }
}
