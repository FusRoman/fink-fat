use std::fmt;

use crate::topocentric_kf::single_kalman::KFState;

impl<'state_lf> fmt::Debug for KFState<'state_lf> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("KFState")
            .field("epoch_mjd_tt", &self.epoch)
            .field("state_attributable", &format!(
                "α={:.6} rad ({}°), δ={:.6} rad ({}°), α̇={:.6e} rad/day, δ̇={:.6e} rad/day, ρ={:.6} AU, ρ̇={:.6e} AU/day",
                self.state[0],
                self.state[0].to_degrees(),
                self.state[1],
                self.state[1].to_degrees(),
                self.state[2],
                self.state[3],
                self.state[4],
                self.state[5]
            ))
            .field("observer_position_au", &format!(
                "({:.6}, {:.6}, {:.6})",
                self.r_obs[0], self.r_obs[1], self.r_obs[2]
            ))
            .field("observer_velocity_au_per_day", &format!(
                "({:.6e}, {:.6e}, {:.6e})",
                self.v_obs[0], self.v_obs[1], self.v_obs[2]
            ))
            .field("covariance_trace", &self.covariance.trace())
            .field("last_kalman_gain_norm", &self.last_gain_frobenius_norm())
            .finish()
    }
}

impl<'state_lf> fmt::Display for KFState<'state_lf> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "╔════════════════════════════════════════════════════════════╗"
        )?;
        writeln!(
            f,
            "║         Topocentric Kalman Filter State (Attributable)    ║"
        )?;
        writeln!(
            f,
            "╚════════════════════════════════════════════════════════════╝"
        )?;
        writeln!(f)?;

        // Epoch
        writeln!(
            f,
            "┌─ Epoch ─────────────────────────────────────────────────────┐"
        )?;
        writeln!(f, "│ MJD TT: {:.10}", self.epoch)?;
        writeln!(
            f,
            "└─────────────────────────────────────────────────────────────┘"
        )?;
        writeln!(f)?;

        // State vector
        writeln!(
            f,
            "┌─ Attributable State Vector ─────────────────────────────────┐"
        )?;
        writeln!(f, "│")?;
        writeln!(f, "│  Angular Position:")?;
        writeln!(
            f,
            "│    α (RA)        = {:12.8} rad  ({:11.6}°)",
            self.state[0],
            self.state[0].to_degrees()
        )?;
        writeln!(
            f,
            "│    δ (Dec)       = {:12.8} rad  ({:11.6}°)",
            self.state[1],
            self.state[1].to_degrees()
        )?;
        writeln!(f, "│")?;
        writeln!(f, "│  Angular Velocity:")?;
        writeln!(f, "│    α̇ (RA rate)   = {:12.6e} rad/day", self.state[2])?;
        writeln!(f, "│    δ̇ (Dec rate)  = {:12.6e} rad/day", self.state[3])?;
        writeln!(f, "│")?;
        writeln!(f, "│  Topocentric Range:")?;
        writeln!(f, "│    ρ (distance)  = {:12.8} AU", self.state[4])?;
        writeln!(f, "│    ρ̇ (rate)      = {:12.6e} AU/day", self.state[5])?;
        writeln!(f, "│")?;
        writeln!(
            f,
            "└─────────────────────────────────────────────────────────────┘"
        )?;
        writeln!(f)?;

        // Observer state
        writeln!(
            f,
            "┌─ Observer Heliocentric State ───────────────────────────────┐"
        )?;
        writeln!(f, "│")?;
        writeln!(f, "│  Position (AU, ecliptic J2000):")?;
        writeln!(
            f,
            "│    x = {:12.8}  y = {:12.8}  z = {:12.8}",
            self.r_obs[0], self.r_obs[1], self.r_obs[2]
        )?;
        writeln!(f, "│    |r_obs| = {:.8} AU", self.r_obs.norm())?;
        writeln!(f, "│")?;
        writeln!(f, "│  Velocity (AU/day, ecliptic J2000):")?;
        writeln!(
            f,
            "│    vx = {:12.6e}  vy = {:12.6e}  vz = {:12.6e}",
            self.v_obs[0], self.v_obs[1], self.v_obs[2]
        )?;
        writeln!(f, "│    |v_obs| = {:.6e} AU/day", self.v_obs.norm())?;
        writeln!(f, "│")?;
        writeln!(
            f,
            "└─────────────────────────────────────────────────────────────┘"
        )?;
        writeln!(f)?;

        // Covariance statistics
        writeln!(
            f,
            "┌─ Covariance Matrix Statistics ──────────────────────────────┐"
        )?;
        writeln!(f, "│")?;
        writeln!(f, "│  Diagonal Variances (1-σ uncertainties):")?;
        writeln!(
            f,
            "│    σ_α      = {:.6e} rad  ({:.4} arcsec)",
            self.covariance[(0, 0)].sqrt(),
            self.covariance[(0, 0)].sqrt().to_degrees() * 3600.0
        )?;
        writeln!(
            f,
            "│    σ_δ      = {:.6e} rad  ({:.4} arcsec)",
            self.covariance[(1, 1)].sqrt(),
            self.covariance[(1, 1)].sqrt().to_degrees() * 3600.0
        )?;
        writeln!(
            f,
            "│    σ_α̇      = {:.6e} rad/day",
            self.covariance[(2, 2)].sqrt()
        )?;
        writeln!(
            f,
            "│    σ_δ̇      = {:.6e} rad/day",
            self.covariance[(3, 3)].sqrt()
        )?;
        writeln!(
            f,
            "│    σ_ρ      = {:.6e} AU",
            self.covariance[(4, 4)].sqrt()
        )?;
        writeln!(
            f,
            "│    σ_ρ̇      = {:.6e} AU/day",
            self.covariance[(5, 5)].sqrt()
        )?;
        writeln!(f, "│")?;
        writeln!(f, "│  Correlations (selected cross-terms):")?;
        writeln!(
            f,
            "│    corr(α, α̇)  = {:.4}",
            self.covariance[(0, 2)]
                / (self.covariance[(0, 0)].sqrt() * self.covariance[(2, 2)].sqrt())
        )?;
        writeln!(
            f,
            "│    corr(δ, δ̇)  = {:.4}",
            self.covariance[(1, 3)]
                / (self.covariance[(1, 1)].sqrt() * self.covariance[(3, 3)].sqrt())
        )?;
        writeln!(
            f,
            "│    corr(ρ, ρ̇)  = {:.4}",
            self.covariance[(4, 5)]
                / (self.covariance[(4, 4)].sqrt() * self.covariance[(5, 5)].sqrt())
        )?;
        writeln!(
            f,
            "│    corr(α, ρ)  = {:.4}",
            self.covariance[(0, 4)]
                / (self.covariance[(0, 0)].sqrt() * self.covariance[(4, 4)].sqrt())
        )?;
        writeln!(
            f,
            "│    corr(δ, ρ)  = {:.4}",
            self.covariance[(1, 4)]
                / (self.covariance[(1, 1)].sqrt() * self.covariance[(4, 4)].sqrt())
        )?;
        writeln!(f, "│")?;
        writeln!(f, "│  Matrix Properties:")?;
        writeln!(f, "│    trace(P)  = {:.6e}", self.covariance.trace())?;
        writeln!(f, "│    det(P)    = {:.6e}", self.covariance.determinant())?;
        writeln!(f, "│")?;
        writeln!(
            f,
            "└─────────────────────────────────────────────────────────────┘"
        )?;
        writeln!(f)?;

        // Last update information
        writeln!(
            f,
            "┌─ Last Measurement Update ───────────────────────────────────┐"
        )?;
        match &self.kalman_gain {
            Some(k) => {
                writeln!(f, "│  Kalman gain norm (Frobenius) = {:.6e}", k.norm())?;
                writeln!(f, "│  Gain shape: 6 × 2 matrix")?;
                writeln!(
                    f,
                    "│  Max gain element: {:.6e}",
                    k.iter().map(|x| x.abs()).fold(f64::NEG_INFINITY, f64::max)
                )?;
            }
            None => {
                writeln!(f, "│  (No update applied yet — freshly initialized state)")?;
            }
        }
        writeln!(
            f,
            "└─────────────────────────────────────────────────────────────┘"
        )?;

        Ok(())
    }
}
