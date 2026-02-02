/// Ingestion configuration when converting a Polars `LazyFrame` into an engine `AlertStore`.
#[derive(Clone, Debug)]
pub struct AlertIngestConfig {
    /// If `true`, interpret `ra` and `dec` columns as degrees and convert to radians.
    /// If `false`, assume they are already in radians.
    pub radec_in_degrees: bool,

    /// If `true`, interpret the `jd` column as Julian Date and convert to MJD via:
    /// `mjd = jd - 2400000.5`.
    ///
    /// Notes
    /// -----
    /// This does not change the time scale (TT/UTC). It only converts the *format*.
    pub jd_to_mjd: bool,

    /// Default 1-sigma astrometric uncertainty to use when `ra_err`/`dec_err` are absent.
    /// Value is in arcseconds.
    pub default_sigma_arcsec: f64,

    /// If `true`, store `magpsf` into `Alert::flux` and `sigmapsf` into `Alert::flux_err`
    /// as a *photometric proxy* (not a physical flux).
    pub store_mag_as_flux_proxy: bool,
}

impl Default for AlertIngestConfig {
    fn default() -> Self {
        Self {
            radec_in_degrees: true,
            jd_to_mjd: true,
            default_sigma_arcsec: 0.3,
            store_mag_as_flux_proxy: true,
        }
    }
}
