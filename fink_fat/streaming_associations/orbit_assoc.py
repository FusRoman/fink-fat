import numpy as np
import pandas as pd
import math

from pyspark import SparkFiles

from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as u

import sbpy.data as sso_py

from fink_science.tester import spark_unit_tests
from fink_fat.streaming_associations.kalman_assoc import roid_mask
from fink_fat.streaming_associations.fitroid_assoc import ang2pix

from fink_fat.others.utils import init_logging
from typing import Tuple


def df_to_orb(df_orb: pd.DataFrame) -> sso_py.Orbit:
    """
    Convert a dataframe into an orbit table

    Parameters
    ----------
    df_orb : pd.DataFrame
        dataframe containing orbital parameters

    Returns
    -------
    sso_py.Orbit
        orbit table

    Examples
    --------
    >>> df_orbit = pd.DataFrame({
    ...     "a": [2.587],
    ...     "e": [0.123],
    ...     "i": [4.526],
    ...     "long. node": [5.956],
    ...     "arg. peric": [9.547],
    ...     "mean anomaly": [12.587],
    ...     "ref_epoch": [2460158.8717433237],
    ...     "ssoCandId": ["FF20230802aaaaaaa"]
    ... })

    >>> df_to_orb(df_orbit)
    <QTable length=1>
       a       e       i    long. node ...   node   argper    M        epoch
       AU             deg              ...   deg     deg     deg
    float64 float64 float64  float64   ... float64 float64 float64     object
    ------- ------- ------- ---------- ... ------- ------- ------- -------------
      2.587   0.123   4.526      5.956 ...   5.956   9.547  12.587 2460158.87174
    """
    with pd.option_context("mode.chained_assignment", None):
        df_orb["targetname"] = df_orb["ssoCandId"]
        df_orb["orbtype"] = "KEP"

        df_orb["H"] = 14.45
        df_orb["G"] = 0.15

    orb_dict = df_orb.to_dict(orient="list")

    orb_dict["a"] = orb_dict["a"] * u.au
    orb_dict["i"] = orb_dict["i"] * u.deg
    orb_dict["node"] = orb_dict["long. node"] * u.deg
    orb_dict["argper"] = orb_dict["arg. peric"] * u.deg
    orb_dict["M"] = orb_dict["mean anomaly"] * u.deg
    orb_dict["epoch"] = Time(orb_dict["ref_epoch"], format="jd")
    orb_dict["H"] = orb_dict["H"] * u.mag

    ast_orb_db = sso_py.Orbit.from_dict(orb_dict)
    return ast_orb_db


def compute_ephem(
    orbits: pd.DataFrame, epochs: list, location: str = "I41"
) -> pd.DataFrame:
    """_summary_

    Parameters
    ----------
    orbits : pd.DataFrame
        dataframe containing orbit parameters
    epochs : float
        time to compute the ephemeries
    location : str, optional
        mpc code observatory, by default "I41" Zwicky Transient Facility

    Returns
    -------
    pd.DataFrame
        dataframe containing the ephemeries

    Examples
    --------
    >>> df_orbit = pd.DataFrame({
    ...     "a": [2.587],
    ...     "e": [0.123],
    ...     "i": [4.526],
    ...     "long. node": [5.956],
    ...     "arg. peric": [9.547],
    ...     "mean anomaly": [12.587],
    ...     "ref_epoch": [2460158.8717433237],
    ...     "ssoCandId": ["FF20230802aaaaaaa"]
    ... })

    >>> compute_ephem(df_orbit, [2460168.8746030545, 2460160.8748369324])
              targetname         RA        DEC  RA*cos(Dec)_rate  DEC_rate  ...        r     Delta          V   trueanom                         epoch
    0  FF20230802aaaaaaa  57.805688  22.400628          0.335055  0.108234  ...  2.28273  2.254726  19.184782  19.219255 2023-08-12 09:00:34.887907351
    1  FF20230802aaaaaaa  54.812999  21.515593          0.356581  0.117211  ...  2.27945  2.342217  19.249921  16.800475 2023-08-04 09:00:55.094959284
    <BLANKLINE>
    [2 rows x 12 columns]
    """
    orb_table = df_to_orb(orbits)

    return sso_py.Ephem.from_oo(
        orb_table, epochs=Time(epochs, format="jd"), location="I41", scope="basic"
    ).table.to_pandas()


def orbit_window(
    orbit_pdf: pd.DataFrame, coord_alerts: SkyCoord, jd: np.ndarray, orbit_tw: int
) -> pd.DataFrame:
    """
    Filter the orbits in orbit_pdf to keep only those close to the alert
    and those that are the most recently updated (within orbit_tw).

    Parameters
    ----------
    orbit_pdf : pd.DataFrame
        dataframe containing the orbit
    coord_alerts : SkyCoord
        coordinates of the alerts in the current spark batch,
        keep only the orbit close to the alerts of the current batch
    jd : np.ndarray
        exposure time of the alerts in the current batch
    orbit_tw : int
        time window to keep the orbit, old orbit are removed

    Returns
    -------
    pd.DataFrame
        orbit filtered

    Examples
    --------
    >>> from fink_fat.test.tester_utils import add_roid_datatest
    >>> add_roid_datatest(spark)
    >>> df_orbit = pd.DataFrame({
    ...     "a": [2.587, 3.258, 15.239, 1.0123],
    ...     "e": [0.123, 0.657, 0.956, 0.001],
    ...     "i": [4.526, 14.789, 87.32, 0.1],
    ...     "long. node": [5.956, 4.756, 14.231, 17.236],
    ...     "arg. peric": [9.547, 10.2369, 87.523, 46.135],
    ...     "mean anomaly": [12.587, 17.235, 123.456, 1.234],
    ...     "ref_epoch": [2460158.8717433237, 2460158.8717433237, 2460138.8717433237, 2460118.8717433237],
    ...     "last_ra": [10, 123, 35, 10],
    ...     "last_dec": [10, 57, 23, 10],
    ...     "ssoCandId": ["FF20230802aaaaaaa", "FF20230802aaaaaab", "FF20230802aaaaaac", "FF20230802aaaaaad"]
    ... })

    >>> coords_alert = SkyCoord(
    ...     [1, 11, 35, 10.5],
    ...     [0, 11, 24, 10.235],
    ...     unit = "deg"
    ... )

    >>> orbit_window(
    ...     df_orbit,
    ...     coords_alert,
    ...     [2460148.8717433237, 2460158.8717433237, 2460168.8717433237, 2460158.8717433237],
    ...     20
    ... )
            a      e       i  long. node  arg. peric  mean anomaly     ref_epoch  last_ra  last_dec          ssoCandId
    0   2.587  0.123   4.526       5.956       9.547        12.587  2.460159e+06       10        10  FF20230802aaaaaaa
    2  15.239  0.956  87.320      14.231      87.523       123.456  2.460139e+06       35        23  FF20230802aaaaaac
    """

    jd_min = np.min(jd)
    jd_max = np.max(jd)
    min_night_jd = Time(math.modf(jd_min)[1], format="jd").jd
    max_night_jd = Time(math.modf(jd_max)[1] + 0.99999999, format="jd").jd

    last_orbits = orbit_pdf[
        (orbit_pdf["ref_epoch"] <= max_night_jd)
        & (orbit_pdf["ref_epoch"] >= (min_night_jd - orbit_tw))
    ]

    NSIDE = 4
    orbit_pix = ang2pix(
        NSIDE, last_orbits["last_ra"].values, last_orbits["last_dec"].values
    )
    alert_pix = ang2pix(NSIDE, coord_alerts.ra.value, coord_alerts.dec.value)
    return last_orbits[np.isin(orbit_pix, alert_pix)]

def ephem_window(
    ephem_pdf: pd.DataFrame, coord_alerts: SkyCoord
) -> pd.DataFrame:
    """
    Return a subset of ephemerides close to the alerts of the current batch.
    The filter use healpix, each ephemerides contains within the same pixels than the alerts
    are output. 

    Parameters
    ----------
    ephem_pdf : pd.DataFrame
        dataframe containing the ephemerides, columns are ssoCandId, RA and DEC.
    coord_alerts : SkyCoord
        equatorial coordinates of the alerts contains in the current batch

    Returns
    -------
    pd.DataFrame
        filtered ephemerides

    Examples
    --------
    >>> rng = np.random.default_rng(42)
    >>> dec = rng.uniform(-10, 10, 100)
    >>> ra = rng.uniform(40, 60, 100)
    >>> coords_alerts = SkyCoord(ra, dec, unit="deg")

    >>> ephem = pd.DataFrame({
    ... "DEC": rng.uniform(-90, 90, 1000),
    ... "RA": rng.uniform(0, 360, 1000)
    ... })
    >>> ephem_window(ephem, coords_alerts)
            DEC         RA
    0 -8.298561  41.517518
    1 -8.222767  56.072762
    2 -3.331559  55.942963
    3 -4.898885  50.323533
    4  7.574226  47.131672
    """
    NSIDE = 32
    orbit_pix = ang2pix(
        NSIDE, ephem_pdf["RA"].values, ephem_pdf["DEC"].values
    )
    alert_pix = ang2pix(NSIDE, coord_alerts.ra.value, coord_alerts.dec.value)
    return ephem_pdf[np.isin(orbit_pix, alert_pix)].reset_index(drop=True)

def orbit_association(
    ra: np.ndarray,
    dec: np.ndarray,
    jd: np.ndarray,
    magpsf: np.ndarray,
    fid: np.ndarray,
    flags: np.ndarray,
    confirmed_sso: bool,
    estimator_id: pd.Series,
    ffdistnr: pd.Series,
    mag_criterion_same_fid: float,
    mag_criterion_diff_fid: float,
    orbit_error: float,
)-> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Associates the alerts from the current spark batch
    with the orbit estimated by fink_fat from the previous nights.

    Parameters
    ----------
    ra : np.ndarray
        right ascension of the alerts (degree)
    dec : np.ndarray
        declination of the alerts (degree)
    jd : np.ndarray
        exposure time of the alerts (julian date)
    magpsf : np.ndarray
        psf magnitude of the alerts
    fid : np.ndarray
        filter identifier of the alerts
    flags : np.ndarray
        roid flags
    confirmed_sso : bool
        if true, run the associations with the alerts flagged as 3 (confirmed sso)
        otherwise, run the association with the alerts flagged as 1 or 2 (candidates sso)
    estimator_id : pd.Series
        will contains the identifier of the orbits associated with the alerts
    ffdistnr : pd.Series
        will contains the distance between the ephemeries and the alerts
    mag_criterion_same_fid : float
        the criterion to filter the alerts with the same filter identifier for the magnitude
        as the last point used to compute the orbit
    mag_criterion_diff_fid : float
        the criterion to filter the alerts with the filter identifier for the magnitude
        different from the last point used to compute the orbit
    orbit_error: float
        error radius to associates the alerts with the orbits

    Returns
    -------
    flags: pd.Series
        contains the flags of the roid module
        see processor.py
    estimator_id:
        contains the orbit identifier, same as the ssoCandId column
    ffdistnr:
        contains the distance between the alerts and the ephemeries (degree)

    Examples
    --------
    >>> from fink_fat.test.tester_utils import add_roid_datatest
    >>> add_roid_datatest(spark)
    >>> flags, estimator_id, ffdistnr = orbit_association(
    ...     np.array([46.328490, 108.603010, 97.600172, 98.928007, 2.05]),
    ...     np.array([18.833964, -67.879693, 32.281571, 23.230676, 3.01]),
    ...     np.array([2460139.8717433237, 2460140.8717433237, 2460139.9917433237, 2460139.8717433237, 2460140.8717432237]),
    ...     np.array([16.2, 18.3, 17.2, 19.5, 17.4]),
    ...     np.array([1, 1, 2, 2, 2]),
    ...     np.array([2, 3, 1, 2, 3]),
    ...     False,
    ...     pd.Series([[], [], [], [], []]),
    ...     pd.Series([[], [], [], [], []]),
    ...     2, 2, 15.0
    ... )

    >>> flags
    array([2, 3, 5, 2, 3])

    >>> estimator_id
    0                     []
    1                     []
    2    [FF20230802aaaaaab]
    3                     []
    4                     []
    dtype: object

    >>> ffdistnr
    0                         []
    1                         []
    2    [4.788559911943089e-06]
    3                         []
    4                         []
    dtype: object
    """
    logger = init_logging()
    (
        _,
        _,
        coord_alerts,
        mag_mask,
        fid_mask,
        jd_mask,
        _,
        idx_keep_mask,
    ) = roid_mask(ra, dec, jd, magpsf, fid, flags, confirmed_sso)

    try:
        # get the latest computed ephemeride for the current observing night
        ephem_pdf = pd.read_parquet(SparkFiles.get("ephem.parquet"))
        # get latest detected orbit
        orbit_pdf = pd.read_parquet(SparkFiles.get("orbital.parquet"))
    except FileNotFoundError:
        logger.warning("files containing the orbits not found", exc_info=1)
        return flags, estimator_id, ffdistnr

    if len(ephem_pdf) != 0:
        ephem_to_keep = ephem_window(ephem_pdf, coord_alerts)
        if len(ephem_to_keep) == 0:
            return flags, estimator_id, ffdistnr
    else:
        return flags, estimator_id, ffdistnr

    # get equatorial coordinates of the ephemerides
    ephem_coord = SkyCoord(ephem_to_keep["RA"].values, ephem_to_keep["DEC"].values, unit=u.degree)

    # return the closest alerts of each ephemerides
    res_search = coord_alerts.match_to_catalog_sky(ephem_coord)
    sep = res_search[1]
    idx_ephem = res_search[0]

    # filter the associations to keep only those satisfying the orbit_error criteria
    f_distance = np.where(sep.arcsecond < orbit_error)[0]

    idx_ephem_assoc = idx_ephem[f_distance]
    close_orbit = ephem_to_keep.loc[idx_ephem_assoc].merge(
        orbit_pdf, on="ssoCandId"
    )[["ssoCandId", "last_mag", "last_fid", "last_jd", "ssoCandId"]]

    mag_assoc = mag_mask[f_distance]
    fid_assoc = fid_mask[f_distance]
    jd_assoc = jd_mask[f_distance]
    idx_assoc = idx_keep_mask[f_distance]

    # filter the associations to keep only those with a credible magnitude rate
    diff_mag = np.abs(mag_assoc - close_orbit["last_mag"])
    diff_jd = jd_assoc - close_orbit["last_jd"]
    rate = diff_mag / np.where(diff_jd >= 1, diff_jd, 1)
    mag_criterion = np.where(
        fid_assoc == close_orbit["last_fid"],
        mag_criterion_same_fid,
        mag_criterion_diff_fid,
    )
    idx_rate = np.where(rate < mag_criterion)[0]
    idx_mag = idx_assoc[idx_rate]

    flags[idx_mag] = 5
    estimator_id[idx_mag] = np.expand_dims(
        close_orbit.loc[idx_rate, "ssoCandId"], axis=1
    ).tolist()
    ffdistnr[idx_mag] = np.expand_dims(sep[f_distance][idx_rate].value, axis=1).tolist()

    # return the distance to the ephem and the associated orbit id
    return flags, estimator_id, ffdistnr


if __name__ == "__main__":
    """Execute the test suite"""

    globs = globals()

    # Run the test suite
    spark_unit_tests(globs)
