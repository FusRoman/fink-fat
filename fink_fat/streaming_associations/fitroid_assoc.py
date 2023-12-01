import numpy as np
import pandas as pd

from astropy.coordinates import SkyCoord, search_around_sky
import astropy.units as u

from pyspark import SparkFiles

from sklearn.neighbors import BallTree

from fink_fat.seeding.dbscan_seeding import dist_3d
from fink_fat.associations.associations import angle_three_point_vect
from fink_fat.others.utils import init_logging
from fink_fat.roid_fitting.roid_fit_prediction import fitroid_prediction
from fink_fat.associations.associations import night_to_night_separation_association

from fink_science.tester import spark_unit_tests
from typing import Tuple


def roid_mask(
    ra: np.ndarray,
    dec: np.ndarray,
    jd: np.ndarray,
    magpsf: np.ndarray,
    fid: np.ndarray,
    candid: np.ndarray,
    flags: np.ndarray,
    confirmed_sso: bool,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Return the inputs masked as solar sytem objects (confirmed or candidates depending of confirmed_sso)

    Parameters
    ----------
    ra : np.ndarray
        right ascension
    dec : np.ndarray
        declination
    jd : np.ndarray
        julian date
    magpsf : np.ndarray
        estimated magnitude of the psf
    fid : np.ndarray
        filter identifier
    candid : np.ndarray
        alert identifier
    flags : np.ndarray
        roid flags
    confirmed_sso : bool
        if true, used confirmed solar system object,
        used candidates otherwise

    Returns
    -------
    ra_mask : np.ndarray
        sso masked right ascension
    dec_mask : np.ndarray
        sso masked declination
    coord_alerts : np.ndarray
        sso masked coordinates
    mag_mask : np.ndarray
        sso masked magpsf
    fid_mask : np.ndarray
        sso masked filter id
    candid_mask : np.ndarray
        sso masked candid
    jd_mask : np.ndarray
        sso masked julian date
    jd_unique : np.ndarray
        sso masked unique julian date
    idx_keep_mask : np.ndarray
        idx in the non masked array of the masked data

    Examples
    --------
    >>> roid_mask(
    ... np.array([0, 1, 2, 3, 4, 5]),
    ... np.array([0, 1, 2, 3, 4, 5]),
    ... np.array([0, 1, 2, 3, 4, 5]),
    ... np.array([15, 16, 17, 18, 19, 20]),
    ... np.array([1, 1, 2, 1, 2, 2]),
    ... np.array([100, 101, 102, 103, 104, 105]),
    ... np.array([1, 3, 2, 2, 3, 0]),
    ... True
    ... )
    (array([1, 4]), array([1, 4]), <SkyCoord (ICRS): (ra, dec) in deg
        [( 1.,  1.), ( 4.,  4.)]>, array([16, 19]), array([1, 2]), array([101, 104]), array([1, 4]), array([1, 4]), array([1, 4]))

    >>> roid_mask(
    ... np.array([0, 1, 2, 3, 4, 5]),
    ... np.array([0, 1, 2, 3, 4, 5]),
    ... np.array([0, 1, 2, 3, 4, 5]),
    ... np.array([15, 16, 17, 18, 19, 20]),
    ... np.array([1, 1, 2, 1, 2, 2]),
    ... np.array([100, 101, 102, 103, 104, 105]),
    ... np.array([1, 3, 2, 2, 3, 0]),
    ... False
    ... )
    (array([0, 2, 3]), array([0, 2, 3]), <SkyCoord (ICRS): (ra, dec) in deg
        [( 0.,  0.), ( 2.,  2.), ( 3.,  3.)]>, array([15, 17, 18]), array([1, 2, 1]), array([100, 102, 103]), array([0, 2, 3]), array([0, 2, 3]), array([0, 2, 3]))
    """
    if confirmed_sso:
        keep_mask = flags == 3
    else:
        mask_first_time = flags == 1
        mask_possible_sso = flags == 2
        keep_mask = mask_first_time | mask_possible_sso

    ra_mask = ra[keep_mask]
    dec_mask = dec[keep_mask]
    coord_alerts = SkyCoord(
        ra_mask,
        dec_mask,
        unit="deg",
    )

    jd_mask = jd[keep_mask]
    mag_mask = magpsf[keep_mask]
    fid_mask = fid[keep_mask]
    candid_mask = candid[keep_mask]
    jd_unique = np.unique(jd_mask)
    idx_keep_mask = np.where(keep_mask)[0]
    return (
        ra_mask,
        dec_mask,
        coord_alerts,
        mag_mask,
        fid_mask,
        candid_mask,
        jd_mask,
        jd_unique,
        idx_keep_mask,
    )


def fitroid_window(pred_pdf: pd.DataFrame, coord_alerts: SkyCoord) -> pd.DataFrame:
    """
    Filter the prediction functions in the pred_pdf to keep only those close to the alerts.

    Parameters
    ----------
    pred_pdf : pd.DataFrame
        dataframe containing the kalman filters
    coord_alerts : SkyCoord
        coordinates of the alerts

    Returns
    -------
    pd.DataFrame
        the kalman dataframe with the kalman only close to the alerts

    Examples
    --------
    >>> pred_pdf = pd.DataFrame({
    ...     "ra_1": [0, 10, 30, 50],
    ...     "dec_1": [0, 10, 30, 50],
    ...     "trajectory_id": [0, 1, 2, 3]
    ... })

    >>> coord_alerts = SkyCoord([10, 50], [10, 50], unit="deg")

    >>> fitroid_window(pred_pdf, coord_alerts)
       ra_1  dec_1  trajectory_id
    1    10     10              1
    3    50     50              3
    """
    coord_kalman = SkyCoord(
        pred_pdf["ra_1"].values,
        pred_pdf["dec_1"].values,
        unit="deg",
    )
    (
        idx_kalman,
        _,
        _,
        _,
    ) = search_around_sky(
        coord_kalman,
        coord_alerts,
        5 * u.deg,
    )
    kalman_to_keep = pred_pdf[
        pred_pdf["trajectory_id"].isin(
            pred_pdf.iloc[idx_kalman]["trajectory_id"].unique()
        )
    ]
    return kalman_to_keep


def fitroid_association(
    ra: np.ndarray,
    dec: np.ndarray,
    jd: np.ndarray,
    magpsf: np.ndarray,
    fid: np.ndarray,
    candid: np.ndarray,
    flags: np.ndarray,
    confirmed_sso: bool,
    estimator_id: pd.Series,
    ffdistnr: pd.Series,
    error_radius: float,
    mag_criterion_same_fid: float,
    mag_criterion_diff_fid: float,
) -> Tuple[np.ndarray, pd.Series, pd.Series]:
    """
    Associates the alerts with the kalman filters

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
    candid: np.ndarray
        alert identifier
    flags : np.ndarray
        roid flags
    confirmed_sso : bool
        if true, run the associations with the alerts flagged as 3 (confirmed sso)
        otherwise, run the association with the alerts flagged as 1 or 2 (candidates sso)
    estimator_id : pd.Series
        will contains the identifier of the orbits associated with the alerts
    ffdistnr : pd.Series
        will contains the distance between the ephemeries and the alerts
    error_radius : pd.Series
        error radius used to associates the alerts with a trajectory
    mag_criterion_same_fid : float
        the criterion to filter the alerts with the same filter identifier for the magnitude
        as the last point used to compute the orbit
    mag_criterion_diff_fid : float
        the criterion to filter the alerts with the filter identifier for the magnitude
        different from the last point used to compute the orbit

    Returns
    -------
    flags: np.ndarray
        contains the flags of the roid module
        see processor.py
    estimator_id: pd.Series
        contains the orbit identifier, same as the ssoCandId column
    ffdistnr: pd.Series
        contains the distance between the alerts and the ephemeries (degree)

    Examples
    --------
    >>> from fink_fat.test.tester_utils import add_roid_datatest
    >>> add_roid_datatest(spark)
    >>> flags, estimator_id, ffdistnr = fitroid_association(
    ...     np.array([316.035113, 316.0302983, 347.8101124, 351.2340633, 350.9250203]),
    ...     np.array([-5.1239501, -5.1298901, 3.2216515, -3.9021139, -0.5572286]),
    ...     np.array(
    ...         [
    ...             2459459.7334491, 
    ...             2459459.7747685, 
    ...             2459459.8152199, 
    ...             2459459.8147454,
    ...             2459459.8147454
    ...         ]
    ...     ),
    ...     np.array([17.60202407836914, 18.120677947998047, 19.915021896362305, 19.479618072509766, 19.0051326751709]),
    ...     np.array([2, 1, 1, 1, 1]),
    ...     np.array([100, 101, 102, 103, 104]),
    ...     np.array([3, 3, 1, 2, 2]),
    ...     False,
    ...     pd.Series([[], [], [], [], []]),
    ...     pd.Series([[], [], [], [], []]),
    ...     15,
    ...     2,
    ...     2,
    ... )

    >>> flags
    array([3, 3, 4, 4, 4])

    >>> estimator_id
    0            []
    1            []
    2         [163]
    3    [183, 281]
    4         [238]
    dtype: object

    >>> ffdistnr
    0                                         []
    1                                         []
    2                       [0.6219832018982673]
    3    [0.5461049333814415, 6.365279160721899]
    4                       [0.6548163695675884]
    dtype: object


    >>> flags, estimator_id, ffdistnr = fitroid_association(
    ...     np.array([316.035113, 316.0302983, 347.8101124, 351.2340633, 350.9250203]),
    ...     np.array([-5.1239501, -5.1298901, 3.2216515, -3.9021139, -0.5572286]),
    ...     np.array(
    ...         [
    ...             2459459.7334491, 
    ...             2459459.7747685, 
    ...             2459459.8152199, 
    ...             2459459.8147454,
    ...             2459459.8147454
    ...         ]
    ...     ),
    ...     np.array([17.60202407836914, 18.120677947998047, 19.915021896362305, 19.479618072509766, 19.0051326751709]),
    ...     np.array([2, 1, 1, 1, 1]),
    ...     np.array([100, 101, 102, 103, 104]),
    ...     np.array([3, 3, 1, 2, 2]),
    ...     True,
    ...     pd.Series([[], [], [], [], []]),
    ...     pd.Series([[], [], [], [], []]),
    ...     15,
    ...     2,
    ...     2,
    ... )

    >>> flags
    array([4, 4, 1, 2, 2])

    >>> estimator_id
    0    [0]
    1    [0]
    2     []
    3     []
    4     []
    dtype: object

    >>> ffdistnr
    0    [0.8667306486778968]
    1       [0.6532971697207]
    2                      []
    3                      []
    4                      []
    dtype: object
    """
    logger = init_logging()
    (
        ra_mask,
        dec_mask,
        coord_masked_alerts,
        mag_mask,
        fid_mask,
        candid_mask,
        jd_mask,
        jd_unique,
        idx_keep_mask,
    ) = roid_mask(ra, dec, jd, magpsf, fid, candid, flags, confirmed_sso)

    try:
        # path where are stored the kalman filters
        # fit_pdf = pd.read_parquet(SparkFiles.get("fit_roid.parquet"))
        fit_pdf = pd.read_parquet(SparkFiles.get("fit_roid.parquet"))
    except FileNotFoundError:
        logger.warning("files containing the kalman filters not found", exc_info=1)
        return flags, estimator_id, ffdistnr

    # filter the kalman estimators to keep only those inside the current exposures.
    fit_to_keep = fitroid_window(fit_pdf, coord_masked_alerts)
    print(fit_to_keep)

    fit_pred = fitroid_prediction(fit_to_keep, jd_unique)

    next_assoc, pred_assoc, sep = night_to_night_separation_association(
        pd.DataFrame(
            {
                "ra": ra_mask,
                "dec": dec_mask,
                "candid": candid_mask,
                "magpsf": mag_mask,
                "jd": jd_mask,
                "fid": fid_mask,
            }
        ),
        fit_pred,
        error_radius * u.arcmin,
    )
    next_assoc["sep"] = sep.arcmin

    merge_assoc = (
        pd.concat(
            [
                next_assoc.reset_index(),
                pred_assoc[["trajectory_id"]]
                .reset_index()
                .rename({"index": "pred_index"}, axis=1),
            ],
            axis=1,
        )
        .drop_duplicates(["candid", "trajectory_id"])
        .merge(
            fit_to_keep[["trajectory_id", "mag_1", "fid_1", "jd_1"]], on="trajectory_id"
        )
    )
    print(fit_to_keep[["trajectory_id", "mag_1", "fid_1", "jd_1"]])
    print(pd.concat(
            [
                next_assoc.reset_index(),
                pred_assoc[["trajectory_id"]]
                .reset_index()
                .rename({"index": "pred_index"}, axis=1),
            ],
            axis=1,
        ).merge(
            fit_to_keep[["trajectory_id", "mag_1", "fid_1", "jd_1"]], on="trajectory_id"
        ))
    merge_assoc["trajectory_id"] = merge_assoc["trajectory_id"].astype(str)
    print(merge_assoc)

    diff_mag = np.abs(merge_assoc["magpsf"] - merge_assoc["mag_1"])
    diff_jd = merge_assoc["jd"] - merge_assoc["jd_1"]
    mag_rate = np.where(diff_jd > 1, diff_mag / diff_jd, diff_mag)
    mag_criterion = np.where(
        merge_assoc["fid"] == merge_assoc["fid_1"],
        mag_criterion_same_fid,
        mag_criterion_diff_fid,
    )
    assoc_filter = merge_assoc[mag_rate < mag_criterion]

    prep_res = (
        assoc_filter[["index", "trajectory_id", "sep"]]
        .groupby("index")
        .agg(list)
        .reset_index()
    )

    idx_to_update = idx_keep_mask[prep_res["index"].values]
    flags[idx_to_update] = 4
    estimator_id[idx_to_update] = prep_res["trajectory_id"]
    ffdistnr[idx_to_update] = prep_res["sep"]

    return flags, estimator_id, ffdistnr


if __name__ == "__main__":
    """Execute the test suite"""

    globs = globals()

    # Run the test suite
    spark_unit_tests(globs)
