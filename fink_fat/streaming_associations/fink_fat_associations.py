import pandas as pd

from fink_fat.streaming_associations.fitroid_assoc import fitroid_association
from fink_fat.streaming_associations.orbit_assoc import orbit_association
import time

from fink_science.tester import spark_unit_tests


def fink_fat_association(
    ra: pd.Series,
    dec: pd.Series,
    magpsf: pd.Series,
    fid: pd.Series,
    jd: pd.Series,
    candid: pd.Series,
    flags: pd.Series,
    confirmed_sso: pd.Series,
    error_radius: pd.Series,
    mag_criterion_same_fid: pd.Series,
    mag_criterion_diff_fid: pd.Series,
    orbit_tw: pd.Series,
    orbit_error: pd.Series,
):
    """
    Associates the alerts with the orbit or the kalman filters estimates from the trajectories

    Parameters
    ----------
    ra : np.ndarray
        right ascension of the alerts (degree)
    dec : np.ndarray
        declination of the alerts (degree)
    magpsf : np.ndarray
        psf magnitude of the alerts
    fid : np.ndarray
        filter identifier of the alerts
    jd : np.ndarray
        exposure time of the alerts (julian date)
    candid : np.ndarray
        alert identifier
    flags : np.ndarray
        roid flags
    confirmed_sso : bool
        if true, run the associations with the alerts flagged as 3 (confirmed sso)
        otherwise, run the association with the alerts flagged as 1 or 2 (candidates sso)
    error_radius: pd.Series,
        error radius used to associates the alerts with a trajectory
    mag_criterion_same_fid : float
        the criterion to filter the alerts with the same filter identifier for the magnitude
        as the last point used to compute the orbit
    mag_criterion_diff_fid : float
        the criterion to filter the alerts with the filter identifier for the magnitude
        different from the last point used to compute the orbit
    orbit_tw : int
        time window used to filter the orbit
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
    >>> flags, estimator_id, ffdistnr = fink_fat_association(
    ...     pd.Series([
    ...         46.328490, 108.603010, 97.600172, 98.928007, 2.05, 46.55802234,
    ...         0.254, 48.147, 34.741, 0.198, 0.192,
    ...         316.035113, 316.0302983, 347.8101124, 351.2340633, 350.9250203
    ...     ]),
    ...     pd.Series([
    ...         18.833964, -67.879693, 32.281571, 23.230676, 3.01, 18.91096543,
    ...         1.036, 65.036, -0.214, 0.987, 0.943,
    ...         -5.1239501, -5.1298901, 3.2216515, -3.9021139, -0.5572286
    ...     ]),
    ...     pd.Series([
    ...         16.2, 18.3, 21.4, 19.5, 17.4, 16.4,
    ...         14.234, 18.3, 21.4, 14.429, 14.231,
    ...         17.60202407836914, 18.120677947998047, 19.915021896362305, 19.479618072509766, 19.0051326751709
    ...     ]),
    ...     pd.Series([
    ...         1, 1, 2, 2, 2, 1,
    ...         1, 1, 2, 2, 1,
    ...         2, 1, 1, 1, 1
    ...     ]),
    ...     pd.Series([
    ...         2460139.8717433237, 2460140.8717433237, 2460139.9917433237, 2460139.8717433237, 2460140.8717432237, 2460140.4217432237,
    ...         2460160.0004537117, 2460160.0007537117, 2460160.0009437117, 2460160.0000537117, 2460160.0009537117,
    ...         2459459.7334491, 2459459.7747685, 2459459.8152199, 2459459.8147454, 2459459.8147454
    ...     ]),
    ...     pd.Series([
    ...         100, 102, 103, 104, 105, 106,
    ...         107, 108, 109, 110, 111,
    ...         112, 113, 114, 115, 116
    ...     ]),
    ...     pd.Series([
    ...         2, 3, 1, 2, 3, 3,
    ...         3, 3, 1, 2, 2,
    ...         3, 3, 1, 2, 2
    ...     ]),
    ...     pd.Series([True]),
    ...     pd.Series([15.0]),
    ...     pd.Series([2]),
    ...     pd.Series([2]),
    ...     pd.Series([30]),
    ...     pd.Series([15.0])
    ... )

    >>> flags
    0     2
    1     3
    2     1
    3     2
    4     3
    5     5
    6     3
    7     3
    8     1
    9     2
    10    2
    11    4
    12    4
    13    1
    14    2
    15    2
    dtype: int64

    >>> estimator_id
    0                      []
    1                      []
    2                      []
    3                      []
    4                      []
    5     [FF20230802aaaaaaa]
    6                      []
    7                      []
    8                      []
    9                      []
    10                     []
    11                    [0]
    12                    [0]
    13                     []
    14                     []
    15                     []
    dtype: object

    >>> ffdistnr
    0                          []
    1                          []
    2                          []
    3                          []
    4                          []
    5     [5.325306762831802e-06]
    6                          []
    7                          []
    8                          []
    9                          []
    10                         []
    11       [0.8667306486778968]
    12          [0.6532971697207]
    13                         []
    14                         []
    15                         []
    dtype: object


    >>> flags, estimator_id, ffdistnr = fink_fat_association(
    ...     pd.Series([
    ...         46.328490, 108.603010, 97.600172, 98.928007, 2.05, 46.55802234,
    ...         0.254, 48.147, 34.741, 0.198, 0.192,
    ...         316.035113, 316.0302983, 347.8101124, 351.2340633, 350.9250203
    ...     ]),
    ...     pd.Series([
    ...         18.833964, -67.879693, 32.281571, 23.230676, 3.01, 18.91096543,
    ...         1.036, 65.036, -0.214, 0.987, 0.943,
    ...         -5.1239501, -5.1298901, 3.2216515, -3.9021139, -0.5572286
    ...     ]),
    ...     pd.Series([
    ...         16.2, 18.3, 21.4, 19.5, 17.4, 16.4,
    ...         14.234, 18.3, 21.4, 14.429, 14.231,
    ...         17.60202407836914, 18.120677947998047, 19.915021896362305, 19.479618072509766, 19.0051326751709
    ...     ]),
    ...     pd.Series([
    ...         1, 1, 2, 2, 2, 1,
    ...         1, 1, 2, 2, 1,
    ...         2, 1, 1, 1, 1
    ...     ]),
    ...     pd.Series([
    ...         2460139.8717433237, 2460140.8717433237, 2460139.9917433237, 2460139.8717433237, 2460140.8717432237, 2460140.4217432237,
    ...         2460160.0004537117, 2460160.0007537117, 2460160.0009437117, 2460160.0000537117, 2460160.0009537117,
    ...         2459459.7334491, 2459459.7747685, 2459459.8152199, 2459459.8147454, 2459459.8147454
    ...     ]),
    ...     pd.Series([
    ...         100, 102, 103, 104, 105, 106,
    ...         107, 108, 109, 110, 111,
    ...         112, 113, 114, 115, 116
    ...     ]),
    ...     pd.Series([
    ...         2, 3, 1, 2, 3, 3,
    ...         3, 3, 1, 2, 2,
    ...         3, 3, 1, 2, 2
    ...     ]),
    ...     pd.Series([False]),
    ...     pd.Series([15.0]),
    ...     pd.Series([2]),
    ...     pd.Series([2]),
    ...     pd.Series([30]),
    ...     pd.Series([15.0])
    ... )

    >>> flags
    0     5
    1     3
    2     1
    3     5
    4     3
    5     3
    6     3
    7     3
    8     1
    9     2
    10    2
    11    3
    12    3
    13    4
    14    4
    15    4
    dtype: int64

    >>> estimator_id
    0     [FF20230802aaaaaaa]
    1                      []
    2                      []
    3     [FF20230802aaaaaad]
    4                      []
    5                      []
    6                      []
    7                      []
    8                      []
    9                      []
    10                     []
    11                     []
    12                     []
    13                  [163]
    14             [183, 281]
    15                  [238]
    dtype: object

    >>> ffdistnr
    0                     [8.956302666882892e-06]
    1                                          []
    2                                          []
    3                    [1.9314410219648141e-07]
    4                                          []
    5                                          []
    6                                          []
    7                                          []
    8                                          []
    9                                          []
    10                                         []
    11                                         []
    12                                         []
    13                       [0.6219832018982673]
    14    [0.5461049333814415, 6.365279160721899]
    15                       [0.6548163695675884]
    dtype: object
    """
    # fink_fat associations
    ffdistnr = pd.Series([[] for _ in range(len(ra))])
    estimator_id = pd.Series([[] for _ in range(len(ra))])

    ra = ra.values
    dec = dec.values
    magpsf = magpsf.values
    fid = fid.values
    jd = jd.values
    candid = candid.values

    confirmed_sso = confirmed_sso.values[0]
    error_radius = error_radius.values[0]
    mag_criterion_same_fid = mag_criterion_same_fid.values[0]
    mag_criterion_diff_fid = mag_criterion_diff_fid.values[0]
    orbit_tw = orbit_tw.values[0]
    orbit_error = orbit_error.values[0]

    # associates the alerts with the orbit
    flags, estimator_id, ffdistnr = orbit_association(
        ra,
        dec,
        jd,
        magpsf,
        fid,
        flags,
        confirmed_sso,
        estimator_id,
        ffdistnr,
        mag_criterion_same_fid,
        mag_criterion_diff_fid,
        orbit_tw,
        orbit_error,
    )

    # associates the alerts with the kalman filters
    flags, estimator_id, ffdistnr = fitroid_association(
        ra,
        dec,
        jd,
        magpsf,
        fid,
        candid,
        flags,
        confirmed_sso,
        estimator_id,
        ffdistnr,
        error_radius,
        mag_criterion_same_fid,
        mag_criterion_diff_fid,
    )

    return flags, estimator_id, ffdistnr


if __name__ == "__main__":
    """Execute the test suite"""

    globs = globals()

    # Run the test suite
    spark_unit_tests(globs)
