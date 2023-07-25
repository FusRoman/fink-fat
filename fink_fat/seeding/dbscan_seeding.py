import numpy as np
import pandas as pd

import astropy.units as u
from astropy.coordinates import SkyCoord
from sklearn.cluster import DBSCAN


def dist_3d(sep_lim):
    """
    Compute the separation limit on a unit sphere in cartesian representation

    Parameters
    ----------
    sep_lim: float
        separation limit in degree

    Return
    ------
    float
        the separation in radians

    Examples
    --------
    >>> dist_3d(1)
    0.01745307099674787
    >>> dist_3d(1/60) # one arcminute
    0.0002908882076401473
    >>> dist_3d(1/3600) # one arcsecond
    4.848136811090612e-06
    """
    return 2.0 * np.sin(np.radians(sep_lim) / 2.0)


def intra_night_seeding(
    night_observation, sep_criterion=(2.585714285714286 * u.arcmin).to("deg")
):
    """
    Find the cluster corresponding to the intra-night trajectory.
    The required columns in the input dataframe are ra, dec and jd.

    Parameters
    ----------
    df: DataFrame
        the input observations of the asteroids, all the observations should be in the same night.
    seplim: float
        the separation limit between the observations to create clusters in degree.

    Return
    ------
    df: DataFrame
        the input dataframe with a cluster column

    Examples
    --------
    >>> import astropy.units as u
    >>> df_test = pd.DataFrame({
    ... "ra": [1, 2, 30, 11, 12],
    ... "dec": [1, 2, 30, 11, 12],
    ... "jd": [0, 1, 2, 3, 4]
    ... })

    >>> intra_night_seeding(df_test, 2 * u.deg)
       ra  dec  jd  trajectory_id
    0   1    1   0              0
    1   2    2   1              0
    2  30   30   2             -1
    3  11   11   3              1
    4  12   12   4              1

    >>> df_test = pd.DataFrame({
    ... "ra": [1, 2, 30, 11, 12, 20, 21, 22],
    ... "dec": [1, 2, 30, 11, 12, 20, 21, 22],
    ... "jd": [0, 1, 2, 3, 4, 5, 5, 6]
    ... })

    >>> intra_night_seeding(df_test, 2 * u.deg)
       ra  dec  jd  trajectory_id
    0   1    1   0              0
    1   2    2   1              0
    2  30   30   2             -1
    3  11   11   3              1
    4  12   12   4              1
    5  20   20   5             -1
    6  21   21   5             -1
    7  22   22   6             -1
    """
    assert sep_criterion.unit == u.Unit("deg")

    coord_ast = SkyCoord(
        night_observation["ra"].values, night_observation["dec"].values, unit="deg"
    )
    cart_coord = coord_ast.cartesian.xyz.T.value

    clustering = DBSCAN(eps=dist_3d(sep_criterion), min_samples=2).fit(cart_coord)
    with pd.option_context("mode.chained_assignment", None):
        night_observation["trajectory_id"] = clustering.labels_

    # remove bad seeds containing observations in the same exposure time
    test_jd_0 = (
        night_observation.sort_values("jd")
        .groupby("trajectory_id")
        .agg(is_same_exp=("jd", lambda x: np.any(np.diff(x) == 0.0)))
        .reset_index()
    )
    test_jd_0 = test_jd_0[test_jd_0["trajectory_id"] != -1.0]
    jd_0_traj_id = test_jd_0[test_jd_0["is_same_exp"]]["trajectory_id"]

    with pd.option_context("mode.chained_assignment", None):
        night_observation.loc[
            night_observation["trajectory_id"].isin(jd_0_traj_id), "trajectory_id"
        ] = -1.0

    return night_observation


def seeding_purity(df: pd.DataFrame) -> float:
    """
    Return the purity of the clustering in per cent.
    purity = how able the clustering is to create pure clusters
        (pure cluster means clusters belonging to the same asteroids))

    Parameters
    ----------
    df: DataFrame
        the input dataframe with the observations
        should contains the following columns
        - ra, ssnamenr, cluster

    Returns
    -------
    float
        the efficiency of the clustering

    Examples
    --------
    >>> df_test = pd.DataFrame({
    ... "ra": [0, 0],
    ... "ssnamenr": [1, 1],
    ... "trajectory_id": [0, 0]
    ... })

    >>> seeding_purity(df_test)
    100.0

    >>> df_test = pd.DataFrame({
    ... "ra": [0, 0, 0, 0, 0],
    ... "ssnamenr": [1, 1, 2, 2, 2],
    ... "trajectory_id": [0, 0, 0, 1, 1]
    ... })

    >>> seeding_purity(df_test)
    0.0

    >>> df_test = pd.DataFrame({
    ... "ra": [0, 0, 0, 0, 0, 0, 0],
    ... "ssnamenr": [1, 1, 2, 2, 2, 3, 2],
    ... "trajectory_id": [0, 0, 0, 1, 1, -1, -1]
    ... })

    >>> seeding_purity(df_test)
    0.0

    >>> df_test = pd.DataFrame({
    ... "ra": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ... "ssnamenr": [1, 1, 2, 2, 2, 3, 2, 4, 4, 5, 5],
    ... "trajectory_id": [0, 0, 0, 1, 1, -1, -1, 2, 2, 3, 3]
    ... })

    >>> seeding_purity(df_test)
    50.0
    """
    # asteroids with at least two points in the night
    nb_p = df.groupby("ssnamenr").agg(nb_point=("ra", len)).reset_index()
    df = df.merge(nb_p, on="ssnamenr")

    recoverable = nb_p[nb_p["nb_point"] > 1]

    with pd.option_context("mode.chained_assignment", None):
        df["ssnamenr_bis"] = df["ssnamenr"].astype(str)
        df["cluster_bis"] = df["trajectory_id"]

    select_cols = [
        "ssnamenr",
        "trajectory_id",
        "ssnamenr_bis",
        "cluster_bis",
        "nb_point",
    ]

    r = (
        df[select_cols]
        .groupby("trajectory_id")
        .agg(
            ssnamenr_list=("ssnamenr_bis", list),
            cluster_list=("cluster_bis", list),
            cluster_size=("cluster_bis", len),
            nb_point=("nb_point", lambda x: list(x)[0]),
        )
        .reset_index()
    )

    def test_cluster(x):
        uniq_ast = np.unique(x["ssnamenr_list"])
        return (
            # is the cluster is complete (all the asteroids observation are in the cluster)
            len(x["cluster_list"]) == x["nb_point"]
            # only one asteroids names in the cluster
            and len(uniq_ast) == 1
        )

    res_cluster = r.apply(test_cluster, axis=1)
    if len(res_cluster) == 0:
        if len(recoverable) == 0:
            return 100
        else:
            return 0

    with pd.option_context("mode.chained_assignment", None):
        r["cluster_ok"] = res_cluster

    return (r["cluster_ok"].sum() / len(recoverable)) * 100


def seeding_completude(df: pd.DataFrame) -> float:
    """
    Return the completude of the clustering in per cent
    (how the clustering can find all the asteroids)

    Parameters
    ----------
    df : pd.DataFrame
        inputs dataframe, required columns are ssnamenr, ra, cluster

    Returns
    -------
    float
        the completude in percentage

    Examples
    --------
    >>> df_test = pd.DataFrame({
    ... "ra": [0, 0],
    ... "ssnamenr": [1, 1],
    ... "trajectory_id": [0, 0]
    ... })

    >>> seeding_completude(df_test)
    100.0

    >>> df_test = pd.DataFrame({
    ... "ra": [0, 0, 0, 0, 0],
    ... "ssnamenr": [1, 1, 2, 2, 2],
    ... "trajectory_id": [0, 0, 0, 1, 1]
    ... })

    >>> seeding_completude(df_test)
    50.0

    >>> df_test = pd.DataFrame({
    ... "ra": [0, 0, 0, 0, 0, 0, 0],
    ... "ssnamenr": [1, 1, 2, 2, 2, 3, 2],
    ... "trajectory_id": [0, 0, 0, 1, 1, -1, -1]
    ... })

    >>> seeding_completude(df_test)
    50.0
    """
    nb_p = df.groupby("ssnamenr").agg(nb_point=("ra", len)).reset_index()
    df = df.merge(nb_p, on="ssnamenr")

    recoverable = df[df["nb_point"] > 1]

    with pd.option_context("mode.chained_assignment", None):
        recoverable["ssnamenr_bis"] = df["ssnamenr"].astype(str)
        recoverable["cluster_bis"] = df["trajectory_id"]

    select_cols = [
        "ssnamenr",
        "trajectory_id",
        "ssnamenr_bis",
        "cluster_bis",
        "nb_point",
    ]

    r = (
        recoverable[select_cols]
        .groupby("ssnamenr")
        .agg(
            ssnamenr_list=("ssnamenr_bis", list),
            cluster_list=("cluster_bis", list),
            unique_cluster=("cluster_bis", lambda x: len(np.unique(list(x)))),
            first_cluster=("cluster_bis", lambda x: list(x)[0]),
            cluster_size=("cluster_bis", len),
            nb_point=("nb_point", lambda x: list(x)[0]),
        )
        .reset_index()
    )

    def test_cluster(x):
        uniq_cluster = np.unique(x["cluster_list"])
        return (
            # is the cluster is complete (all the asteroids observation are in the cluster)
            x["cluster_size"] == x["nb_point"]
            # only one asteroids names in the cluster
            and len(uniq_cluster) == 1
            # and the asteroid observations are not noise (in DBSCAN, -1 means noise so not in a cluster)
            and uniq_cluster[0] != -1
        )

    res_cluster = r.apply(test_cluster, axis=1)
    if len(res_cluster) == 0:
        if len(recoverable) == 0:
            return 100
        else:
            return 0

    with pd.option_context("mode.chained_assignment", None):
        r["cluster_ok"] = res_cluster

    return (r["cluster_ok"].sum() / len(np.unique(recoverable["ssnamenr"]))) * 100


if __name__ == "__main__":  # pragma: no cover
    import sys
    import doctest

    if "unittest.util" in __import__("sys").modules:
        # Show full diff in self.assertEqual.
        __import__("sys").modules["unittest.util"]._MAX_LENGTH = 999999999

    sys.exit(doctest.testmod()[0])
