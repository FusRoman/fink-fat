import numpy as np
import pandas as pd
import sys
import doctest
from typing import Tuple

from copy import deepcopy

from fink_fat.others.utils import repeat_chunk
from fink_fat.others.utils import LoggerNewLine


def update_kalman(
    kalman_copy: pd.Series,
    ra_alert: float,
    dec_alert: float,
    jd_alert: float,
    mag_alert: float,
    fid_alert: int,
    tr_id: int,
) -> pd.Series:
    """
    Update the kalman filter in input inplace.

    Parameters
    ----------
    kalman_copy : pd.Series
        a row of the kalman dataframe
    ra_alert : float
        right ascencion
    dec_alert : float
        declination
    jd_alert : float
        julian date
    mag_alert : float
        magnitude
    fid_alert : int
        filter identifier
    tr_id : int
        new trajectory_id

    Returns
    -------
    pd.Series
        same row as input but updated with the new alert
    """
    dt = jd_alert - kalman_copy["jd_1"].values[0]

    if dt == 0:
        # if the dt is 0, the previous data point of the kalman are in the same exposure.
        # it happens only for the assocations between the tracklets and the kalman
        # and skip this point is not too bad for the rest of the kalman estimation.
        return kalman_copy
    with pd.option_context("mode.chained_assignment", None):
        kalman_copy["ra_0"] = kalman_copy["ra_1"]
        kalman_copy["dec_0"] = kalman_copy["dec_1"]
        kalman_copy["jd_0"] = kalman_copy["jd_1"]

        kalman_copy["ra_1"] = ra_alert
        kalman_copy["dec_1"] = dec_alert
        kalman_copy["jd_1"] = jd_alert

        kalman_copy["mag_1"] = mag_alert
        kalman_copy["fid_1"] = fid_alert

        kalman_copy["dt"] = kalman_copy["jd_1"] - kalman_copy["jd_0"]
        kalman_copy["vel_ra"] = (
            kalman_copy["ra_1"] - kalman_copy["ra_0"]
        ) / kalman_copy["dt"]
        kalman_copy["vel_dec"] = (
            kalman_copy["dec_1"] - kalman_copy["dec_0"]
        ) / kalman_copy["dt"]
        kalman_copy["trajectory_id"] = tr_id

    A = np.array(
        [
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )

    Y = np.array(
        [
            [ra_alert],
            [dec_alert],
            [kalman_copy["vel_ra"].values[0]],
            [kalman_copy["vel_dec"].values[0]],
        ]
    )

    kalman_copy["kalman"].values[0].update(
        Y,
        A,
    )

    return kalman_copy


def kalman_rowcopy(kalman_row: pd.Series, new_traj_id: int) -> pd.Series:
    """
    Make a copy of one rows of the kalman dataframe

    Parameters
    ----------
    kalman_row : pd.Series
        one row of a kalman dataframe

    Returns
    -------
    pd.Series
        a copy of the input row.

    Examples
    --------
    >>> from fink_fat.kalman.asteroid_kalman import KalfAst

    >>> traj_id = 0
    >>> r1 = pd.Series({
    ... "trajectory_id": traj_id,
    ... "ra_0": 0,
    ... "dec_0": 0,
    ... "ra_1": 1,
    ... "dec_1": 1,
    ... "jd_0": 0,
    ... "jd_1": 1,
    ... "mag_1": 0,
    ... "fid_1": 1,
    ... "dt": 1,
    ... "vel_ra": 0,
    ... "vel_dec": 0
    ... })

    >>> kalman = KalfAst(
    ... traj_id,
    ... r1["ra_1"],
    ... r1["dec_1"],
    ... r1["vel_ra"],
    ... r1["vel_dec"],
    ... [
    ...     [1, 0, 0.5, 0],
    ...     [0, 1, 0, 0.5],
    ...     [0, 0, 1, 0],
    ...     [0, 0, 0, 1],
    ... ],
    ... )
    >>> r1["kalman"] = kalman
    >>> r2 = kalman_rowcopy(pd.DataFrame(r1).T, 5)

    >>> id(r1) != id(r2)
    True
    >>> id(r1["ra_0"]) != id(r2["ra_0"])
    True
    >>> id(r1["kalman"]) != id(r2["kalman"])
    True
    >>> r2["kalman"].values[0].kf_id
    5
    """
    r = deepcopy(kalman_row)
    r["kalman"] = deepcopy(r["kalman"].values)
    r["kalman"].values[0].kf_id = int(new_traj_id)
    return r


def trajectory_extension(
    trajectory: pd.DataFrame,
    cluster: pd.DataFrame,
    cluster_id: int,
    kalman: pd.DataFrame,
    new_tr_id: int = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tmp_cluster = (
        cluster[cluster["trajectory_id"] == cluster_id]
        .drop("estimator_id", axis=1)
        .sort_values("jd")
        .drop_duplicates("objectId")
    )
    data_cluster = tmp_cluster[["ra", "dec", "jd", "magpsf", "fid"]].values

    # update the kalman with the new alerts from the clusters
    for ra, dec, jd, magpsf, fid in data_cluster:
        kalman = update_kalman(kalman, ra, dec, jd, magpsf, fid, new_tr_id)

    extended_traj = pd.concat([trajectory, tmp_cluster])

    if new_tr_id is not None:
        extended_traj["trajectory_id"] = new_tr_id
    return extended_traj, kalman


def tracklets_associations(
    trajectory_df: pd.DataFrame,
    kalman_df: pd.DataFrame,
    new_alerts: pd.DataFrame,
    logger: LoggerNewLine,
    verbose: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Associates the intra night trajectories with the kalman trajectories.

    Parameters
    ----------
    trajectory_df : pd.DataFrame
        trajectories estimated from kalman filters
    kalman_df : pd.DataFrame
        dataframe containing the kalman filters informations
    new_alerts : pd.DataFrame
        new associateds alerts from the new observing night.
    logger : LoggerNewLine
        logger class used to print the logs
    verbose : bool
        if true, print the logs

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        * the updated trajectories
        * the updated kalman filters

    """
    traj_id_to_update = np.sort(new_alerts["estimator_id"].unique())
    traj_id_to_update = traj_id_to_update[traj_id_to_update != -1]

    # get the trajectory to update
    mask_tr_update = trajectory_df["trajectory_id"].isin(traj_id_to_update)

    # keep the trajectory with associations
    tr_to_update = trajectory_df[mask_tr_update]

    # same for the kalman filters
    mask_kalman_update = kalman_df["trajectory_id"].isin(traj_id_to_update)
    kalman_to_update = kalman_df[mask_kalman_update]

    cluster_df = new_alerts[new_alerts["trajectory_id"] != -1]

    res_updated_traj = []
    res_updated_kalman = []

    new_tr_id = np.max(trajectory_df["trajectory_id"].unique()) + 1
    for tr_id in traj_id_to_update:
        current_tr = tr_to_update[tr_to_update["trajectory_id"] == tr_id]
        current_kalman = kalman_to_update[kalman_to_update["trajectory_id"] == tr_id]
        current_cluster = cluster_df[cluster_df["estimator_id"] == tr_id]
        current_cluster_id = np.sort(current_cluster["trajectory_id"].unique())
        for cl_id in current_cluster_id:
            next_extended_traj, next_updated_kalman = trajectory_extension(
                current_tr,
                cluster_df,
                cl_id,
                kalman_rowcopy(current_kalman, new_tr_id),
                new_tr_id,
            )

            res_updated_traj.append(next_extended_traj)
            res_updated_kalman.append(next_updated_kalman)
            new_tr_id += 1

    if len(res_updated_traj) == 0:
        if verbose:
            logger.info(
                "no associations between the intra night tracklets and the trajectories"
            )
        return (
            pd.DataFrame(columns=trajectory_df.columns),
            pd.DataFrame(columns=kalman_df.columns),
            new_tr_id,
        )

    if verbose:
        logger.info(
            f"number of associations between the intra night tracklets and the trajectories: {len(res_updated_kalman)}"
        )

    # merge the extended trajectories
    all_extended_traj = pd.concat(res_updated_traj)
    all_new_kalman = pd.concat(res_updated_kalman)

    return all_extended_traj, all_new_kalman, new_tr_id


def single_alerts_associations(
    trajectory_df: pd.DataFrame,
    kalman_df: pd.DataFrame,
    new_alerts: pd.DataFrame,
    max_tr_id: int,
    logger: LoggerNewLine,
    verbose: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Associates the single alerts with the kalman trajectories

    Parameters
    ----------
    trajectory_df : pd.DataFrame
        trajectories estimated from kalman filters
    kalman_df : pd.DataFrame
        dataframe containing the kalman filters informations
    new_alerts : pd.DataFrame
        new associateds alerts from the new observing night.
    max_tr_id : int
        maximum trajectory id to assign to the new kalman trajectories
    logger : LoggerNewLine
        logger class used to print the logs
    verbose : bool
        if true, print the logs

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        * new trajectories with the new added alerts
        * new kalman filters
    """
    cluster_df = new_alerts[
        (new_alerts["trajectory_id"] == -1) & (new_alerts["estimator_id"] != -1)
    ]
    traj_counts_duplicates = cluster_df["estimator_id"].value_counts().sort_index()
    new_traj_id = np.arange(max_tr_id, max_tr_id + np.sum(traj_counts_duplicates))
    with pd.option_context("mode.chained_assignment", None):
        cluster_df["trajectory_id"] = new_traj_id
    cluster_df = cluster_df.sort_values("estimator_id")

    if len(cluster_df) == 0:
        if verbose:
            logger.info(
                "no associations between the single alerts and the trajectories"
            )
        return pd.DataFrame(columns=trajectory_df.columns), pd.DataFrame(
            columns=kalman_df.columns
        )

    new_kalman = pd.concat(
        [
            update_kalman(
                kalman_rowcopy(kalman_df[kalman_df["trajectory_id"] == est_id], tr_id),
                ra,
                dec,
                jd,
                magpsf,
                fid,
                tr_id,
            )
            for ra, dec, jd, magpsf, fid, tr_id, est_id in cluster_df[
                ["ra", "dec", "jd", "magpsf", "fid", "trajectory_id", "estimator_id"]
            ].values
        ]
    )
    if verbose:
        logger.info(
            f"number of kalman trajectories to updated with single alert: {len(new_kalman)}"
        )
    traj_to_update = (
        trajectory_df[trajectory_df["trajectory_id"].isin(cluster_df["estimator_id"])]
        .sort_values(["trajectory_id", "jd"])
        .reset_index(drop=True)
    )
    traj_size = traj_to_update["trajectory_id"].value_counts().sort_index()
    duplicate_id = repeat_chunk(
        traj_to_update.index.values, traj_size.values, traj_counts_duplicates.values
    )

    traj_duplicate = traj_to_update.loc[duplicate_id]
    nb_repeat = np.repeat(traj_size.values, traj_counts_duplicates.values)
    tr_id_repeat = np.repeat(cluster_df["trajectory_id"].values, nb_repeat)

    traj_duplicate["trajectory_id"] = tr_id_repeat
    new_traj = pd.concat([traj_duplicate, cluster_df.drop("estimator_id", axis=1)])
    return new_traj, new_kalman


def kalman_association(
    trajectory_df: pd.DataFrame,
    kalman_df: pd.DataFrame,
    new_alerts: pd.DataFrame,
    logger: LoggerNewLine,
    verbose: bool,
    confirmed_sso: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Update the kalman filters and the trajectories with the new associations found during the night.
    This function initialize also the new kalman filters based on the seeds found in the night.

    Parameters
    ----------
    trajectory_df : pd.DataFrame
        dataframe containing the observations of each trajectories
    kalman_df : pd.DataFrame
        dataframe containing the kalman filters
    new_alerts : pd.DataFrame
        dataframe containing the alerts from the current nigth to associates with the kalman filters,
        the dataframe must contains the trajectory_id corresponding to the seeds find by the seeding.
    logger : LoggerNewLine
        logger class used to print the logs
    verbose : bool
        if true, print the logs
    confirmed_sso : boolean
        if true, used the confirmed sso (for test purpose)

    Returns
    -------
    new_traj: pd.DataFrame
        the trajectories with the new associated points
    new_kalman: pd.DataFrame
        the updated kalman filters

    Examples
    --------
    #### see fink_fat/test/kalman_test/update_kalman_test
    """
    if confirmed_sso:
        roid_flag = [3, 4]
    else:
        roid_flag = [1, 2, 4]

    if verbose:
        logger.info("- start the association with the kalman filters")
    new_alerts = new_alerts[new_alerts["roid"].isin(roid_flag)].explode(
        ["estimator_id", "ffdistnr"]
    )
    new_alerts = new_alerts.explode(["ffdistnr", "estimator_id"])
    new_alerts["estimator_id"] = new_alerts["estimator_id"].fillna(-1).astype(int)
    traj_id_to_update = np.sort(new_alerts["estimator_id"].unique())

    # get the trajectory to update
    mask_tr_update = trajectory_df["trajectory_id"].isin(traj_id_to_update)
    mask_kalman_update = kalman_df["trajectory_id"].isin(traj_id_to_update)
    non_tr_update_df = trajectory_df[~mask_tr_update]
    non_kalman_update = kalman_df[~mask_kalman_update]

    res_tr, res_kalman, max_tr_id = tracklets_associations(
        trajectory_df, kalman_df, new_alerts, logger, verbose
    )
    new_traj, new_kalman = single_alerts_associations(
        trajectory_df, kalman_df, new_alerts, max_tr_id, logger, verbose
    )

    new_traj = pd.concat([res_tr, new_traj])
    new_kalman = pd.concat([res_kalman, new_kalman])

    # add a column to know which trajectories has been updated this night (Y: yes, N: no)
    with pd.option_context("mode.chained_assignment", None):
        new_traj["updated"] = "Y"
        non_tr_update_df["updated"] = "N"

    traj_results = pd.concat([non_tr_update_df, new_traj])
    kalman_results = pd.concat([non_kalman_update, new_kalman])
    if verbose:
        logger.newline()
    return traj_results, kalman_results


if __name__ == "__main__":  # pragma: no cover
    if "unittest.util" in __import__("sys").modules:
        # Show full diff in self.assertEqual.
        __import__("sys").modules["unittest.util"]._MAX_LENGTH = 999999999

    sys.exit(doctest.testmod()[0])
