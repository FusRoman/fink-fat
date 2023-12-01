import numpy as np
import pandas as pd
from typing import Tuple

from fink_fat.others.utils import repeat_chunk
from fink_fat.others.utils import LoggerNewLine
from fink_fat.roid_fitting.init_roid_fitting import init_polyast
import time


def merge_trajectory_cluster(
    tr_to_update: pd.DataFrame,
    fit_roid_df: pd.DataFrame,
    base_tr_id: int,
    new_alerts: pd.DataFrame,
    logger: LoggerNewLine,
    verbose: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Merge the trajectory with the intra night tracklets (cluster from dbscan)

    Parameters
    ----------
    tr_to_update : pd.DataFrame
        trajectory to associates with tracklets
    fit_roid_df : pd.DataFrame
        the prediction functions of the trajectories
    base_tr_id : int
        the start trajectory_id to assign to the new divergent trajectories
    new_alerts : pd.DataFrame
        the alerts from the new night
        columns explained:
            - trajectory_id: id return by the intra night associations (observations of the same tracklets)
            - roid: flag from the asteroids science module of fink, 4 means alerts associated with a fink-fat trajectory
            - estimator_id: id of the fink-fat trajectory associated with this alerts.
            The goal of this function is to merge the cluster (all alerts with the same trjaectory_id)
            to the trajectory with the same estimator_id.
    logger : LoggerNewLine
        the logger object for logs
    verbose : bool
        if true, print logs

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        new_traj_tracklets -> the trajectories extends with the clusters
        new_fit_roid_tracklets -> the prediction functions updated with the new added points from the clusters

    Examples
    --------
    see stream_association_test
    """

    cluster_df = new_alerts[
        (new_alerts["trajectory_id"] != -1) & (new_alerts["roid"] == 4)
    ]

    # new_tr_id = np.max(trajectory_df["trajectory_id"].unique()) + 1
    new_traj_tracklets = pd.DataFrame(columns=tr_to_update.columns)
    new_fit_roid_tracklets = pd.DataFrame(columns=fit_roid_df.columns)
    t_before = time.time()
    if len(cluster_df) != 0:
        if verbose:
            t_before = time.time()
            logger.info("- start merging trajectories with clusters")
        traj_id_for_tracklet_assoc = np.sort(cluster_df["estimator_id"].unique())
        traj_extended = [
            pd.concat(
                [
                    # get the trajectory
                    tr_to_update[tr_to_update["trajectory_id"] == tr_id],
                    # get the cluster
                    (
                        cluster_df[cluster_df["trajectory_id"] == cl_id].sort_values(
                            "jd"
                        )
                        # .drop_duplicates("objectId")
                    ),
                ]
            )
            for tr_id in traj_id_for_tracklet_assoc
            for cl_id in np.sort(
                cluster_df[cluster_df["estimator_id"] == tr_id][
                    "trajectory_id"
                ].unique()
            )
        ]

        # create the new trajectory id for each new trajectory merge with a cluster
        new_traj_id = np.arange(base_tr_id, base_tr_id + len(traj_extended))
        traj_id_tracklets = np.array(
            [
                new_traj_id[i]
                for i, tr in enumerate(traj_extended)
                for _ in range(len(tr))
            ]
        )

        new_traj_tracklets = pd.concat(traj_extended)
        new_traj_tracklets["trajectory_id"] = traj_id_tracklets
        new_traj_tracklets = new_traj_tracklets.sort_values(["trajectory_id", "jd"])
        new_fit_roid_tracklets = init_polyast(new_traj_tracklets)

    if verbose:
        logger.info(
            f"time to merge trajectories with clusters : {time.time() - t_before}"
        )
    return new_traj_tracklets, new_fit_roid_tracklets


def merge_trajectory_alerts(
    trajectory_df: pd.DataFrame,
    fit_roid_df: pd.DataFrame,
    next_traj_id: int,
    new_alerts: pd.DataFrame,
    logger: LoggerNewLine,
    verbose: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Merge the trajectory with the intra night tracklets (cluster from dbscan)

    Parameters
    ----------
    trajectory_df : pd.DataFrame
        dataframe containing the observations of the trajectories
    fit_roid_df : pd.DataFrame
        the prediction functions of the trajectories
    next_traj_id : int
        the start trajectory_id to assign to the new divergent trajectories
    new_alerts : pd.DataFrame
        the alerts from the new night
    logger : LoggerNewLine
        the logger object for logs
    verbose : bool
        if true, print logs

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        new_traj -> the trajectories with the new added points
        new_fit_roid_single -> the new estimations of the prediction functions

    Examples
    --------
    see stream_association_test
    """
    single_alerts = new_alerts[
        (new_alerts["trajectory_id"] == -1) & (new_alerts["estimator_id"] != -1)
    ]
    new_traj = pd.DataFrame(columns=single_alerts.columns)
    new_fit_roid_single = pd.DataFrame(columns=fit_roid_df.columns)

    if len(single_alerts) != 0:
        t_before = time.time()

        traj_counts_duplicates = (
            single_alerts["estimator_id"].value_counts().sort_index()
        )
        new_traj_id = np.arange(
            next_traj_id, next_traj_id + np.sum(traj_counts_duplicates)
        )
        single_alerts = single_alerts.sort_values("estimator_id")
        with pd.option_context("mode.chained_assignment", None):
            single_alerts["trajectory_id"] = new_traj_id

        traj_to_update = (
            trajectory_df[
                trajectory_df["trajectory_id"].isin(single_alerts["estimator_id"])
            ]
            .sort_values(["trajectory_id", "jd"])
            .reset_index(drop=True)
        )
        traj_size = traj_to_update["trajectory_id"].value_counts().sort_index()
        duplicate_id = repeat_chunk(
            traj_to_update.index.values, traj_size.values, traj_counts_duplicates.values
        )
        traj_duplicate = traj_to_update.loc[duplicate_id]

        nb_repeat = np.repeat(traj_size.values, traj_counts_duplicates.values)
        tr_id_repeat = np.repeat(single_alerts["trajectory_id"].values, nb_repeat)

        traj_duplicate["trajectory_id"] = tr_id_repeat

        new_traj = pd.concat([traj_duplicate, single_alerts])
        new_fit_roid_single = init_polyast(new_traj)
        if verbose:
            logger.info(
                f"merge trajectories and single alerts elapsed time: {time.time() - t_before:.4f} seconds"
            )

    return new_traj, new_fit_roid_single


def stream_association(
    trajectory_df: pd.DataFrame,
    fit_roid_df: pd.DataFrame,
    new_alerts: pd.DataFrame,
    logger: LoggerNewLine,
    verbose: bool,
    confirmed_sso: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Merge the trajectories with the new observations from the new night.
    The intra night associations (legacy or dbscan) must be run before.
    Keep the trajectories with no associations in the output.

    Parameters
    ----------
    trajectory_df : pd.DataFrame
        the observations of the trajectories
    fit_roid_df : pd.DataFrame
        the prediction functions of the trajectories
    new_alerts : pd.DataFrame
        the new observations of the last observing night
    logger : LoggerNewLine
        the logger used to print logs
    verbose : bool
        if true, print logs
    confirmed_sso : bool, optional
        if true, perform the associations with known sso, perform associations with candidates otherwise, by default False

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        traj_results -> the trajectories extends with the new points
        fit_roid_results -> the new estimation of the prediction functions with the new points

    Examples
    --------
    see stream_association_test
    """

    if confirmed_sso:
        roid_flag = [3, 4]
    else:
        roid_flag = [1, 2, 4]

    if verbose:
        logger.info("- start the association with the results from the roid stream")

    new_alerts = new_alerts[new_alerts["roid"].isin(roid_flag)]
    new_alerts = new_alerts.explode(["ffdistnr", "estimator_id"])
    new_alerts["estimator_id"] = new_alerts["estimator_id"].fillna(-1).astype(int)
    traj_id_to_update = np.sort(new_alerts["estimator_id"].unique())

    # get the trajectory to update
    mask_tr_update = trajectory_df["trajectory_id"].isin(traj_id_to_update)
    mask_fit_update = fit_roid_df["trajectory_id"].isin(traj_id_to_update)
    non_tr_update_df = trajectory_df[~mask_tr_update]
    non_fit_update = fit_roid_df[~mask_fit_update]

    # keep the trajectory with associations
    tr_to_update = trajectory_df[mask_tr_update]

    # ------ MERGE TRAJECTORY WITH CLUSTER ------
    new_tr_id = np.max(trajectory_df["trajectory_id"].unique()) + 1
    new_traj_tracklets, new_fit_roid_tracklets = merge_trajectory_cluster(
        tr_to_update, fit_roid_df, new_tr_id, new_alerts, logger, verbose
    )

    uniq_new_traj = new_fit_roid_tracklets["trajectory_id"].unique()
    if len(uniq_new_traj) > 0:
        next_traj_id = np.max(uniq_new_traj) + 1
    else:
        next_traj_id = new_tr_id

    # ------ MERGE TRAJECTORY WITH SINGLE ALERTS ------
    new_traj, new_fit_roid_single = merge_trajectory_alerts(
        trajectory_df, fit_roid_df, next_traj_id, new_alerts, logger, verbose
    )

    new_traj = pd.concat([new_traj_tracklets, new_traj])
    new_fit_roid = pd.concat([new_fit_roid_tracklets, new_fit_roid_single])

    # add a column to know which trajectories has been updated this night (Y: yes, N: no)
    with pd.option_context("mode.chained_assignment", None):
        new_traj["updated"] = "Y"
        non_tr_update_df["updated"] = "N"

    traj_results = pd.concat([non_tr_update_df, new_traj]).drop("estimator_id", axis=1)
    fit_roid_results = pd.concat([non_fit_update, new_fit_roid])
    if verbose:
        logger.newline()
    return traj_results, fit_roid_results
