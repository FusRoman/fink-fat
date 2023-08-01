import numpy as np
import pandas as pd
import sys
import doctest

from copy import deepcopy

from fink_fat.kalman.init_kalman import init_kalman


def update_kalman(kalman_copy, new_alert):
    """
    Update the kalman filter contains in the

    Parameters
    ----------
    kalman_copy : pd.Series
        a row of the kalman dataframe
    new_alert : numpy array
        a numpy array containing data of one alert

    Returns
    -------
    pd.Series
        same row as input but updated with the new alert
    """
    Y = np.array(
        [
            [
                new_alert[2],
                new_alert[3],
            ]
        ]
    )

    dt = new_alert[4] - kalman_copy["jd_1"].values[0]
    A = np.array(
        [
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )

    kalman_copy["kalman"].values[0].update(
        Y,
        A,
    )
    with pd.option_context(
        "mode.chained_assignment",
        None,
    ):
        kalman_copy["ra_0"] = kalman_copy["ra_1"]
        kalman_copy["dec_0"] = kalman_copy["dec_1"]
        kalman_copy["jd_0"] = kalman_copy["jd_1"]

        kalman_copy["ra_1"] = new_alert[2]
        kalman_copy["dec_1"] = new_alert[3]
        kalman_copy["jd_1"] = new_alert[4]

        kalman_copy["mag_1"] = new_alert[9]
        kalman_copy["fid_1"] = new_alert[6]

        kalman_copy["dt"] = kalman_copy["jd_1"] - kalman_copy["jd_0"]
        kalman_copy["vel_ra"] = (
            kalman_copy["ra_1"] - kalman_copy["ra_0"]
        ) / kalman_copy["dt"]
        kalman_copy["vel_dec"] = (
            kalman_copy["dec_1"] - kalman_copy["dec_0"]
        ) / kalman_copy["dt"]

    return kalman_copy


def kalman_rowcopy(
    kalman_row,
):
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
    >>> r1 = pd.Series({
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
    >>> r2 = kalman_rowcopy(pd.DataFrame(r1).T)

    >>> id(r1) != id(r2)
    True
    >>> id(r1["ra_0"]) != id(r2["ra_0"])
    True
    >>> id(r1["kalman"]) != id(r2["kalman"])
    True
    """
    r = deepcopy(kalman_row)
    r["kalman"] = deepcopy(r["kalman"].values)
    return r


def update_trajectories(
    trajectory_df,
    kalman_df,
    new_alerts,
):
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
        dataframe containing the alerts from the current nigth to associates with the kalman filters

    Returns
    -------
    new_traj: pd.DataFrame
        the trajectories with the new associated points
    new_kalman: pd.DataFrame
        the updated kalman filters
    """
    # get the alerts associated with at least one kalman filter
    new_alerts["seeds_next_night"] = new_alerts["trajectory_id"]
    new_alerts_explode = new_alerts.explode(
        [
            "ffdistnr",
            "estimator_id",
        ]
    )
    new_alerts_explode = new_alerts_explode[~new_alerts_explode["ffdistnr"].isna()]

    # initialize new kalman filters based on the seeds no associated with a known kalman filter
    tracklets_no_assoc = new_alerts[
        ~new_alerts["trajectory_id"].isin(new_alerts_explode["estimator_id"])
    ]
    new_seeds = tracklets_no_assoc[tracklets_no_assoc["trajectory_id"] != -1]
    kalman_new_seeds = init_kalman(new_seeds)

    new_alerts = new_alerts.drop("trajectory_id", axis=1)
    new_alerts_explode = new_alerts_explode.drop("trajectory_id", axis=1)

    new_traj = []
    new_kalman = []
    new_traj_id = kalman_df["trajectory_id"].max() + 1

    unique_traj_id_to_update = new_alerts_explode["estimator_id"].unique()
    for trajectory_id in unique_traj_id_to_update:
        current_assoc = new_alerts_explode[
            new_alerts_explode["estimator_id"] == trajectory_id
        ]
        for seeds in current_assoc["seeds_next_night"].unique():
            if seeds != -1.0:
                # add an intra night tracklets to a trajectory
                current_trajectory = deepcopy(
                    trajectory_df[trajectory_df["trajectory_id"] == trajectory_id]
                )
                current_seeds = deepcopy(
                    new_alerts[new_alerts["seeds_next_night"] == seeds]
                )
                current_kalman_pdf = kalman_rowcopy(
                    kalman_df[kalman_df["trajectory_id"] == trajectory_id]
                )

                assert len(current_kalman_pdf) == 1

                for el in current_seeds.sort_values("jd").values:
                    current_kalman_pdf = update_kalman(
                        current_kalman_pdf,
                        el,
                    )

                with pd.option_context(
                    "mode.chained_assignment",
                    None,
                ):
                    current_trajectory["trajectory_id"] = new_traj_id
                    current_kalman_pdf["trajectory_id"] = new_traj_id
                    current_seeds["trajectory_id"] = new_traj_id
                new_traj.append(current_trajectory)
                new_traj.append(current_seeds[current_trajectory.columns])
                new_kalman.append(current_kalman_pdf)

                new_traj_id += 1

            else:
                current_seeds = deepcopy(
                    current_assoc[(current_assoc["seeds_next_night"] == seeds)]
                )
                current_seeds["trajectory_id"] = current_seeds["estimator_id"]
                cols_to_keep = list(current_seeds.columns[:-5]) + ["trajectory_id"]
                for el in current_seeds[cols_to_keep].values:
                    current_trajectory = deepcopy(
                        trajectory_df[trajectory_df["trajectory_id"] == trajectory_id]
                    )
                    current_kalman_pdf = kalman_rowcopy(
                        kalman_df[kalman_df["trajectory_id"] == trajectory_id]
                    )
                    assert len(current_kalman_pdf) == 1
                    current_kalman_pdf = update_kalman(
                        current_kalman_pdf,
                        el,
                    )

                    with pd.option_context(
                        "mode.chained_assignment",
                        None,
                    ):
                        el[-1] = new_traj_id
                        current_trajectory["trajectory_id"] = new_traj_id
                        current_kalman_pdf["trajectory_id"] = new_traj_id
                        current_seeds["trajectory_id"] = new_traj_id

                    new_traj.append(current_trajectory)
                    new_traj.append(
                        pd.DataFrame(
                            [el],
                            columns=cols_to_keep,
                        )
                    )
                    new_kalman.append(current_kalman_pdf)

                    new_traj_id += 1

    previous_traj = trajectory_df[
        ~trajectory_df["trajectory_id"].isin(unique_traj_id_to_update)
    ]
    previous_kalman = kalman_df[
        ~kalman_df["trajectory_id"].isin(unique_traj_id_to_update)
    ]

    new_traj = pd.concat([previous_traj, pd.concat(new_traj)]).sort_values(
        [
            "trajectory_id",
            "jd",
        ]
    )
    new_kalman = pd.concat([previous_kalman, pd.concat(new_kalman)])

    max_traj_id = new_kalman["trajectory_id"].max() + 1
    map_new_traj_id = {
        curr_traj_id: new_traj_id
        for curr_traj_id, new_traj_id in zip(
            kalman_new_seeds["trajectory_id"],
            np.arange(max_traj_id, max_traj_id + len(kalman_new_seeds), dtype=int),
        )
    }
    new_seeds["trajectory_id"] = new_seeds["trajectory_id"].map(map_new_traj_id)
    kalman_new_seeds["trajectory_id"] = kalman_new_seeds["trajectory_id"].map(
        map_new_traj_id
    )

    new_traj = new_traj.append(new_seeds[new_traj.columns]).reset_index(drop=True)
    new_kalman = new_kalman.append(kalman_new_seeds).reset_index(drop=True)

    return new_traj, new_kalman


if __name__ == "__main__":  # pragma: no cover
    if "unittest.util" in __import__("sys").modules:
        # Show full diff in self.assertEqual.
        __import__("sys").modules["unittest.util"]._MAX_LENGTH = 999999999

    sys.exit(doctest.testmod()[0])
