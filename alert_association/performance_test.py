import pandas as pd
import time as t
import numpy as np
from intra_night_association import intra_night_association
from intra_night_association import new_trajectory_id_assignation
from intra_night_association import get_n_last_observations_from_trajectories
from night_to_night_association import night_to_night_association
import matplotlib.pyplot as plt
import astropy.units as u
from pandas.testing import assert_frame_equal
import test_sample as ts
import sys
import night_report

if __name__ == "__main__":
    import doctest

    data_path = "data/month=0"
    all_df = []

    # load all data
    for i in range(3, 7):
        df_sso = pd.read_pickle(data_path + str(i))
        all_df.append(df_sso)

    df_sso = pd.concat(all_df).sort_values(["jd"]).drop_duplicates()

    df_sso = df_sso.drop_duplicates(["candid"])
    df_sso = df_sso[df_sso["fink_class"] == "Solar System MPC"]

    # "1951", "53317",

    # "1584" : exemple problématique avec une time window de 14 "53317", "1951", "80343", "1196", "23101",

    #  & (df_sso['nid'] >= 1610) & (df_sso['nid'] <= 1624)
    specific_mpc = df_sso[
        (df_sso["ssnamenr"].isin(["53317", "1951", "80343", "1196", "23101", "1758"]))
    ]

    mpc_plot = (
        specific_mpc.groupby(["ssnamenr"]).agg({"ra": list, "dec": list}).reset_index()
    )

    all_night = np.unique(specific_mpc["nid"])

    last_nid = all_night[0]
    df_night1 = specific_mpc[specific_mpc["nid"] == last_nid]

    left, right, _ = intra_night_association(df_night1)

    last_trajectory_id = 0
    traj_df = new_trajectory_id_assignation(left, right, last_trajectory_id)
    traj_df = traj_df.reset_index(drop=True)

    if len(traj_df) > 0:  # pragma: no cover
        old_observation = pd.concat(
            [df_night1[~df_night1["candid"].isin(traj_df["candid"])]]
        )

        last_trajectory_id = np.max(traj_df["trajectory_id"].values) + 1
    else:
        old_observation = df_night1

    time_window_limit = 14
    verbose = False
    save_report = False

    for i in range(1, len(all_night)):
        t_before = t.time()
        new_night = all_night[i]

        df_next_night = specific_mpc[specific_mpc["nid"] == new_night]

        if verbose:  # pragma: no cover
            print()
            print()
            print("incoming night : {}".format(new_night))
            print("nb new observation : {}".format(len(df_next_night)))
            print()

        current_night_id = df_next_night["nid"].values[0]

        if len(traj_df) > 0:
            last_trajectory_id = np.max(traj_df["trajectory_id"].values) + 1

        (oldest_traj, most_recent_traj), old_observation = time_window_management(
            traj_df, old_observation, last_nid, current_night_id, time_window_limit
        )

        traj_df, old_observation, report = night_to_night_association(
            most_recent_traj,
            old_observation,
            df_next_night,
            last_trajectory_id,
            intra_night_sep_criterion=10 * u.arcminute,
            sep_criterion=1 * u.degree,
            mag_criterion_same_fid=0.6,
            mag_criterion_diff_fid=0.85,
            angle_criterion=30,
            run_intra_night_metrics=True,
        )

        if save_report:
            night_report.save_report(report, df_next_night["jd"].values[0])

        traj_df = pd.concat([traj_df, oldest_traj])
        last_nid = current_night_id

        if verbose:  # pragma: no cover
            print()
            print("elapsed time: {}".format(t.time() - t_before))
            print()
            print("-----------------------------------------------")

    all_alert_not_associated = specific_mpc[
        ~specific_mpc["candid"].isin(traj_df["candid"])
    ]

    show_results = False

    if show_results:  # pragma: no cover

        gb_traj = (
            traj_df.groupby(["trajectory_id"])
            .agg(
                {
                    "ra": list,
                    "dec": list,
                    "dcmag": list,
                    "fid": list,
                    "nid": list,
                    "candid": lambda x: len(x),
                }
            )
            .reset_index()
        )

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(40, 40))

        for _, rows in mpc_plot.iterrows():
            ra = rows["ra"]
            dec = rows["dec"]
            label = rows["ssnamenr"]

            ax1.scatter(ra, dec, label=label)

        ax1.legend(title="mpc name")

        for _, rows in gb_traj.iterrows():
            ra = rows["ra"]
            dec = rows["dec"]
            label = rows["trajectory_id"]
            ax2.scatter(ra, dec, label=label)

        ax1.set_title("real trajectories")
        ax2.set_title("detected trajectories")

        ax2.legend(title="trajectory identifier")
        plt.show()

    try:
        assert_frame_equal(
            traj_df.reset_index(drop=True), ts.traj_df_expected, check_dtype=False
        )

        assert_frame_equal(
            all_alert_not_associated,
            pd.DataFrame(
                columns=[
                    "ra",
                    "dec",
                    "ssnamenr",
                    "jd",
                    "fid",
                    "nid",
                    "fink_class",
                    "objectId",
                    "candid",
                    "magpsf",
                    "sigmapsf",
                    "magnr",
                    "sigmagnr",
                    "magzpsci",
                    "isdiffpos",
                    "dcmag",
                    "dcmagerr",
                ]
            ),
            check_index_type=False,
            check_dtype=False,
        )

        sys.exit(doctest.testmod()[0])
    except AssertionError:  # pragma: no cover
        sys.exit(-1)
