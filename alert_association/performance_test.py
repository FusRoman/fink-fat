import pandas as pd
import time as t
import numpy as np
from intra_night_association import intra_night_association
from intra_night_association import new_trajectory_id_assignation
from night_to_night_association import night_to_night_association
import matplotlib.pyplot as plt
import astropy.units as u
import night_report
import continuous_integration as ci

if __name__ == "__main__":
    print("Lancement du test de performance")

    data_path = "../data/month=0"
    df_sso = ci.load_data(data_path, "Solar System MPC")

    print("total alert: {}".format(len(df_sso)))

    mpc_plot = df_sso.groupby(["ssnamenr"]).agg({"ra": list, "dec": list}).reset_index()

    all_night = np.unique(df_sso["nid"])

    last_nid = all_night[0]
    df_night1 = df_sso[df_sso["nid"] == last_nid]

    print("first intra night association to begin with some trajectories")
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
    verbose = True
    save_report = False
    show_results = False

    if verbose:
        print("Begin association process")

    for i in range(1, len(all_night)):
        t_before = t.time()
        new_night = all_night[i]

        df_next_night = df_sso[df_sso["nid"] == new_night]

        if verbose:  # pragma: no cover
            print()
            print()
            print("incoming night : {}".format(new_night))
            print("nb new observation : {}".format(len(df_next_night)))
            print()

        current_night_id = df_next_night["nid"].values[0]

        if len(traj_df) > 0:
            last_trajectory_id = np.max(traj_df["trajectory_id"].values) + 1

        (oldest_traj, most_recent_traj), old_observation = ci.time_window_management(
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

        traj_df = pd.concat([traj_df, oldest_traj])
        last_nid = current_night_id

        if save_report:
            report["computation time of the night"] = t.time() - t_before
            night_report.save_report(report, df_next_night["jd"].values[0])

        if verbose:  # pragma: no cover
            print()
            print("elapsed time: {}".format(t.time() - t_before))
            print()
            print(
                "nb_trajectories: {}".format(len(np.unique(traj_df["trajectory_id"])))
            )
            print("-----------------------------------------------")

    all_alert_not_associated = df_sso[~df_sso["candid"].isin(traj_df["candid"])]
    print()
    print()

    print("not associated alerts: {}".format(len(all_alert_not_associated)))
    print(
        "diff size between known trajectory MPC df and detected trajectory: {}".format(
            len(df_sso) - len(traj_df)
        )
    )

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
