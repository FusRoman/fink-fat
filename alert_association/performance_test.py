import pandas as pd
import time as t
import numpy as np
from intra_night_association import intra_night_association
from intra_night_association import new_trajectory_id_assignation
import matplotlib.pyplot as plt

# import astropy.units as u
import night_report
import continuous_integration as ci
import sys
import pyarrow as pa
import pyarrow.parquet as pq
from collections import Counter
from scipy import stats
import night_report as nr


if __name__ == "__main__":

    nb_trajectories_limit = True
    nb_traj_point_limit = 0
    nb_traj_limit = 5000000
    mpc_trajectories = [
        "232351",
        "73972",
        "75653",
        "53317",
        "1951",
        "80343",
        "1196",
        "23101",
        "1758",
    ]

    get_unknown_traj = True

    if len(sys.argv) != 2:
        print(
            "you need to add a main argument : 1 to launch performance test and 2 to show performance results"
        )
        exit(0)
    else:
        print("Load dataset")

        data_path = "../data/month=0"
        df_sso = ci.load_data(data_path, "Solar System MPC")

        if nb_trajectories_limit:
            mpc_traj = df_sso.groupby(["ssnamenr"]).agg(
                {"candid": len, "ssnamenr": list}
            )

            if get_unknown_traj:
                mpc_trajectories = np.unique(
                    mpc_traj[mpc_traj["candid"] > nb_traj_point_limit].explode(
                        ["ssnamenr"]
                    )["ssnamenr"]
                )[:nb_traj_limit]

            df_sso = df_sso[(df_sso["ssnamenr"].isin(mpc_trajectories))]

        print("total alert: {}".format(len(df_sso)))

        mpc_plot = (
            df_sso.groupby(["ssnamenr"]).agg({"ra": list, "dec": list}).reset_index()
        )
        print("number of known trajectory in this dataset: {}".format(len(mpc_plot)))

        if int(sys.argv[1]) == 1:
            print("Launch Performance test")

            all_night = np.unique(df_sso["nid"])

            last_nid = all_night[0]
            df_night1 = df_sso[df_sso["nid"] == last_nid]

            print("first intra night association to begin with some trajectories")
            t_before = t.time()
            left, right, first_intra_report = intra_night_association(
                df_night1, compute_metrics=True
            )

            last_trajectory_id = 0
            traj_df = new_trajectory_id_assignation(left, right, last_trajectory_id)
            traj_df = traj_df.reset_index(drop=True)

            save_report = True
            if save_report:
                first_intra_report["nb most recent traj"] = 0
                first_intra_report["nb old observations"] = 0
                first_intra_report["nb new observations"] = len(df_night1)
                nb_traj = len(np.unique(traj_df["trajectory_id"]))
                first_intra_report["nb trajectories"] = nb_traj
                first_intra_report["computation time of the night"] = float(
                    t.time() - t_before
                )
                first_intra_report["number of intra night tracklets"] = len(
                    np.unique(traj_df["trajectory_id"])
                )
                night_report.save_report(first_intra_report, df_night1["jd"].values[0])

            if len(traj_df) > 0:  # pragma: no cover
                old_observation = pd.concat(
                    [df_night1[~df_night1["candid"].isin(traj_df["candid"])]]
                )

                last_trajectory_id = np.max(traj_df["trajectory_id"].values) + 1
            else:
                old_observation = df_night1

            time_window_limit = 20
            verbose = True

            if verbose:
                print("Begin association process")

            it_limit = 5
            current_it = -10000
            most_recent_traj = pd.DataFrame(columns=["trajectory_id"])
            for i in range(1, len(all_night)):
                if current_it > it_limit:
                    break

                current_it += 1
                t_before = t.time()
                new_night = all_night[i]

                df_next_night = df_sso[df_sso["nid"] == new_night]

                current_night_id = df_next_night["nid"].values[0]

                if len(traj_df) > 0:
                    last_trajectory_id = np.max(traj_df["trajectory_id"].values) + 1

                (
                    (oldest_traj, most_recent_traj),
                    old_observation,
                ) = ci.time_window_management(
                    traj_df,
                    old_observation,
                    last_nid,
                    current_night_id,
                    time_window_limit,
                    remove_small_track=False,
                )
                if verbose:  # pragma: no cover
                    print()
                    print()
                    print("incoming night : {}".format(new_night))
                    if len(most_recent_traj) != 0:
                        nb_most_recent_traj = len(
                            np.unique(most_recent_traj["trajectory_id"])
                        )
                        print("nb most recent traj: {}".format(nb_most_recent_traj))
                    else:
                        nb_most_recent_traj = 0
                        print("nb most recent traj: {}".format(nb_most_recent_traj))
                    nb_old_obs = len(old_observation)
                    nb_new_obs = len(df_next_night)
                    print("nb old observation: {}".format(nb_old_obs))
                    print("nb new observation : {}".format(nb_new_obs))
                    print()

                # traj_df, old_observation, report = night_to_night_association(
                #     most_recent_traj,
                #     old_observation,
                #     df_next_night,
                #     last_trajectory_id,
                #     intra_night_sep_criterion=108.8 * u.arcsecond,
                #     sep_criterion=16.45 * u.arcminute,
                #     mag_criterion_same_fid=0.1,
                #     mag_criterion_diff_fid=0.34,
                #     angle_criterion=1.19,
                #     run_metrics=True,
                # )

                traj_df = pd.concat([traj_df, oldest_traj])
                last_nid = current_night_id
                nb_traj = len(np.unique(traj_df["trajectory_id"]))
                # if save_report:
                #     report["nb most recent traj"] = nb_most_recent_traj
                #     report["nb old observations"] = nb_old_obs
                #     report["nb new observations"] = nb_new_obs
                #     report["nb trajectories"] = nb_traj
                #     report["computation time of the night"] = float(t.time() - t_before)
                #     night_report.save_report(report, df_next_night["jd"].values[0])

                if verbose:  # pragma: no cover
                    print()
                    print("elapsed time: {}".format(t.time() - t_before))
                    print()
                    try:
                        print("nb_trajectories: {}".format(nb_traj))
                    except KeyError:
                        print("nb_trajectories: {}".format(0))
                    print("-----------------------------------------------")

            all_alert_not_associated = df_sso[~df_sso["candid"].isin(traj_df["candid"])]
            print()
            print()

            print("not associated alerts: {}".format(len(all_alert_not_associated)))

            print()

            if save_report:
                print("write trajectory dataframe on disk")

                traj_table = pa.Table.from_pandas(traj_df)
                pq.write_table(traj_table, "trajectory_output.parquet")

            print("end of the performance test")
        elif int(sys.argv[1]) == 2:
            print("show results")
            traj_df = pq.read_table("trajectory_output.parquet").to_pandas()

            print(
                "number of detected trajectories: {}".format(
                    len(np.unique(traj_df["trajectory_id"]))
                )
            )

            print("group by trajectory_id")
            gb_traj = (
                traj_df.groupby(["trajectory_id"])
                .agg(
                    {
                        "ra": list,
                        "dec": list,
                        "dcmag": list,
                        "fid": list,
                        "nid": list,
                        "ssnamenr": list,
                        "candid": len,
                    }
                )
                .reset_index()
            )

            gb_traj = gb_traj[gb_traj["candid"] > 3].sort_values(["candid"])
            print(
                "number of trajectories that have a number of point greather strictly than {}: {}".format(
                    3, len(gb_traj)
                )
            )

            def trajectory_metrics(rows):
                ssnamenr = rows["ssnamenr"]
                count = Counter(ssnamenr)

                nb_traj_change = len(count)
                most_commons = count.most_common()

                most_common = most_commons[0][1]
                sum_other = np.sum([el[1] for el in most_commons[1:]])

                traj_precision = sum_other / most_common

                return [nb_traj_change, traj_precision]

            all_counter = np.array(gb_traj.apply(trajectory_metrics, axis=1).tolist())

            fig, ax = plt.subplots(1, 1, figsize=(15, 15))
            ax.hist(all_counter[:, 0])
            ax.set_title("Number of ssnamenr tags by detected trajectories")

            traj_precision = all_counter[:, 1]
            prec_occur = Counter(traj_precision)
            print(
                "Ratio of perfectly detected trajectory: ({} / {}) * 100 = {} %".format(
                    prec_occur[0], len(gb_traj), (prec_occur[0] / len(gb_traj) * 100)
                )
            )

            print("min: {}".format(np.min(traj_precision)))
            print("max: {}".format(np.max(traj_precision)))
            print("median: {}".format(np.median(traj_precision)))
            print("mean: {}".format(np.mean(traj_precision)))

            print("skewness: {}".format(stats.mstats.skew(traj_precision)))

            fig, ax = plt.subplots(1, 1, figsize=(15, 15))
            ax.set_title(
                "Ratio between the most common ssnamenr tag and the other ssnamenr tags"
            )

            bins = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.75, 2, 2.5, 3, 4, 5, 6]
            ax.hist(traj_precision, 100)

            all_performance_values = nr.load_performance_stat()
            show_traj = False
            if show_traj:
                nr.plot_trajectories(traj_df, mpc_plot)
            nr.plot_performance_test(*all_performance_values)
