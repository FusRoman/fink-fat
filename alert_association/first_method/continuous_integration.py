import pandas as pd
import time as t
import numpy as np
from intra_night_association import intra_night_association
from intra_night_association import new_trajectory_id_assignation
from intra_night_association import get_n_last_observations_from_trajectories
from night_to_night_association import night_to_night_association
import matplotlib.pyplot as plt
import astropy.units as u


if __name__ == "__main__":
    data_path = "../../data/month=0"
    all_df = []

    for i in range(3, 7):
        df_sso = pd.read_pickle(data_path + str(i))
        all_df.append(df_sso)

    df_sso = pd.concat(all_df).sort_values(["jd"]).drop_duplicates()

    df_sso = df_sso.drop_duplicates(["candid"])
    df_sso = df_sso[df_sso["fink_class"] == "Solar System MPC"]

    # "1951", "53317", 
    specific_mpc = df_sso[df_sso["ssnamenr"].isin(["1951", "53317","80343"])]

    print(specific_mpc)

    mpc_plot = specific_mpc.groupby(['ssnamenr']).agg({
        "ra": list,
        "dec": list
    }).reset_index()

    fig, ax = plt.subplots(1, 1, figsize=(20, 20))

    for _, rows in mpc_plot.iterrows():
        ra = rows['ra']
        dec = rows['dec']
        label = rows['ssnamenr']

        ax.scatter(ra, dec, label=label)

    plt.legend()
    plt.show()

    all_night = np.unique(specific_mpc["nid"])

    n1 = all_night[0]
    df_night1 = specific_mpc[specific_mpc["nid"] == n1]


    left, right, _ = intra_night_association(df_night1)

    last_trajectory_id = 0
    traj_df = new_trajectory_id_assignation(left, right, last_trajectory_id)
    traj_df = traj_df.reset_index(drop=True)

    if len(traj_df) > 0:
        old_observation = pd.concat(
            [df_night1[~df_night1["candid"].isin(traj_df["candid"])]]
        )

        last_trajectory_id = np.max(traj_df["trajectory_id"].values) + 1
    else:
        old_observation = df_night1

    time_window_limit = 12
    verbose = False

    for i in range(1, len(all_night)):
        t_before = t.time()
        new_night = all_night[i]

        df_next_night = specific_mpc[specific_mpc["nid"] == new_night]

        if verbose:
            print()
            print()
            print("incoming night : {}".format(new_night))
            print("nb new observation : {}".format(len(df_next_night)))
            print()

        current_night_id = df_next_night["nid"].values[0]

        most_recent_traj = pd.DataFrame()
        oldest_traj = pd.DataFrame()

        if len(traj_df) > 0:
            last_obs_of_all_traj = get_n_last_observations_from_trajectories(traj_df, 1)

            last_obs_of_all_traj["diff_nid"] = (
                current_night_id - last_obs_of_all_traj["nid"]
            )

            most_recent_last_obs = last_obs_of_all_traj[
                last_obs_of_all_traj["diff_nid"] <= time_window_limit
            ]

            if verbose:
                print(
                    "nb most recent traj to associate : {}".format(
                        len(most_recent_last_obs)
                    )
                )

            mask_traj = traj_df["trajectory_id"].isin(
                most_recent_last_obs["trajectory_id"]
            )

            most_recent_traj = traj_df[mask_traj]
            oldest_traj = traj_df[~mask_traj]

            last_trajectory_id = np.max(traj_df["trajectory_id"].values) + 1

        diff_nid_old_observation = current_night_id - old_observation["nid"]
        old_observation = old_observation[diff_nid_old_observation < time_window_limit]

        traj_df, old_observation, _ = night_to_night_association(
            most_recent_traj,
            old_observation,
            df_next_night,
            last_trajectory_id,
            intra_night_sep_criterion=2 * u.degree,
            sep_criterion=2 * u.degree,
            mag_criterion_same_fid=0.18,
            mag_criterion_diff_fid=0.7,
            angle_criterion=8.8,
        )

        traj_df = pd.concat([traj_df, oldest_traj])

        if verbose:
            print()
            print("elapsed time: {}".format(t.time() - t_before))
            print()
            print(
                "nb observation in the trajectory dataframe : {}\nnb old observations : {}".format(
                    len(traj_df), len(old_observation)
                )
            )
            print("-----------------------------------------------")       


    print(len(specific_mpc))

    print(len(traj_df))

    print(traj_df)

    print()
    print("show the alert not in traj_df")
    print(specific_mpc[~specific_mpc['candid'].isin(traj_df['candid'])])
    print()

    gb_traj = traj_df.groupby(['trajectory_id']).agg({
        "ra": list,
        "dec": list,
        "nid": list
    }).reset_index()


    print()
    print()
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(gb_traj)

    print()
    print()
    print(gb_traj['nid'][0])
    print()
    print()
    print(gb_traj['nid'][1])

    fig, ax = plt.subplots(1, 1, figsize=(20, 20))

    for _, rows in gb_traj.iterrows():
        ra = rows['ra']
        dec = rows['dec']
        label = rows['trajectory_id']

        ax.scatter(ra, dec, label=label)

    plt.legend()
    plt.show()
