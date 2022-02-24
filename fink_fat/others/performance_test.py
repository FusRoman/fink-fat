import json
import pandas as pd
import time as t
import numpy as np

import astropy.units as u
from fink_fat.associations.inter_night_associations import night_to_night_association
from fink_fat.others.utils import cast_obs_data
import psutil as pu

if __name__ == "__main__":

    from fink_fat.others.utils import load_data

    # constant to locate the ram file system
    ram_dir = "/media/virtuelram/"

    df_sso = load_data("Solar System MPC", 0)

    df_sso = df_sso.drop(
        [
            "fink_class",
            "magpsf",
            "sigmapsf",
            "magnr",
            "sigmagnr",
            "magzpsci",
            "isdiffpos",
        ],
        axis=1,
    )

    tr_orb_columns = [
        "provisional designation",
        "ref_epoch",
        "a",
        "e",
        "i",
        "long. node",
        "arg. peric",
        "mean anomaly",
        "rms_a",
        "rms_e",
        "rms_i",
        "rms_long. node",
        "rms_arg. peric",
        "rms_mean anomaly",
        "not_updated",
        "trajectory_id",
    ]

    required_columns = ["ra", "dec", "jd", "nid", "fid", "dcmag", "candid"]

    trajectory_df = pd.DataFrame(columns=tr_orb_columns)
    old_observation = pd.DataFrame(columns=["nid"])

    last_nid = np.min(df_sso["nid"])

    max_night_iter = 2000
    current_loop = 0

    all_time = []
    all_nb_traj = []

    current_test_parameters = {
        "traj_time_window": 15,
        "obs_time_window": 2,
        "traj_2_points_time_window": 7,
        "sep_criterion": 0.35,
        "mag_criterion_same_fid": 0.3,
        "mag_criterion_diff_fid": 0.7,
        "orbfit_limit": 1000,
        "angle_criterion": 1.5,
        "store_kd_tree": False,
        "do_track_and_traj_assoc": True,
        "do_traj_and_new_obs_assoc": True,
        "do_track_and_old_obs_assoc": True,
        "do_new_obs_and_old_obs_assoc": True,
    }

    for tr_nid in np.unique(df_sso["nid"]):

        memory = pu.virtual_memory()
        print("active memory : {}".format(memory.active / 1e9))
        print("available memory : {}".format(memory.available / 1e9))

        if current_loop > max_night_iter:
            print("BREAK NIGHT")
            print(tr_nid)
            print()
            break

        print("---New Night---")
        print(tr_nid)
        print()

        new_observation = df_sso[df_sso["nid"] == tr_nid]
        with pd.option_context("mode.chained_assignment", None):
            new_observation[tr_orb_columns] = -1.0
            new_observation["not_updated"] = np.ones(
                len(new_observation), dtype=np.bool_
            )

        next_nid = new_observation["nid"].values[0]

        nb_traj = len(np.unique(trajectory_df["trajectory_id"]))
        all_nb_traj.append(nb_traj)
        print("nb trajectories: {}".format(nb_traj))
        print("nb old obs: {}".format(len(old_observation)))
        print("nb new obs: {}".format(len(new_observation)))
        print()

        # # .to_dict(orient='list')
        # print(trajectory_df.to_dict(orient="list"))
        # print()
        # print()
        # print(old_observation.to_dict(orient="list"))
        # print()
        # print()
        # print(new_observation.to_dict(orient="list"))

        t_before = t.time()
        trajectory_df, old_observation, report = night_to_night_association(
            trajectory_df,
            old_observation,
            new_observation,
            last_nid,
            next_nid,
            traj_time_window=current_test_parameters["traj_time_window"],
            obs_time_window=current_test_parameters["obs_time_window"],
            traj_2_points_time_window=current_test_parameters[
                "traj_2_points_time_window"
            ],
            sep_criterion=current_test_parameters["sep_criterion"] * u.degree,
            mag_criterion_same_fid=current_test_parameters["mag_criterion_same_fid"],
            mag_criterion_diff_fid=current_test_parameters["mag_criterion_diff_fid"],
            orbfit_limit=current_test_parameters["orbfit_limit"],
            angle_criterion=current_test_parameters["angle_criterion"],
            ram_dir=ram_dir,
            store_kd_tree=current_test_parameters["store_kd_tree"],
            do_track_and_traj_assoc=current_test_parameters["do_track_and_traj_assoc"],
            do_traj_and_new_obs_assoc=current_test_parameters[
                "do_traj_and_new_obs_assoc"
            ],
            do_track_and_old_obs_assoc=current_test_parameters[
                "do_track_and_old_obs_assoc"
            ],
            do_new_obs_and_old_obs_assoc=current_test_parameters[
                "do_new_obs_and_old_obs_assoc"
            ],
            verbose=True,
        )

        # print()
        # print("------------------------")
        # print("ASSOCIATION DONE")
        # print()
        # print("trajectory")
        # print(trajectory_df.to_dict(orient="list"))
        # print()
        # print()
        # print(old_observation.to_dict(orient="list"))
        # print()
        # print()
        # print(report)
        # print()
        # print()

        elapsed_time = t.time() - t_before
        all_time.append(elapsed_time)

        if elapsed_time <= 60:
            print()
            print("associations elapsed time: {} sec".format(round(elapsed_time, 3)))
        else:
            time_min = int(elapsed_time / 60)
            time_sec = round(elapsed_time % 60, 3)
            print()
            print(
                "associations elapsed time: {} min: {} sec".format(time_min, time_sec)
            )

        print()
        print()

        print()
        orb_elem = trajectory_df[trajectory_df["a"] != -1.0]
        print(
            "number of trajectories with orbital elements: {}".format(
                len(np.unique(orb_elem["trajectory_id"]))
            )
        )

        print()
        print("---End Night---")
        print()
        print()
        print()
        print()

        last_nid = next_nid

        current_loop += 1

    record = False
    if len(trajectory_df) > 0 and record:
        test_name = "perf_test_4"
        trajectory_df = cast_obs_data(trajectory_df)
        trajectory_df["ssnamenr"] = trajectory_df["ssnamenr"].astype(str)
        trajectory_df["objectId"] = trajectory_df["objectId"].astype(str)

        trajectory_df = trajectory_df.drop(["provisional designation"], axis=1)
        trajectory_df.to_parquet(
            "fink_fat/others/perf_test/{}.parquet".format(test_name)
        )

        details = {"time": all_time, "trajectory_size": all_nb_traj}

        with open("fink_fat/others/perf_test/{}.json".format(test_name), "w") as file:
            file.write(json.dumps(details, indent=4))

        with open(
            "fink_fat/others/perf_test/params_{}.json".format(test_name), "w"
        ) as file:
            file.write(json.dumps(current_test_parameters, indent=4))
