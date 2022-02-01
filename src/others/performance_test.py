import json
import pandas as pd
import time as t
import numpy as np

import astropy.units as u
from sympy import det
from src.associations.inter_night_associations import night_to_night_association


if __name__ == "__main__":

    from src.others.utils import load_data

    # constant to locate the ram file system
    ram_dir = "/media/virtuelram/"

    df_sso = load_data("Solar System MPC", 0)

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

    max_night_iter = 10
    current_loop = 0

    all_time = []
    all_nb_traj = []

    for tr_nid in np.unique(df_sso["nid"]):

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

        print(
            "nb trajectories: {}".format(nb_traj)
        )
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
            traj_time_window=8,
            obs_time_window=3,
            sep_criterion=0.3 * u.degree,
            acceleration_criteria=0.24,
            mag_criterion_same_fid=0.3,
            mag_criterion_diff_fid=0.7,
            orbfit_limit=5,
            angle_criterion=1,
            ram_dir=ram_dir,
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

    print()
    print()
    print()
    print()

    print(
        trajectory_df[["ssnamenr", "trajectory_id", "a", "e", "i"]].sort_values(
            ["trajectory_id"]
        )
    )
    gb = trajectory_df.groupby(["trajectory_id", "ssnamenr"])

    print(len(np.unique(gb.count().reset_index()["ssnamenr"])))
    print()
    print(gb.agg({"ra": len, "a": list, "e": list, "i": list}))

    record = True
    if len(trajectory_df) > 0 and record:
        test_name = "perf_test_1"
        trajectory_df = trajectory_df.infer_objects()
        trajectory_df["ssnamenr"] = trajectory_df["ssnamenr"].astype(str)
        trajectory_df["fink_class"] = trajectory_df["fink_class"].astype(str)
        trajectory_df["objectId"] = trajectory_df["objectId"].astype(str)

        trajectory_df = trajectory_df.drop(["provisional designation"], axis=1)
        trajectory_df.to_parquet("src/others/perf_test/{}.parquet".format(test_name))

        details = {
            "time": all_time,
            "trajectory_size": all_nb_traj
        }

        with open('src/others/perf_test/{}.json'.format(test_name), 'w') as file:
            file.write(json.dumps(details))

