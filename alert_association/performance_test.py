import pandas as pd
import time as t
import numpy as np

import astropy.units as u
from alert_association.inter_night_associations import night_to_night_association


if __name__ == "__main__":

    import doctest

    from alert_association.continuous_integration import load_data

    # constant to locate the ram file system
    ram_dir = "/media/virtuelram/"

    df_sso = load_data("Solar System MPC", 0)

    traj_count = df_sso.groupby(["ssnamenr"]).count().reset_index()

    traj_name = traj_count[traj_count["ra"].isin([20])]["ssnamenr"][:5]
    df_sso = df_sso[df_sso["ssnamenr"].isin(traj_name)].sort_values(["ssnamenr"])

    print(df_sso.groupby(["ssnamenr"]).count())
    print()
    print()
    print()
    
    # exit()
    # for i in range(len(all_night) - 1):
    #     print(all_night[i], all_night[i + 1])
    #     print()
    #     n1 = df_sso[df_sso["nid"] == all_night[i]]
    #     n2 = df_sso[df_sso["nid"] == all_night[i + 1]]

    #     n1_gb = n1.groupby(["ssnamenr"]).count().reset_index()
    #     n2_gb = n2.groupby(["ssnamenr"]).count().reset_index()

    #     old_obs = n1_gb[n1_gb["ra"] == 1]
    #     track = n2_gb[n2_gb["ra"] == 1]

    #     print(old_obs.merge(track, on="ssnamenr"))
    #     print()
    #     print()

    
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

    for tr_nid in np.unique(df_sso["nid"]):
        print("---New Night---")
        print(tr_nid)
        print()

        # if tr_nid == 1540:
        #     break

        new_observation = df_sso[df_sso["nid"] == tr_nid]
        with pd.option_context("mode.chained_assignment", None):
            new_observation[tr_orb_columns] = -1.0
            new_observation["not_updated"] = np.ones(
                len(new_observation), dtype=np.bool_
            )

        next_nid = new_observation["nid"].values[0]

        # print(
        #     "nb trajectories: {}".format(len(np.unique(trajectory_df["trajectory_id"])))
        # )
        # print("nb old obs: {}".format(len(old_observation)))
        # print("nb new obs: {}".format(len(new_observation)))
        # print()

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
            traj_time_window=200,
            obs_time_window=200,
            intra_night_sep_criterion=500 * u.arcsecond,
            sep_criterion=0.5 * u.degree,
            acceleration_criteria=1000,
            mag_criterion_same_fid=5,
            mag_criterion_diff_fid=5,
            orbfit_limit=5,
            angle_criterion=200,
            ram_dir=ram_dir
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
        print(report)
        print()
        print()

        elapsed_time = t.time() - t_before
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

        # print()
        # orb_elem = trajectory_df[trajectory_df["a"] != -1.0]
        # print(
        #     "number of trajectories with orbital elements: {}".format(
        #         len(np.unique(orb_elem["trajectory_id"]))
        #     )
        # )

        # print()
        # print()
        # if len(trajectory_df) > 0:
        #     print("/////////TEST JD DUPLICATES///////////")
        #     test = (
        #         trajectory_df.groupby(["trajectory_id"])
        #         .agg(jd=("jd", list))
        #         .reset_index()
        #     )
        #     diff_jd = test.apply(lambda x: np.any(np.diff(x["jd"]) == 0), axis=1)
        #     keep_traj = test[diff_jd]["trajectory_id"]
        #     tttt = trajectory_df[trajectory_df["trajectory_id"].isin(keep_traj)]
        #     print(
        #         tttt[
        #             ["ra", "dec", "objectId", "jd", "nid", "ssnamenr", "trajectory_id"]
        #         ]
        #     )
        #     print()
        #     print("/////////FIN TEST JD DUPLICATES///////////")

        print()
        print("---End Night---")
        print()
        print()
        print()
        print()

        last_nid = next_nid


    print()
    print()
    print()
    print()

    print(trajectory_df[["ssnamenr", "trajectory_id", "a", "e", "i"]].sort_values(["trajectory_id"]))
    gb = trajectory_df.groupby(["trajectory_id", "ssnamenr"])
    
    print(len(np.unique(gb.count().reset_index()["ssnamenr"])))
    print()
    print(gb.agg({"ra": len, "a": list, "e": list, "i": list}))


    record = False
    if len(trajectory_df) > 0 and record:
        trajectory_df = trajectory_df.infer_objects()
        trajectory_df["ssnamenr"] = trajectory_df["ssnamenr"].astype(str)
        trajectory_df["fink_class"] = trajectory_df["fink_class"].astype(str)
        trajectory_df["objectId"] = trajectory_df["objectId"].astype(str)

        trajectory_df = trajectory_df.drop(["provisional designation"], axis=1)
        trajectory_df.to_parquet("alert_association/CI_expected_output.parquet")

