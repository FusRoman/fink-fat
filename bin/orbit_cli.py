import os
import subprocess

import pandas as pd

import fink_fat


def intro_reset_orbit():
    print("WARNING !!!")
    print(
        "you will loose the previously computed orbital elements and all the associated observations, Continue ? [Y/n]"
    )


def yes_orbit_reset(arguments, orb_res_path, traj_orb_path):
    if os.path.exists(orb_res_path) and os.path.exists(traj_orb_path):
        print("Removing files :\n\t{}\n\t{}".format(orb_res_path, traj_orb_path))
        try:
            os.remove(orb_res_path)
            os.remove(traj_orb_path)
        except OSError as e:
            if arguments["--verbose"]:
                print("Failed with:", e.strerror)
                print("Error code:", e.code)
    else:
        print("File with orbital elements not exists.")


def get_orbital_data(config, tr_df_path):
    # test if the trajectory_df exist in the output directory.
    if os.path.exists(tr_df_path):
        trajectory_df = pd.read_parquet(tr_df_path)
    else:
        print(
            "Trajectory file doesn't exist, run 'fink_fat association (mpc | candidates)' to create it."
        )
        exit()

    # get trajectories with a number of points greater than the orbfit limit
    gb = trajectory_df.groupby(["trajectory_id"]).count().reset_index()
    traj = gb[gb["ra"] >= int(config["SOLVE_ORBIT_PARAMS"]["orbfit_limit"])][
        "trajectory_id"
    ]
    test_orb = trajectory_df["trajectory_id"].isin(traj)
    traj_to_orbital = trajectory_df[test_orb]
    traj_no_orb = trajectory_df[~test_orb]

    return traj_to_orbital, traj_no_orb


def cluster_mode(config, traj_to_orbital):
    traj_to_orbital.to_parquet("tmp_traj.parquet")

    master_manager = config["SOLVE_ORBIT_PARAMS"]["manager"]
    principal_group = config["SOLVE_ORBIT_PARAMS"]["principal"]
    secret = config["SOLVE_ORBIT_PARAMS"]["secret"]
    role = config["SOLVE_ORBIT_PARAMS"]["role"]
    executor_env = config["SOLVE_ORBIT_PARAMS"]["exec_env"]
    driver_mem = config["SOLVE_ORBIT_PARAMS"]["driver_memory"]
    exec_mem = config["SOLVE_ORBIT_PARAMS"]["executor_memory"]
    max_core = config["SOLVE_ORBIT_PARAMS"]["max_core"]
    exec_core = config["SOLVE_ORBIT_PARAMS"]["executor_core"]

    application = os.path.join(
        os.path.dirname(fink_fat.__file__), "orbit_fitting", "orbfit_cluster.py",
    )

    application += " " + master_manager

    spark_submit = "spark-submit \
        --master {} \
        --conf spark.mesos.principal={} \
        --conf spark.mesos.secret={} \
        --conf spark.mesos.role={} \
        --conf spark.executorEnv.HOME={} \
        --driver-memory {}G \
        --executor-memory {}G \
        --conf spark.cores.max={} \
        --conf spark.executor.cores={} \
        {}".format(
        master_manager,
        principal_group,
        secret,
        role,
        executor_env,
        driver_mem,
        exec_mem,
        max_core,
        exec_core,
        application,
    )

    process = subprocess.Popen(
        spark_submit,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        shell=True,
    )

    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(stderr)
        print(stdout)
        exit()

    traj_pdf = pd.read_parquet("res_orb.parquet")

    orbital_columns = [
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
        "chi_reduced",
    ]

    split_df = pd.DataFrame(
        traj_pdf["orbital_elements"].tolist(), columns=orbital_columns
    )
    orbit_results = pd.concat([traj_pdf["trajectory_id"], split_df], axis=1)

    os.remove("tmp_traj.parquet")
    os.remove("res_orb.parquet")

    return orbit_results
