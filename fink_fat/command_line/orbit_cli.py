import os
import subprocess

import pandas as pd

import fink_fat


def intro_reset_orbit():  # pragma: no cover
    print("WARNING !!!")
    print(
        "you will loose the previously computed orbital elements and all the associated observations, Continue ? [Y/n]"
    )


def yes_orbit_reset(arguments, orb_res_path, traj_orb_path):  # pragma: no cover
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


def get_orbital_data(config, tr_df_path):  # pragma: no cover
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


def cluster_mode(config, traj_to_orbital):  # pragma: no cover
    traj_to_orbital.to_parquet("tmp_traj.parquet")

    ram_dir = config["SOLVE_ORBIT_PARAMS"]["ram_dir"]
    n_triplets = config["SOLVE_ORBIT_PARAMS"]["n_triplets"]
    noise_ntrials = config["SOLVE_ORBIT_PARAMS"]["noise_ntrials"]
    prop_epoch = config["SOLVE_ORBIT_PARAMS"]["prop_epoch"]
    orbfit_verbose = config["SOLVE_ORBIT_PARAMS"]["orbfit_verbose"]

    master_manager = config["SOLVE_ORBIT_PARAMS"]["manager"]
    principal_group = config["SOLVE_ORBIT_PARAMS"]["principal"]
    secret = config["SOLVE_ORBIT_PARAMS"]["secret"]
    role = config["SOLVE_ORBIT_PARAMS"]["role"]
    executor_env = config["SOLVE_ORBIT_PARAMS"]["exec_env"]
    driver_mem = config["SOLVE_ORBIT_PARAMS"]["driver_memory"]
    exec_mem = config["SOLVE_ORBIT_PARAMS"]["executor_memory"]
    max_core = config["SOLVE_ORBIT_PARAMS"]["max_core"]
    exec_core = config["SOLVE_ORBIT_PARAMS"]["executor_core"]
    orbfit_home = config["SOLVE_ORBIT_PARAMS"]["orbfit_path"]

    application = os.path.join(
        os.path.dirname(fink_fat.__file__),
        "orbit_fitting",
        "orbfit_cluster.py prod",
    )

    application += " " + master_manager
    application += " " + ram_dir
    application += " " + n_triplets
    application += " " + noise_ntrials
    application += " " + prop_epoch
    application += " " + orbfit_verbose

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
        --conf spark.executorEnv.ORBFIT_HOME={} \
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
        orbfit_home,
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
