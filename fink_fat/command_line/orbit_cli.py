import os
import subprocess
from typing import Tuple

import pandas as pd
import numpy as np

import fink_fat
from fink_fat.others.utils import init_logging, LoggerNewLine
from fink_fat.orbit_fitting.orbfit_local import (
    get_last_detection,
    compute_df_orbit_param,
)
from fink_fat.command_line.utils_cli import assig_tags


def intro_reset_orbit():  # pragma: no cover
    logger = init_logging()
    logger.info("WARNING !!!")
    logger.info(
        "you will loose the previously computed orbital elements and all the associated observations, Continue ? [Y/n]"
    )


def yes_orbit_reset(arguments, orb_res_path, traj_orb_path):  # pragma: no cover
    logger = init_logging()
    if os.path.exists(orb_res_path) and os.path.exists(traj_orb_path):
        logger.info("Removing files :\n\t{}\n\t{}".format(orb_res_path, traj_orb_path))
        try:
            os.remove(orb_res_path)
            os.remove(traj_orb_path)
        except OSError as e:
            if arguments["--verbose"]:
                logger.info("Failed with:", e.strerror)
                logger.info("Error code:", e.code)
    else:
        logger.info("File with orbital elements not exists.")


def get_orbital_data(config, tr_df_path):  # pragma: no cover
    logger = init_logging()
    # test if the trajectory_df exist in the output directory.
    if os.path.exists(tr_df_path):
        trajectory_df = pd.read_parquet(tr_df_path)
    else:
        logger.info(
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


def cluster_mode(
    config: dict, traj_to_orbital: pd.DataFrame, year: str, month: str, day: str
) -> pd.DataFrame:  # pragma: no cover
    """
    Compute orbits using the cluster mode

    Parameters
    ----------
    config : dict
        data of the configuration file
    traj_to_orbital : pd.DataFrame
        trajectory dataframe
        mandatory columns: ra, dec, jd, magpsf, fid, trajectory_id

    Returns
    -------
    pd.DataFrame
        dataframe containing the orbital parameters of the input trajectories
        columns: ref_epoch, a, e, i, long. node, arg. peric, mean anomaly, rms_a, rms_e,
        rms_i, rms_long. node, rms_arg. peric, rms_mean anomaly, chi_reduced,
        last_ra, last_dec, last_jd, last_mag, last_fid, ssoCandId

    Examples
    --------
    >>> tr = pd.read_parquet(traj_sample)
    >>> config, output_path = init_cli({"--config": ""})
    >>> orbits = cluster_mode(config, tr)
    >>> cols_to_drop = ["rms_a", "rms_e", "rms_i", "rms_long. node", "rms_arg. peric", "rms_mean anomaly", "chi_reduced"]
    >>> assert_frame_equal(
    ...     orbits.drop(cols_to_drop, axis=1).round(decimals=4),
    ...     ts2.cluster_mode_cli_test.drop(cols_to_drop, axis=1).round(decimals=4)
    ... )
    """
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
    application += " " + year
    application += " " + month
    application += " " + day

    # FIXME
    # temporary dependencies (only during the performance test phase)
    FINK_FAT = "/home/roman.le-montagner/home_big_storage/Doctorat/Asteroids/fink-fat/dist/fink_fat-1.0.0-py3.9.egg"
    FINK_SCIENCE = "/home/roman.le-montagner/home_big_storage/Doctorat/fink-science/dist/fink_science-4.4-py3.7.egg"

    spark_submit = f"spark-submit \
        --master {master_manager} \
        --conf spark.mesos.principal={principal_group} \
        --conf spark.mesos.secret={secret} \
        --conf spark.mesos.role={role} \
        --conf spark.executorEnv.HOME={executor_env} \
        --driver-memory {driver_mem}G \
        --executor-memory {exec_mem}G \
        --conf spark.cores.max={max_core} \
        --conf spark.executor.cores={exec_core} \
        --conf spark.driver.maxResultSize=6G\
        --conf spark.sql.execution.arrow.pyspark.enabled=true\
        --conf spark.sql.execution.arrow.maxRecordsPerBatch=1000000\
        --conf spark.kryoserializer.buffer.max=512m\
        --conf spark.executorEnv.ORBFIT_HOME={orbfit_home} \
        --py-files {FINK_FAT},{FINK_SCIENCE}\
        {application}"

    process = subprocess.run(spark_submit, shell=True)

    if process.returncode != 0:
        logger = init_logging()
        logger.info(process.stderr)
        logger.info(process.stdout)
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

    last_det = get_last_detection(traj_to_orbital)
    orbit_results = orbit_results.merge(last_det, on="trajectory_id")

    os.remove("tmp_traj.parquet")
    os.remove("res_orb.parquet")

    return orbit_results


def switch_local_cluster(
    config: dict, traj_orb: pd.DataFrame, year: str, month: str, day: str, verbose=False
) -> pd.DataFrame:
    """
    Run the orbit fitting and choose cluster mode if the number of trajectories are above
    the local limit set in the config file

    Parameters
    ----------
    config : dict
        contains data from the config file
    traj_orb : pd.DataFrame
        contains the alerts of the trajectories

    Returns
    -------
    new_orbit_pdf : pd.DataFrame
        contains the orbital parameters estimated from the traj_orb parameters by the orbit fitting.
    """
    nb_orb = len(traj_orb["trajectory_id"].unique())
    if nb_orb > int(config["SOLVE_ORBIT_PARAMS"]["local_mode_limit"]):
        traj_cols = traj_orb.columns
        if "estimator_id" in traj_cols:
            traj_orb = traj_orb.drop("estimator_id", axis=1)
        if "ffdistnr" in traj_cols:
            traj_orb = traj_orb.drop("ffdistnr", axis=1)
        new_orbit_pdf = cluster_mode(config, traj_orb, year, month, day)
    else:
        config_epoch = config["SOLVE_ORBIT_PARAMS"]["prop_epoch"]
        prop_epoch = None if config_epoch == "None" else float(config_epoch)

        prop_epoch = config["SOLVE_ORBIT_PARAMS"]["prop_epoch"]
        # return orbit results from local mode
        new_orbit_pdf = compute_df_orbit_param(
            traj_orb,
            int(config["SOLVE_ORBIT_PARAMS"]["cpu_count"]),
            config["SOLVE_ORBIT_PARAMS"]["ram_dir"],
            int(config["SOLVE_ORBIT_PARAMS"]["n_triplets"]),
            int(config["SOLVE_ORBIT_PARAMS"]["noise_ntrials"]),
            prop_epoch=float(prop_epoch) if prop_epoch != "None" else None,
            verbose_orbfit=int(config["SOLVE_ORBIT_PARAMS"]["orbfit_verbose"]),
            verbose=verbose,
        ).drop("provisional designation", axis=1)
    return new_orbit_pdf


def trcand_to_orbit(
    config: dict,
    trajectory_df: pd.DataFrame,
    trajectory_orb: pd.DataFrame,
    trparams_df: pd.DataFrame,
    orbits: pd.DataFrame,
    last_night: str,
    logger: LoggerNewLine,
    verbose: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute and return the orbits for the trajectories with enough points found by the prediction functions

    Parameters
    ----------
    arguments : dict
        the commend line arguments
    config : dict
        the data from the config file
    trajectory_df : pd.DataFrame
        trajectories found by the function predictors
    trajectory_orb : pd.DataFrame
        trajectories with orbits
    trparams_df : pd.DataFrame
        the fit functions parameters
    orbits : pd.DataFrame
        the orbits parameters
    logger : LoggerNewLine
        the logging object used to print the logs
    verbose : bool
        if true, print the logs

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame,]
        * trajectories found by the prediction functions without those sent to the orbit fitting
        * kalman filters without those with the associated trajectories sent to the orbit fitting
        * trajectories with the ones with orbits
        * the orbits parameters with the new ones
    """
    # get the trajectories large enough to go to the orbit fitting
    orbit_limit = int(config["SOLVE_ORBIT_PARAMS"]["orbfit_limit"])
    traj_size = trajectory_df["trajectory_id"].value_counts()
    large_traj = traj_size[traj_size >= orbit_limit]

    if len(large_traj) == 0:
        return trajectory_df, trparams_df, trajectory_orb, orbits

    # to be send to the orbit fitting,
    # a trajectory must have a number of point above the orbit point limit set in the config file
    # and must have been updated during the current night
    traj_to_orb = trajectory_df[
        (trajectory_df["trajectory_id"].isin(large_traj.index.values))
        & (trajectory_df["updated"] == "Y")
    ].sort_values(["trajectory_id", "jd"])
    nb_traj_to_orb = len(traj_to_orb["trajectory_id"].unique())
    if verbose:
        logger.info(
            f"number of trajectories send to the orbit fitting: {nb_traj_to_orb}"
        )

    if nb_traj_to_orb == 0:
        # no trajectory updated during this night to send to orbit fitting
        return trajectory_df, trparams_df, trajectory_orb, orbits

    year, month, day = last_night.split("-")
    new_orbits = switch_local_cluster(config, traj_to_orb, year, month, day, verbose)
    new_orbits = new_orbits[new_orbits["a"] != -1.0]
    if verbose:
        logger.info(
            f"number of orbit fitted: {len(new_orbits)} ({(len(new_orbits) / nb_traj_to_orb) * 100} %)"
        )

    # get the trajectories with orbit and assign the ssoCandId
    new_traj_id = new_orbits["trajectory_id"].values
    new_traj_orb = traj_to_orb[traj_to_orb["trajectory_id"].isin(new_traj_id)]
    new_orbits, new_traj_orb = assig_tags(new_orbits, new_traj_orb, len(orbits) + 1)

    # add the new orbits and new trajectories with orbit
    orbits = pd.concat([orbits, new_orbits])
    trajectory_orb = pd.concat([trajectory_orb, new_traj_orb])

    # remove the new trajectories with orbit from trajectory_df and the associated kalman filters
    trajectory_df = trajectory_df[
        ~trajectory_df["trajectory_id"].isin(new_traj_id)
    ].reset_index(drop=True)
    trparams_df = trparams_df[
        ~trparams_df["trajectory_id"].isin(new_traj_id)
    ].reset_index(drop=True)

    failed_orbit = np.setdiff1d(large_traj.index.values, new_traj_id)
    # FIXME
    # as the trajectory_id change between the night if the trajectories are updated, this solution to keep track
    # of the failed orbit as well as the number of time they have been sent to the orbit fitting doesn't works.
    # Remove the failed orbit in the meantime
    # mask_failed = trparams_df["trajectory_id"].isin(failed_orbit)
    # with pd.option_context("mode.chained_assignment", None):
    #     trparams_df.loc[mask_failed, "orbfit_test"] = (
    #         trparams_df.loc[mask_failed, "orbfit_test"] + 1
    #     )

    # Remove the failed orbit
    trajectory_df = trajectory_df[~trajectory_df["trajectory_id"].isin(failed_orbit)]
    trparams_df = trparams_df[~trparams_df["trajectory_id"].isin(failed_orbit)]
    return trajectory_df, trparams_df, trajectory_orb, orbits


if __name__ == "__main__":
    from fink_science.tester import spark_unit_tests
    from pandas.testing import assert_frame_equal  # noqa: F401
    from fink_fat.command_line.utils_cli import init_cli  # noqa: F401
    import fink_fat.test.test_sample_2 as ts2  # noqa: F401

    globs = globals()

    traj_sample = "fink_fat/test/cluster_test/trajectories_sample.parquet"
    globs["traj_sample"] = traj_sample

    # Run the test suite
    spark_unit_tests(globs)
