import numpy as np
import pandas as pd
import os
import datetime
import configparser
from collections import Counter
from typing import Tuple

import astropy.units as u
from astropy.time import Time

from fink_fat.associations.association_orbit import orbit_associations
from fink_fat.streaming_associations.spark_ephem import launch_spark_ephem
from fink_fat.roid_fitting.init_roid_fitting import init_polyast
from fink_fat.associations.stream_association import stream_association

from fink_fat.seeding.dbscan_seeding import intra_night_seeding

from fink_fat.command_line.utils_cli import string_to_bool, time_window, chi_filter
from fink_fat.command_line.orbit_cli import trcand_to_orbit

from fink_fat.command_line.association_cli import get_last_roid_streaming_alert

from fink_fat.others.utils import LoggerNewLine


def get_default_input():
    """_summary_

    Returns
    -------
    _type_
        _description_
    """
    trajectory_orb = pd.DataFrame(
        columns=[
            "objectId",
            "candid",
            "ra",
            "dec",
            "ssnamenr",
            "jd",
            "magpsf",
            "fid",
            "nid",
            "ssoCandId",
            "ffdistnr",
        ]
    )

    orbits = pd.DataFrame(
        columns=[
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
            "last_ra",
            "last_dec",
            "last_jd",
            "last_mag",
            "last_fid",
            "ssoCandId",
        ]
    )
    return trajectory_orb, orbits


def roid_flags(config: configparser.ConfigParser) -> Tuple[bool, list]:
    is_mpc_flag = string_to_bool(config["ASSOC_PARAMS"]["roid_mpc"])
    return is_mpc_flag, [3] if is_mpc_flag else [1, 2]


def fitroid_associations(
    arguments: dict,
    config: configparser.ConfigParser,
    logger: LoggerNewLine,
    output_path: str,
):
    """ """

    is_mpc, flags = roid_flags(config)

    last_night = datetime.datetime.now() - datetime.timedelta(days=1)
    last_night = last_night.strftime("%Y-%m-%d")
    if arguments["--night"]:
        last_night = arguments["--night"]

    # path to the orbits data
    path_orbit = os.path.join(output_path, "orbital.parquet")
    path_trajectory_orb = os.path.join(output_path, "trajectory_orb.parquet")
    path_old_orbits = os.path.join(output_path, "old_orbits.parquet")

    # path to the candidate trajectories
    path_fit_roid = os.path.join(output_path, "fit_roid.parquet")
    path_trajectory_df = os.path.join(output_path, "trajectory_df.parquet")

    # load the alerts from the last streaming night (roid science module with fink-fat must have been run)
    alerts_night = get_last_roid_streaming_alert(
        config, last_night, output_path, is_mpc, arguments["--verbose"], logger
    )

    if len(alerts_night) == 0:
        if arguments["--verbose"]:
            logger.info("No alerts in the current night, compute the ephemeries for the next night")
        # even if no alerts for the current night, compute the ephemeries for the next night in any case
        year, month, day = last_night.split("-")
        launch_spark_ephem(
            config, path_orbit, os.path.join(output_path, "ephem.parquet"), year, month, day
        )
        return

    if arguments["--verbose"]:
        logger.info(
            """
STATISTICS - STREAMING NIGHT
-----------------------

roid count:
{}
                    """.format(
                alerts_night["roid"].value_counts().sort_index()
            )
        )

    trajectory_orb, orbits = get_default_input()

    # associations between trajectories with orbit and the new alerts associated with ephemeride
    if os.path.exists(path_orbit) and os.path.exists(path_trajectory_orb):
        if arguments["--verbose"]:
            logger.info("orbits file detected, start the associations with the orbits")
        orbits = pd.read_parquet(path_orbit)
        trajectory_orb = pd.read_parquet(path_trajectory_orb)
        orbits, trajectory_orb, old_orbits = orbit_associations(
            config, alerts_night, trajectory_orb, orbits, last_night, logger, True
        )
        if os.path.exists(path_old_orbits):
            save_old_orbits = pd.read_parquet(path_old_orbits)
            old_orbits = pd.concat([save_old_orbits, old_orbits])
        old_orbits.to_parquet(path_old_orbits, index=False)

    keep_flags = flags + [4]
    sso_night = alerts_night[alerts_night["roid"].isin(keep_flags)]

    if len(sso_night) == 0:
        if arguments["--verbose"]:
            logger.info(
                "No associations with predictions function in this night\n save updated orbits and continue"
            )
        trajectory_orb.to_parquet(path_trajectory_orb, index=False)
        orbits.to_parquet(path_orbit, index=False)
        return

    if arguments["--verbose"]:
        logger.info("start seeding")
    intra_sep = (
        float(config["ASSOC_PARAMS"]["intra_night_separation"]) * u.arcsecond
    ).to("deg")
    seeds = intra_night_seeding(sso_night, intra_sep)

    nb_tr_last_night = 0
    nb_deviating_trcand = 0
    nb_new_trcand = 0

    if os.path.exists(path_trajectory_df) and os.path.exists(path_fit_roid):
        trajectory_df = pd.read_parquet(path_trajectory_df)
        fit_roid_df = pd.read_parquet(path_fit_roid)
        nb_tr_last_night = len(fit_roid_df)

        if len(sso_night[sso_night["roid"] == 4]) != 0:
            nb_trcand_before = len(fit_roid_df)
            trajectory_df, fit_roid_df = stream_association(
                trajectory_df, fit_roid_df, seeds, logger, True, confirmed_sso=True
            )
            nb_deviating_trcand = len(fit_roid_df) - nb_trcand_before

        # get the seeds with no associations to run the initialization of the fit functions them.
        test_seeds = seeds[seeds["trajectory_id"] != -1]
        if len(test_seeds) != 0:
            test_seeds = (
                test_seeds.fillna(-1)
                .groupby("trajectory_id")
                .agg(
                    tr_no_assoc=("estimator_id", lambda x: (np.array(x) == -1).all()),
                    list_est=("estimator_id", list),
                )
                .reset_index()
            )

            test_seeds = test_seeds[test_seeds["tr_no_assoc"]]
            seeds_no_assoc = seeds[
                seeds["trajectory_id"].isin(test_seeds["trajectory_id"])
            ]
            max_traj_id = fit_roid_df["trajectory_id"].max() + 1
            clusters_id = seeds_no_assoc["trajectory_id"].unique()
            new_traj_id = np.arange(max_traj_id, max_traj_id + len(clusters_id)).astype(
                int
            )
            assert len(new_traj_id) == len(clusters_id)
            map_new_tr = {
                cl_id: tr_id for tr_id, cl_id in zip(new_traj_id, clusters_id)
            }
            with pd.option_context("mode.chained_assignment", None):
                seeds_no_assoc["trajectory_id"] = seeds_no_assoc["trajectory_id"].map(
                    map_new_tr
                )
            new_fit_df = init_polyast(seeds_no_assoc)
            with pd.option_context("mode.chained_assignment", None):
                seeds_no_assoc["updated"] = "Y"
                new_fit_df["orbfit_test"] = 0
            nb_new_trcand = len(new_fit_df)

            trajectory_df = pd.concat([trajectory_df, seeds_no_assoc])
            fit_roid_df = pd.concat([fit_roid_df, new_fit_df])

    else:
        if arguments["--verbose"]:
            logger.info(
                "NO prediction functions file detected, start to init prediction functions from the intra night seeds"
            )
        trajectory_df = seeds[seeds["trajectory_id"] != -1.0]
        fit_roid_df = init_polyast(trajectory_df)
        with pd.option_context("mode.chained_assignment", None):
            trajectory_df["updated"] = "Y"
            fit_roid_df["orbfit_test"] = 0
        nb_new_trcand = len(fit_roid_df)

    trajectory_df["trajectory_id"] = trajectory_df["trajectory_id"].astype(int)
    if "updated" not in trajectory_df:
        trajectory_df["updated"] = "N"
    trajectory_df, fit_roid_df, trajectory_orb, orbits = trcand_to_orbit(
        config,
        trajectory_df,
        trajectory_orb,
        fit_roid_df,
        orbits,
        last_night,
        logger,
        True,
    )

    nb_tr_before_tw = len(fit_roid_df)
    trajectory_df, fit_roid_df = time_window(
        trajectory_df,
        fit_roid_df,
        Time(last_night).jd,
        int(config["TW_PARAMS"]["predict_function_keep_limit"]),
    )
    nb_after_tw = len(fit_roid_df)
    nb_remove_tw = nb_tr_before_tw - nb_after_tw

    trajectory_df, fit_roid_df = chi_filter(trajectory_df, fit_roid_df, 10e-5)
    nb_remove_chi = nb_after_tw - len(fit_roid_df)

    if arguments["--verbose"]:
        nb_trcand = len(fit_roid_df)
        diff_last_night = nb_trcand - nb_tr_last_night

        nb_orbits = len(orbits)
        traj_cand_size = Counter(
            trajectory_df["trajectory_id"].value_counts().sort_index()
        )
        traj_orbits_size = Counter(
            trajectory_orb["ssoCandId"].value_counts().sort_index()
        )

        logger.info(
            f"""
STATISTICS - ASSOCIATIONS
----------

number of orbits: {nb_orbits}
number of total trcand (end of the processing): {nb_trcand}
 * difference from last night: {f"+{diff_last_night}" if diff_last_night > 0 else f"{diff_last_night}"}
 * number of deviating trajectory: {nb_deviating_trcand}
 * number of new trajectory: {nb_new_trcand}
 * number of trajectories removed by time_window: {nb_remove_tw}
 * number of trajectories removed by chi_square filter: {nb_remove_chi}

trajectories candidate size:
{Counter(traj_cand_size)}

orbits trajectories size:
{Counter(traj_orbits_size)}
"""
        )
        logger.newline(2)

    if arguments["--verbose"]:
        logger.info("write the results")

    with pd.option_context("mode.chained_assignment", None):
        trajectory_df["trajectory_id"] = trajectory_df["trajectory_id"].astype(int)
        fit_roid_df["trajectory_id"] = fit_roid_df["trajectory_id"].astype(int)

    trajectory_df = trajectory_df.drop("updated", axis=1)
    trajectory_df.to_parquet(path_trajectory_df, index=False)

    assert (fit_roid_df["trajectory_id"].value_counts() == 1).all()

    fit_roid_df.to_parquet(path_fit_roid, index=False)
    trajectory_orb.to_parquet(path_trajectory_orb, index=False)
    orbits.to_parquet(path_orbit, index=False)

    # compute the ephemerides for the next observation night
    if arguments["--verbose"]:
        logger.info("start to compute ephemerides using spark")

    year, month, day = last_night.split("-")
    launch_spark_ephem(
        config, path_orbit, os.path.join(output_path, "ephem.parquet"), year, month, day
    )

    if arguments["--verbose"]:
        logger.info("END OF THE FITROID ASSOCIATION")
        logger.info("------------------")
        logger.newline(3)

    return
