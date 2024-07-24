import pandas as pd
import os
import time
import datetime
import configparser
from typing import Tuple

import astropy.units as u
from astropy.time import Time

from fink_fat.associations.association_orbit import orbit_associations
from fink_fat.streaming_associations.spark_ephem import launch_spark_ephem
from fink_fat.roid_fitting.init_roid_fitting import init_polyast
from fink_fat.associations.stream_association import (
    stream_association,
    trajectories_from_remaining_seeds,
)

from fink_fat.seeding.dbscan_seeding import intra_night_seeding

from fink_fat.command_line.utils_cli import (
    string_to_bool,
    time_window,
    verbose_and_slack,
)
from fink_fat.command_line.orbit_cli import trcand_to_orbit

from fink_fat.command_line.association_cli import get_last_roid_streaming_alert

from fink_fat.others.utils import LoggerNewLine


def get_default_input() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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

    orb_cols = [
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
        "class",
    ]

    orbits = pd.DataFrame(columns=orb_cols)
    old_orbits = pd.DataFrame(columns=orb_cols)
    return trajectory_orb, orbits, old_orbits


def roid_flags(config: configparser.ConfigParser) -> Tuple[bool, list]:
    """
    According to the 'roid_mpc' value in the configuration file,
    start fink-fat with confirmed SSO (flag 3) or
    with one detection objects (flag 1) and candidate alerts (flag 2)

    Parameters
    ----------
    config : configparser.ConfigParser
        object containing the values of the configuration file

    Returns
    -------
    Tuple[bool, list]
        is_mpc: bool
            tell to the spark job returning the alerts from hdfs if it should return alerts flagged as 3
            or alerts flagged as 1 and 2.
        flags: integer list
            [3] if fink-fat should associate confirmed objects (for test and performance)
            or [1, 2] if fink-fat associate candidate alerts
    """
    is_mpc_flag = string_to_bool(config["ASSOC_PARAMS"]["roid_mpc"])
    flags = [3] if is_mpc_flag else [1, 2]
    return is_mpc_flag, flags


def seconds_to_hms(seconds: int) -> Tuple[int, int, int]:
    """
    Convert seconds into hours, minutes and seconds

    Parameters
    ----------
    seconds : int
        number of seconds

    Returns
    -------
    Tuple[int, int, int]
        hours, minutes, seconds
    """
    dt = datetime.timedelta(seconds=seconds)
    days, seconds = dt.days, dt.seconds

    hours = (days * 24 + seconds) // 3600
    minutes = (seconds % 3600) // 60
    sec = seconds % 60
    return hours, minutes, sec


def fitroid_associations(
    arguments: dict,
    config: configparser.ConfigParser,
    logger: LoggerNewLine,
    output_path: str,
):
    """
    (1) Initialise new trajectories
    (2) Associate the alerts flagged by the asteroid science of fink with trajectories
    (3) (re)compute the orbits
    (4) filter trajectories
    (5) post message on slack

    Parameters
    ----------
    arguments: dict
        argument from the command line
    config: configparser.ConfigParser
        object containing the values of the configuration file
    logger: LoggerNewLine
        custom logger object of fink-fat
    output_path: str
        location of the fink-fat outputs
            - trajectory_df
                alerts and their contents linked by fink-fat
            - fit_roid_df
                parameters of the polyfit predictors
            - trajectory_orb
                alerts and their contents for trajectories with orbit
            - orbits
                orbital parameters
    """

    start_assoc_time = time.time()

    is_mpc, flags = roid_flags(config)

    last_night = datetime.datetime.now() - datetime.timedelta(days=1)
    last_night = last_night.strftime("%Y-%m-%d")
    if arguments["--night"]:
        last_night = arguments["--night"]

    year, month, day = last_night.split("-")
    # path to the orbits data
    path_orbit = os.path.join(output_path, "orbital.parquet")
    path_trajectory_orb = os.path.join(output_path, "trajectory_orb.parquet")
    path_old_orbits = os.path.join(output_path, "old_orbits.parquet")

    # path to the candidate trajectories
    path_fit_roid = os.path.join(output_path, "fit_roid.parquet")
    path_trajectory_df = os.path.join(output_path, "trajectory_df.parquet")

    if os.path.exists(path_fit_roid):
        fit_roid_last_assoc = pd.read_parquet(
            path_fit_roid, columns=["last_assoc_date"]
        )
        last_assoc_date = max(
            pd.to_datetime(fit_roid_last_assoc["last_assoc_date"], format="%Y-%m-%d")
        )

        current_date = datetime.datetime.strptime(last_night, "%Y-%m-%d")

        if last_assoc_date == current_date:
            logger.newline()
            logger.error("Association already done for this night.")
            logger.info(
                """
                last recorded association date: {} == current date: {}
                Wait a next observation night and the end of the alert stream to start a new run of association.
                """.format(
                    current_date, last_assoc_date
                )
            )
            exit()
        if last_assoc_date > current_date:
            logger.newline()
            logger.error(
                """
                last recorded association date: {} > current date: {}
                Try to associates alerts from a night before the last night recorded in the trajectories
                """.format(
                    current_date, last_assoc_date
                )
            )
            logger.info(
                "Maybe try with a more recent night or reset the associations with 'fink_fat association -r'"
            )
            exit()

    # load the alerts from the last streaming night (roid science module with fink-fat must have been run)
    alerts_night = get_last_roid_streaming_alert(
        config, last_night, output_path, is_mpc, arguments["--verbose"], logger
    )

    if len(alerts_night) == 0:
        if arguments["--verbose"]:
            t_before = time.time()
            logger.info(
                "No alerts in the current night, compute the ephemeries for the next night"
            )
        # even if no alerts for the current night, compute the ephemeries for the next night in any case
        launch_spark_ephem(
            config,
            path_orbit,
            os.path.join(output_path, "ephem.parquet"),
            year,
            month,
            day,
        )

        if arguments["--verbose"]:
            logger.info(
                f"ephemeries computing time: {time.time() - t_before:.4f} seconds"
            )
            logger.newline()
            hours, minutes, secondes = seconds_to_hms(time.time() - start_assoc_time)
            logger.info(
                f"total execution time: {hours} hours, {minutes} minutes, {secondes} seconds"
            )
            logger.info("END OF THE FITROID ASSOCIATION")
            logger.info("------------------")
            logger.newline(3)

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

    trajectory_orb, orbits, old_orbits = get_default_input()
    new_or_updated_orbits = []

    # associations between trajectories with orbit and the new alerts associated with ephemeride
    if os.path.exists(path_orbit) and os.path.exists(path_trajectory_orb):
        if arguments["--verbose"]:
            logger.info("orbits file detected, start the associations with the orbits")
        orbits = pd.read_parquet(path_orbit)
        trajectory_orb = pd.read_parquet(path_trajectory_orb)

        orbits, trajectory_orb, old_orbits, updated_ssocandid = orbit_associations(
            config, alerts_night, trajectory_orb, orbits, last_night, logger, True
        )
        new_or_updated_orbits += updated_ssocandid.tolist()

        if os.path.exists(path_old_orbits):
            save_old_orbits = pd.read_parquet(path_old_orbits)
            old_orbits = pd.concat([save_old_orbits, old_orbits])
        old_orbits.to_parquet(path_old_orbits, index=False)

    # take alerts associated with trajectory predictors from the stream
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
                trajectory_df, fit_roid_df, seeds, logger, True
            )
            nb_deviating_trcand = len(fit_roid_df) - nb_trcand_before

        # create new trajectories from the remaining seeds with no associations
        seeds_no_assoc, new_fit_df, nb_new_trcand = trajectories_from_remaining_seeds(
            seeds, fit_roid_df
        )
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

    with pd.option_context("mode.chained_assignment", None):
        trajectory_df["trajectory_id"] = trajectory_df["trajectory_id"].astype(int)

    if "updated" not in trajectory_df:
        trajectory_df["updated"] = "N"

    trajectory_df, fit_roid_df, trajectory_orb, orbits, new_ssoCandId = trcand_to_orbit(
        config,
        trajectory_df,
        trajectory_orb,
        fit_roid_df,
        orbits,
        last_night,
        logger,
        True,
    )
    new_or_updated_orbits += new_ssoCandId.tolist()

    nb_tr_before_tw = len(fit_roid_df)
    trajectory_df, fit_roid_df = time_window(
        trajectory_df,
        fit_roid_df,
        Time(last_night).jd,
        int(config["TW_PARAMS"]["predict_function_keep_limit"]),
    )
    nb_after_tw = len(fit_roid_df)
    nb_remove_tw = nb_tr_before_tw - nb_after_tw

    if arguments["--verbose"]:
        logger.info("start to compute chi-filter")
        t_before = time.time()

    # trajectory_df, fit_roid_df = chi_filter(trajectory_df, fit_roid_df, 10e-5)

    if arguments["--verbose"]:
        chi_filter_time = time.time() - t_before
        logger.info(f"chi-filter elapsed time: {chi_filter_time:.4f} seconds")

    nb_remove_chi = nb_after_tw - len(fit_roid_df)

    verbose_and_slack(
        (
            nb_tr_last_night,
            nb_deviating_trcand,
            nb_new_trcand,
            nb_remove_tw,
            nb_remove_chi,
        ),
        (
            trajectory_df,
            fit_roid_df,
            trajectory_orb,
            orbits,
            old_orbits,
            new_or_updated_orbits,
        ),
        last_night,
        arguments["--verbose"],
        logger,
        config["SLACK"]["post_on_slack"],
    )

    if arguments["--verbose"]:
        logger.info("write the results")

    with pd.option_context("mode.chained_assignment", None):
        trajectory_df["trajectory_id"] = trajectory_df["trajectory_id"].astype(int)
        fit_roid_df["trajectory_id"] = fit_roid_df["trajectory_id"].astype(int)
        fit_roid_df["last_assoc_date"] = f"{year}-{month}-{day}"

    trajectory_df = trajectory_df.drop("updated", axis=1)
    trajectory_df.to_parquet(path_trajectory_df, index=False)

    assert (fit_roid_df["trajectory_id"].value_counts() == 1).all()

    fit_roid_df.to_parquet(path_fit_roid, index=False)
    trajectory_orb.to_parquet(path_trajectory_orb, index=False)
    orbits.to_parquet(path_orbit, index=False)

    if len(orbits) > 0:
        # compute the ephemerides for the next observation night
        if arguments["--verbose"]:
            logger.info("start to compute ephemerides using spark")
            t_before = time.time()

        launch_spark_ephem(
            config,
            path_orbit,
            os.path.join(output_path, "ephem.parquet"),
            year,
            month,
            day,
        )

        if arguments["--verbose"]:
            logger.info(
                f"ephemeries computing time: {time.time() - t_before:.4f} seconds"
            )
            logger.newline()

    if arguments["--verbose"]:
        hours, minutes, secondes = seconds_to_hms(time.time() - start_assoc_time)
        logger.info(
            f"total execution time: {hours} hours, {minutes} minutes, {secondes} seconds"
        )
        logger.info("END OF THE FITROID ASSOCIATION")
        logger.info("------------------")
        logger.newline(3)

    return
