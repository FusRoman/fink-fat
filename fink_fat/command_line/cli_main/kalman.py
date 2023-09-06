import pandas as pd
import numpy as np
import os
import datetime
from collections import Counter
from typing import Tuple

import astropy.units as units

from astropy.time import Time
from fink_fat.others.utils import init_logging
from fink_fat.command_line.utils_cli import get_class
from fink_fat.seeding.dbscan_seeding import (
    intra_night_seeding,
    seeding_completude,
    seeding_purity,
)
from fink_fat.associations.association_orbit import orbit_associations
from fink_fat.associations.association_kalman import kalman_association
from fink_fat.kalman.init_kalman import init_kalman
from fink_fat.command_line.orbit_cli import kalman_to_orbit


def get_default_input() -> (
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
):
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

    trajectory_df = pd.DataFrame(
        columns=[
            "objectId",
            "candid",
            "ra",
            "dec",
            "jd",
            "magpsf",
            "sigmapsf",
            "fid",
            "ssnamenr",
            "roid",
            "estimator_id",
            "ffdistnr",
            "trajectory_id",
        ]
    )

    kalman_df = pd.DataFrame(
        columns=[
            "trajectory_id",
            "ra_0",
            "dec_0",
            "ra_1",
            "dec_1",
            "jd_0",
            "jd_1",
            "mag_1",
            "fid_1",
            "dt",
            "vel_ra",
            "vel_dec",
            "kalman",
            "orbfit_test",
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
    return trajectory_df, kalman_df, trajectory_orb, orbits


def load_sso_data(sso_path: pd.DataFrame) -> pd.DataFrame:
    sso_night = pd.read_parquet(sso_path)
    candidate_pdf = pd.json_normalize(sso_night["candidate"]).drop("candid", axis=1)
    roid_pdf = pd.json_normalize(sso_night["ff_roid"])
    sso_night = pd.concat(
        [sso_night, candidate_pdf, roid_pdf],
        axis=1,
    )
    sso_night = sso_night  # .explode(["estimator_id", "ffdistnr"])

    cols_to_keep = [
        "objectId",
        "candid",
        "ra",
        "dec",
        "jd",
        "magpsf",
        "sigmapsf",
        "fid",
        "ssnamenr",
        "roid",
        "estimator_id",
        "ffdistnr",
    ]

    return sso_night[cols_to_keep]


def kalman_window_management(
    trajectory_df: pd.DataFrame,
    kalman_df: pd.DataFrame,
    config: dict,
    current_datetime: datetime.datetime,
) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    # kalman time window management
    test_tr_window = (
        trajectory_df.sort_values(["trajectory_id", "jd"])
        .groupby("trajectory_id")
        .agg(max_jd=("jd", np.max), updated_tag=("updated", lambda x: list(x)[0]))
        .reset_index()
    )
    test_tr_window["diff_win"] = Time(current_datetime).jd - test_tr_window["max_jd"]
    mask_window = (
        test_tr_window["diff_win"] <= int(config["TW_PARAMS"]["kalman_keep_limit"])
    ) | (test_tr_window["updated_tag"] == "Y")
    tr_to_keep = test_tr_window[mask_window]["trajectory_id"]
    nb_removed_kalman = len(test_tr_window) - len(tr_to_keep)
    trajectory_df = trajectory_df[trajectory_df["trajectory_id"].isin(tr_to_keep)]
    kalman_df = kalman_df[kalman_df["trajectory_id"].isin(tr_to_keep)]
    return trajectory_df, kalman_df, nb_removed_kalman


def seeds_new_id(seeds: pd.DataFrame, kalman_df: pd.DataFrame) -> pd.DataFrame:
    max_traj_id = kalman_df["trajectory_id"].max() + 1
    clusters_id = seeds["trajectory_id"].unique()
    new_traj_id = np.arange(max_traj_id, max_traj_id + len(clusters_id)).astype(int)
    assert len(new_traj_id) == len(clusters_id)
    map_new_tr = {cl_id: tr_id for tr_id, cl_id in zip(new_traj_id, clusters_id)}
    with pd.option_context("mode.chained_assignment", None):
        seeds["trajectory_id"] = seeds["trajectory_id"].map(map_new_tr)
    return seeds


def cli_kalman_associations(arguments: dict, config: dict, output_path: str):
    logger = init_logging()

    # get the path according to the class mpc or candidates
    output_path, object_class = get_class(arguments, output_path)
    is_verbose = arguments["--verbose"]

    last_night = datetime.datetime.now() - datetime.timedelta(days=1)
    last_night = last_night.strftime("%Y-%m-%d")
    if arguments["--night"]:
        last_night = arguments["--night"]
    current_datetime = datetime.datetime.strptime(last_night, "%Y-%m-%d")
    if is_verbose:
        logger.info(f"start to process date {last_night}")
        logger.newline()

    roid_output_path = config["OUTPUT"]["fink_roid_output"]

    roid_output_path = os.path.join(
        roid_output_path,
        f"year={current_datetime.year}/month={current_datetime.month}/day={current_datetime.day}",
    )
    if not os.path.exists(roid_output_path):
        logger.info(f"roid output path does not exists, {roid_output_path}")
        exit()

    alerts_night = load_sso_data(roid_output_path)
    if is_verbose:
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

    trajectory_df, kalman_df, trajectory_orb, orbits = get_default_input()

    # get path of the fink fat outputs
    path_orbit = os.path.join(output_path, "orbital.parquet")
    path_trajectory_orb = os.path.join(output_path, "trajectory_orb.parquet")

    path_kalman = os.path.join(output_path, "kalman.pkl")
    path_trajectory_df = os.path.join(output_path, "trajectory_df.parquet")
    path_old_orbits = os.path.join(output_path, "old_orbits.parquet")

    if os.path.exists(path_orbit) and os.path.exists(path_trajectory_orb):
        orbits = pd.read_parquet(path_orbit)
        trajectory_orb = pd.read_parquet(path_trajectory_orb)

    if os.path.exists(path_trajectory_df) and os.path.exists(path_kalman):
        trajectory_df = pd.read_parquet(path_trajectory_df)
        kalman_df = pd.read_pickle(path_kalman)

        if len(trajectory_df) > 0:
            last_tr_date = Time(kalman_df["jd_1"], format="jd").to_datetime()
            if len(orbits) > 0:
                last_orbits_date = Time(orbits["last_jd"], format="jd").to_datetime()
            else:
                last_orbits_date = last_tr_date
            last_request_date = max(last_tr_date.max(), last_orbits_date.max())

            if last_request_date == current_datetime:
                logger.newline()
                logger.info("ERROR !!!")
                logger.info("Association already done for this night.")
                logger.info("Wait a next observation night to do new association")
                logger.info("or run 'fink_fat solve_orbit' to get orbital_elements.")
                exit()
            if last_request_date > current_datetime:
                logger.newline()
                logger.info("ERROR !!!")
                logger.info(
                    "Query alerts from a night before the last night in the recorded trajectory/old_observations."
                )
                logger.info(
                    "Maybe try with a more recent night or reset the associations with 'fink_fat association -r'"
                )
                exit()

    orbits, trajectory_orb, old_orbits = orbit_associations(
        config, alerts_night, trajectory_orb, orbits, logger, is_verbose
    )
    if os.path.exists(path_old_orbits):
        save_old_orbits = pd.read_parquet(path_old_orbits)
        old_orbits = pd.concat([save_old_orbits, old_orbits])
    old_orbits.to_parquet(path_old_orbits, index=False)

    confirmed_sso = object_class == "Solar System MPC"
    if confirmed_sso:
        keep_flags = [3, 4]
    else:
        keep_flags = [1, 2, 4]
    sso_night = alerts_night[alerts_night["roid"].isin(keep_flags)]

    if is_verbose:
        logger.info("start seeding")

    intra_sep = (
        float(config["ASSOC_PARAMS"]["intra_night_separation"]) * units.arcsecond
    ).to("deg")
    seeds = intra_night_seeding(sso_night, intra_sep)

    if confirmed_sso and is_verbose:
        logger.info(
            f"purity: {seeding_purity(seeds)}%, completude: {seeding_completude(seeds)}"
        )
        logger.newline()

    nb_deviating_kalman = 0
    nb_new_kalman = 0

    if len(sso_night[sso_night["roid"] == 4]) != 0:
        nb_kalman_before = len(kalman_df)
        trajectory_df, kalman_df = kalman_association(
            trajectory_df, kalman_df, seeds, logger, True, confirmed_sso=True
        )
        nb_deviating_kalman = len(kalman_df) - nb_kalman_before

        # get the seeds with no associations to run the init_kalman on them.
        test_seeds = (
            seeds[seeds["trajectory_id"] != -1]
            .explode(["estimator_id", "ffdistnr"])
            .fillna(-1)
            .groupby("trajectory_id")
            .agg(
                tr_no_assoc=("estimator_id", lambda x: (np.array(x) == -1).all()),
                list_est=("estimator_id", list),
            )
            .reset_index()
        )
        test_seeds = test_seeds[test_seeds["tr_no_assoc"]]
        seeds_no_assoc = seeds[seeds["trajectory_id"].isin(test_seeds["trajectory_id"])]
        seeds_no_assoc = seeds_new_id(seeds_no_assoc, kalman_df)
        new_kalman_df = init_kalman(seeds_no_assoc)
        with pd.option_context("mode.chained_assignment", None):
            seeds_no_assoc["updated"] = "Y"
            new_kalman_df["orbfit_test"] = 0
        nb_new_kalman = len(new_kalman_df)

        trajectory_df = pd.concat([trajectory_df, seeds_no_assoc])
        kalman_df = pd.concat([kalman_df, new_kalman_df])
    else:
        if is_verbose:
            logger.info(
                "No kalman file detected, start to init kalman filters from the intra night seeds"
            )

        seeds = seeds[seeds["trajectory_id"] != -1.0]
        if len(kalman_df) > 0:
            seeds = seeds_new_id(seeds, kalman_df)
        new_kalman_df = init_kalman(seeds)

        with pd.option_context("mode.chained_assignment", None):
            seeds["updated"] = "Y"
            new_kalman_df["orbfit_test"] = 0
        nb_new_kalman = len(kalman_df)

        trajectory_df = pd.concat([trajectory_df, seeds])
        kalman_df = pd.concat([kalman_df, new_kalman_df])

    trajectory_df, kalman_df, trajectory_orb, orbits = kalman_to_orbit(
        config, trajectory_df, trajectory_orb, kalman_df, orbits, logger, True
    )

    trajectory_df, kalman_df, nb_removed_kalman = kalman_window_management(
        trajectory_df, kalman_df, config, current_datetime
    )

    if is_verbose:
        nb_kalman = len(kalman_df)
        nb_orbits = len(orbits)
        traj_kalman_size = Counter(
            trajectory_df["trajectory_id"].value_counts().sort_index()
        )
        traj_orbits_size = Counter(
            trajectory_orb["ssoCandId"].value_counts().sort_index()
        )
        logger.newline(2)
        logger.info(
            f"""
STATISTICS - ASSOCIATIONS
----------

number of orbits: {nb_orbits}
number of kalman: {nb_kalman}
* number of deviating kalman: {nb_deviating_kalman}
* number of new kalman: {nb_new_kalman}
* number of kalman removed by the time window: {nb_removed_kalman}

kalman trajectories size:
{Counter(traj_kalman_size)}

orbits trajectories size:
{Counter(traj_orbits_size)}
    """
        )
        logger.newline(2)

        logger.info("write the results")

    with pd.option_context("mode.chained_assignment", None):
        trajectory_df["trajectory_id"] = trajectory_df["trajectory_id"].astype(int)
        kalman_df["trajectory_id"] = kalman_df["trajectory_id"].astype(int)

    trajectory_df = trajectory_df.drop("updated", axis=1).explode(
        ["estimator_id", "ffdistnr"]
    )
    trajectory_df.to_parquet(path_trajectory_df, index=False)

    assert (kalman_df["trajectory_id"].value_counts() == 1).all()

    kalman_df.to_pickle(path_kalman)
    tr_orb_cols = trajectory_orb.columns
    if "updated" in tr_orb_cols:
        trajectory_orb = trajectory_orb.drop("updated", axis=1)
    if len(trajectory_orb) > 0:
        trajectory_orb = trajectory_orb.explode(["estimator_id", "ffdistnr"])
    trajectory_orb.to_parquet(path_trajectory_orb, index=False)
    orbits.to_parquet(path_orbit, index=False)

    if is_verbose:
        logger.info(f"{last_night} END PROCESSING")
        logger.newline()
