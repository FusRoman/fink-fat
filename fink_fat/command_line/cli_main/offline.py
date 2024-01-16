import numpy as np
import pandas as pd
import os
import json
import time as t
import datetime
from astropy import units as u
from fink_fat.command_line.offline_cli import offline_intro_reset, offline_yes_reset
from fink_fat.command_line.orbit_cli import cluster_mode
from fink_fat.command_line.utils_cli import (
    get_class,
    string_to_bool,
    yes_or_no,
    assig_tags,
)

from fink_fat.associations.inter_night_associations import night_to_night_association
from fink_fat.others.utils import cast_obs_data
from fink_fat.orbit_fitting.orbfit_local import compute_df_orbit_param
from fink_fat.command_line.association_cli import (
    get_last_sso_alert,
    no_reset,
)
from fink_fat.others.utils import init_logging


def cli_offline(arguments, config, output_path):
    """
    Launch fink-fat in offline mode using the command line

    Parameters
    ----------
    arguments : dict
        command_line arguments
    config : ConfigParser
        object containing the data from the config file
    output_path : string
        path where are located the fink-fat data
    """
    logger = init_logging()
    logger.info("offline mode")

    output_path, object_class = get_class(arguments, output_path)

    # path to the associations data
    tr_df_path = os.path.join(output_path, "trajectory_df.parquet")
    obs_df_path = os.path.join(output_path, "old_obs.parquet")

    # path to the orbit data
    orb_res_path = os.path.join(output_path, "orbital.parquet")
    traj_orb_path = os.path.join(output_path, "trajectory_orb.parquet")

    # remove the save data from previous associations if the user say yes
    if arguments["--reset"]:
        yes_or_no(
            offline_intro_reset,
            offline_yes_reset,
            no_reset,
            yes_args=(
                arguments,
                tr_df_path,
                obs_df_path,
                orb_res_path,
                traj_orb_path,
            ),
        )

    trajectory_columns = [
        "ra",
        "dec",
        "jd",
        "nid",
        "fid",
        "magpsf",
        "sigmapsf",
        "candid",
        "not_updated",
        "ssnamenr",
        "trajectory_id",
        "last_assoc_date",
    ]
    trajectory_df = pd.DataFrame(columns=trajectory_columns)
    old_obs_df = pd.DataFrame(columns=trajectory_columns)

    delta_day = datetime.timedelta(days=1)

    # Default: begin the offline mode from the last night
    current_date = datetime.datetime.now() - delta_day

    # test if the trajectory_df and old_obs_df exists in the output directory.
    if os.path.exists(tr_df_path) and os.path.exists(obs_df_path):
        if arguments["<start>"] is not None:
            logger.info("A save of trajectories candidates already exists.")
            logger.info(
                "Remove the <start> argument if you want to continue with the save"
            )
            logger.info(
                "or use the -r options to restart the associations from your <start> date."
            )
            logger.info("Abort offline mode.")
            exit()

        trajectory_df = pd.read_parquet(tr_df_path)
        old_obs_df = pd.read_parquet(obs_df_path)

        # first case: trajectories already exists: begin the offline mode with the last associations date + 1
        last_tr_date = pd.to_datetime(
            trajectory_df["last_assoc_date"], format="%Y-%m-%d"
        )

        last_obs_date = pd.to_datetime(old_obs_df["last_assoc_date"], format="%Y-%m-%d")

        current_date = max(last_tr_date.max(), last_obs_date.max())
        current_date += delta_day

    # last case: <start> options given by the user, start the offline mode from this date.
    if arguments["<start>"] is not None:
        current_date = datetime.datetime.strptime(arguments["<start>"], "%Y-%m-%d")

    # stop date
    stop_date = datetime.datetime.strptime(arguments["<end>"], "%Y-%m-%d")

    # tomorrow
    today = datetime.datetime.now().date()

    if current_date.date() > stop_date.date():
        logger.info("Error !!! Start date is greater than stop date.")
        exit()

    orb_df = pd.DataFrame()
    traj_orb_df = pd.DataFrame()

    # load the orbit data if already exists
    if os.path.exists(orb_res_path):
        orb_df = pd.read_parquet(orb_res_path)

    if os.path.exists(traj_orb_path):
        traj_orb_df = pd.read_parquet(traj_orb_path)

    if arguments["--save"]:
        save_path = os.path.join(output_path, "save", "")
        if not os.path.isdir(save_path):
            os.mkdir(save_path)

    stats_dict = {}

    while True:
        if arguments["--verbose"]:
            logger.info("current processing date: {}".format(current_date))
            logger.newline()

        t_before = t.time()
        new_alerts = get_last_sso_alert(
            object_class, current_date.strftime("%Y-%m-%d"), arguments["--verbose"]
        )

        if arguments["--verbose"]:
            logger.info(
                "Number of alerts retrieve from fink: {}".format(len(new_alerts))
            )

        if arguments["--save"] and object_class == "Solar System MPC":
            if len(new_alerts) > 0:
                new_alerts.to_parquet(
                    os.path.join(
                        save_path,
                        "alert_{}".format(current_date.strftime("%Y-%m-%d")),
                    )
                )

        # if no alerts are available
        if len(new_alerts) == 0:
            current_date += delta_day

            if current_date == stop_date + delta_day:
                break
            if current_date.date() == today:
                logger.info(
                    "The current processing day is greater than today. Out of the offline loop."
                )
                break

            continue

        if arguments["--verbose"]:
            logger.info(
                "time taken to retrieve alerts from fink broker: {}".format(
                    t.time() - t_before
                )
            )
            logger.newline()

        next_nid = new_alerts["nid"][0]
        last_nid = np.max([np.max(trajectory_df["nid"]), np.max(old_obs_df["nid"])])

        # get the last trajectory_id as baseline for new trajectories
        last_trajectory_id = 0
        if len(trajectory_df) > 0:
            last_trajectory_id = np.max(trajectory_df["trajectory_id"])

        nb_traj = len(np.unique(trajectory_df["trajectory_id"]))
        nb_old_obs = len(old_obs_df)
        nb_new_alerts = len(new_alerts)
        t_before = t.time()

        trajectory_df, old_obs_df = night_to_night_association(
            trajectory_df,
            old_obs_df,
            new_alerts,
            last_trajectory_id + 1,
            last_nid,
            next_nid,
            int(config["TW_PARAMS"]["trajectory_keep_limit"]),
            int(config["TW_PARAMS"]["old_observation_keep_limit"]),
            int(config["TW_PARAMS"]["trajectory_2_points_keep_limit"]),
            float(config["ASSOC_PARAMS"]["intra_night_separation"]) * u.arcsecond,
            float(config["ASSOC_PARAMS"]["intra_night_magdiff_limit_same_fid"]),
            float(config["ASSOC_PARAMS"]["intra_night_magdiff_limit_diff_fid"]),
            string_to_bool(config["ASSOC_PARAMS"]["use_dbscan"]),
            float(config["ASSOC_PARAMS"]["inter_night_separation"]) * u.degree,
            float(config["ASSOC_PARAMS"]["inter_night_magdiff_limit_same_fid"]),
            float(config["ASSOC_PARAMS"]["inter_night_magdiff_limit_diff_fid"]),
            float(config["ASSOC_PARAMS"]["maximum_angle"]),
            string_to_bool(config["ASSOC_PERF"]["store_kd_tree"]),
            int(config["SOLVE_ORBIT_PARAMS"]["orbfit_limit"]),
            string_to_bool(config["ASSOC_SYSTEM"]["tracklets_with_trajectories"]),
            string_to_bool(
                config["ASSOC_SYSTEM"]["trajectories_with_new_observations"]
            ),
            string_to_bool(config["ASSOC_SYSTEM"]["tracklets_with_old_observations"]),
            string_to_bool(
                config["ASSOC_SYSTEM"]["new_observations_with_old_observations"]
            ),
            arguments["--verbose"],
        )
        assoc_time = t.time() - t_before

        trajectory_df = cast_obs_data(trajectory_df)
        old_obs_df = cast_obs_data(old_obs_df)

        # get trajectories with a number of points greater than the orbfit limit
        gb = trajectory_df.groupby(["trajectory_id"]).count().reset_index()
        traj = gb[gb["ra"] >= int(config["SOLVE_ORBIT_PARAMS"]["orbfit_limit"])][
            "trajectory_id"
        ]
        test_orb = trajectory_df["trajectory_id"].isin(traj)
        traj_to_orbital = trajectory_df[test_orb]
        trajectory_df = trajectory_df[~test_orb]

        # orbfit stats
        nb_traj_to_orbfit = len(np.unique(traj_to_orbital["trajectory_id"]))
        orbfit_time = 0.0
        nb_orb = 0

        if len(traj_to_orbital) > 0:
            if arguments["--verbose"]:
                logger.newline()
                logger.info(
                    "Number of trajectories candidates send to solve orbit: {}".format(
                        nb_traj_to_orbfit
                    )
                )
                logger.info("Solve orbit...")

            if arguments["local"]:
                t_before = t.time()
                prop_epoch = config["SOLVE_ORBIT_PARAMS"]["prop_epoch"]
                # return orbit results from local mode
                orbit_results = compute_df_orbit_param(
                    traj_to_orbital,
                    int(config["SOLVE_ORBIT_PARAMS"]["cpu_count"]),
                    config["SOLVE_ORBIT_PARAMS"]["ram_dir"],
                    int(config["SOLVE_ORBIT_PARAMS"]["n_triplets"]),
                    int(config["SOLVE_ORBIT_PARAMS"]["noise_ntrials"]),
                    prop_epoch=float(prop_epoch) if prop_epoch != "None" else None,
                    verbose_orbfit=int(config["SOLVE_ORBIT_PARAMS"]["orbfit_verbose"]),
                    verbose=arguments["--verbose"],
                ).drop("provisional designation", axis=1)
                orbfit_time = t.time() - t_before

                if arguments["--verbose"]:
                    logger.info("time taken to get orbit: {}".format(orbfit_time))

            elif arguments["cluster"]:
                t_before = t.time()
                # return orbit results from cluster mode
                orbit_results = cluster_mode(config, traj_to_orbital)
                orbfit_time = t.time() - t_before

                if arguments["--verbose"]:
                    logger.info("time taken to get orbit: {}".format(orbfit_time))

            if len(orbit_results) > 0:
                # get only the trajectories with orbital elements
                # the other one are discards
                current_traj_with_orb_elem = orbit_results[orbit_results["a"] != -1.0]
                nb_orb = len(current_traj_with_orb_elem)

                if arguments["--verbose"]:
                    logger.info("number of trajectories with orbit: {}".format(nb_orb))
                    ratio_traj_to_orb = (nb_orb / nb_traj_to_orbfit) * 100
                    logger.info("ratio: {0:.3f} %".format(ratio_traj_to_orb))

                # get the observations of trajectories with orbital elements
                current_obs_with_orb = traj_to_orbital[
                    traj_to_orbital["trajectory_id"].isin(
                        current_traj_with_orb_elem["trajectory_id"]
                    )
                ]

                current_traj_with_orb_elem, current_obs_with_orb = assig_tags(
                    current_traj_with_orb_elem, current_obs_with_orb, len(orb_df)
                )

                orb_df = pd.concat([orb_df, current_traj_with_orb_elem])
                traj_orb_df = pd.concat([traj_orb_df, current_obs_with_orb])

                # (trajectory_df, orb_df, traj_orb_df,) = align_trajectory_id(
                #     trajectory_df, orb_df, traj_orb_df
                # )
            else:
                if arguments["--verbose"]:
                    logger.info("No orbit found")

        stats_dict[current_date.strftime("%Y-%m-%d")] = {
            "assoc_time": assoc_time,
            "nb_traj": nb_traj,
            "nb_old_obs": nb_old_obs,
            "nb_new_alerts": nb_new_alerts,
            "nb_traj_to_orbfit": nb_traj_to_orbfit,
            "orbfit_time": orbfit_time,
            "nb_orb": nb_orb,
        }

        current_date += delta_day

        if current_date == stop_date + delta_day:
            break
        if current_date.date() == today:
            logger.info(
                "The current processing day is greater than today. Out of the offline loop."
            )
            break

    # if "last_assoc_date" in trajectory_df:
    #     trajectory_df["last_assoc_date"] = current_date
    # else:
    #     trajectory_df.insert(
    #         len(trajectory_df.columns), "last_assoc_date", current_date
    #     )

    # save the new data computed by the online mode
    cast_obs_data(trajectory_df).to_parquet(tr_df_path)
    cast_obs_data(old_obs_df).to_parquet(obs_df_path)

    orb_df.to_parquet(orb_res_path)
    traj_orb_df.to_parquet(traj_orb_path)

    if arguments["--save"]:
        save_path = os.path.join(output_path, "save", "")
        stats_path = os.path.join(save_path, "stats.json")
        with open(stats_path, "w") as f:
            json.dump(stats_dict, f, indent=4, sort_keys=True)

    logger.info("Offline mode ended")
