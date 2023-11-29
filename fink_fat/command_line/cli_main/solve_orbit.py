import numpy as np
import pandas as pd
import os
import time as t
import json

from fink_fat.command_line.utils_cli import get_class, yes_or_no, assig_tags

from fink_fat.command_line.association_cli import no_reset

from fink_fat.command_line.orbit_cli import (
    cluster_mode,
    get_orbital_data,
    intro_reset_orbit,
    yes_orbit_reset,
)

from fink_fat.orbit_fitting.orbfit_local import compute_df_orbit_param
from fink_fat.others.utils import init_logging


def cli_solve_orbit(arguments, config, output_path):
    """
    Solve the orbit for the current night using the command line

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
    output_path, object_class = get_class(arguments, output_path)
    tr_df_path = os.path.join(output_path, "trajectory_df.parquet")
    orb_res_path = os.path.join(output_path, "orbital.parquet")
    traj_orb_path = os.path.join(output_path, "trajectory_orb.parquet")

    if arguments["--reset"]:
        yes_or_no(
            intro_reset_orbit,
            yes_orbit_reset,
            no_reset,
            yes_args=(arguments, orb_res_path, traj_orb_path),
        )

    traj_to_orbital, traj_no_orb = get_orbital_data(config, tr_df_path)

    if len(traj_to_orbital) > 0:
        nb_traj_to_orbfit = len(np.unique(traj_to_orbital["trajectory_id"]))
        if arguments["--verbose"]:
            logger.info(
                "number of trajectories send to the orbit solver: {}".format(
                    nb_traj_to_orbfit
                )
            )

        # solve orbit in local mode
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
                verbose=arguments["--verbose"]
            ).drop("provisional designation", axis=1)
            orbfit_time = t.time() - t_before

            if arguments["--verbose"]:
                logger.info("time taken to get orbit: {}".format(orbfit_time))

        # solve orbit in cluster mode
        elif arguments["cluster"]:
            t_before = t.time()
            # return orbit results from cluster mode
            orbit_results = cluster_mode(config, traj_to_orbital)
            orbfit_time = t.time() - t_before

            if arguments["--verbose"]:
                logger.info("time taken to get orbit: {}".format(orbfit_time))

        nb_orb = 0
        # if new orbit has been computed
        if len(orbit_results) > 0:
            # get only the trajectories with orbital elements
            traj_with_orb_elem = orbit_results[orbit_results["a"] != -1.0]
            nb_orb = len(traj_with_orb_elem)

            if arguments["--verbose"]:
                logger.info("number of trajectories with orbit: {}".format(nb_orb))
                ratio_traj_to_orb = (nb_orb / nb_traj_to_orbfit) * 100
                logger.info("ratio: {0:.3f} %".format(ratio_traj_to_orb))

            # get the observations of trajectories with orbital elements
            obs_with_orb = traj_to_orbital[
                traj_to_orbital["trajectory_id"].isin(
                    traj_with_orb_elem["trajectory_id"]
                )
            ]

            if os.path.exists(orb_res_path):
                # if a save of orbit exists, append the new trajectories to it.
                orb_df = pd.read_parquet(orb_res_path)
                # if a save of orbit exist then a save of obs orbit necessarily exist
                traj_orb_df = pd.read_parquet(traj_orb_path)

                traj_with_orb_elem, obs_with_orb = assig_tags(
                    traj_with_orb_elem, obs_with_orb, len(orb_df)
                )

                orb_df = pd.concat([orb_df, traj_with_orb_elem])
                traj_orb_df = pd.concat([traj_orb_df, obs_with_orb])

                # traj_no_orb, orb_df, traj_orb_df = align_trajectory_id(
                #     traj_no_orb, orb_df, traj_orb_df
                # )

                orb_df.to_parquet(orb_res_path)
                traj_orb_df.to_parquet(traj_orb_path)

                # write the trajectory_df without the trajectories with more than orbfit_limit point
                traj_no_orb.to_parquet(tr_df_path)
            else:
                # traj_no_orb, traj_with_orb_elem, obs_with_orb = align_trajectory_id(
                #     traj_no_orb, traj_with_orb_elem, obs_with_orb
                # )

                # else we create the save of orbital elements and the associated observations
                traj_with_orb_elem, obs_with_orb = assig_tags(
                    traj_with_orb_elem, obs_with_orb, 0
                )
                traj_with_orb_elem.to_parquet(orb_res_path)
                obs_with_orb.to_parquet(traj_orb_path)

                # write the trajectory_df without the trajectories with more than orbfit_limit point
                traj_no_orb.to_parquet(tr_df_path)

            if arguments["--verbose"]:
                logger.info("Orbital elements saved")

        else:
            if arguments["--verbose"]:
                logger.info("No orbital elements found.")

        if arguments["--save"]:
            save_path = os.path.join(output_path, "save", "")
            stats_path = os.path.join(save_path, "stats.json")
            nb_traj_to_orbfit = len(np.unique(traj_to_orbital["trajectory_id"]))
            if os.path.exists(stats_path):
                with open(stats_path, "r+") as f:
                    stats_dict = json.load(f)
                    f.seek(0)
                    last_date = list(stats_dict.keys())[-1]
                    stats_dict[last_date]["nb_traj_to_orbfit"] = nb_traj_to_orbfit
                    stats_dict[last_date]["orbfit_time"] = orbfit_time
                    stats_dict[last_date]["nb_orb"] = nb_orb
                    json.dump(stats_dict, f, indent=4, sort_keys=True)
                    f.truncate()
            else:
                logger.info(
                    "No stats file exists. Run fink-fat in associations mode with the options --save to add it. "
                )

    else:
        logger.info("No trajectory with enough points to send to orbfit.")
        logger.info("Wait more night to produce trajectories with more points")
