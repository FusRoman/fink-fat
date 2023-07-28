import pandas as pd
import os
import time as t
from astropy.time import Time
import astropy.units as u

from fink_fat.command_line.utils_cli import get_class
from fink_fat.orbit_fitting.orbfit_merger import merge_orbit


def cli_merge_orbit(arguments, config, output_path):
    """
    Merge the orbit from the orbit parquet save by fink-fat using the command line.

    Parameters
    ----------
    arguments : dict
        command_line arguments
    config : ConfigParser
        object containing the data from the config file
    output_path : string
        path where are located the fink-fat data
    """
    output_path, object_class = get_class(arguments, output_path)
    orb_res_path = os.path.join(output_path, "orbital.parquet")
    traj_orb_path = os.path.join(output_path, "trajectory_orb.parquet")

    if os.path.exists(orb_res_path):
        # if a save of orbit exists, append the new trajectories to it.
        orb_df = pd.read_parquet(orb_res_path)
        # if a save of orbit exist then a save of obs orbit necessarily exist
        traj_orb_df = pd.read_parquet(traj_orb_path)

        if arguments["--verbose"]:
            print("Beginning of the merging !")

        t_before = t.time()
        # call the orbit identification that will merge trajectories
        merge_traj = merge_orbit(
            traj_orb_df,
            orb_df,
            config["SOLVE_ORBIT_PARAMS"]["ram_dir"],
            int(config["MERGE_ORBIT_PARAMS"]["neighbor"]),
            int(config["SOLVE_ORBIT_PARAMS"]["cpu_count"]),
            prop_epoch=(Time.now() + 2 * u.hour).jd,
        )

        merge_traj.to_parquet(os.path.join(output_path, "merge_traj.parquet"))

        if arguments["--verbose"]:
            print("Merging of the trajectories done !")
            print("elapsed time: {:.3f}".format(t.time() - t_before))

    else:
        print("No orbital elements found !")
        print("Abort merging !")
        exit()
