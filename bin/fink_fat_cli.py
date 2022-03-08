"""
Usage:
    fink_fat associations (mpc | candidates) [--night <date>] [options]
    fink_fat solve_orbit (mpc | candidates) (local | cluster) [options]
    fink_fat stats (mpc | candidates) [options]
    fink_fat -h | --help
    fink_fat --version

Options:
  associations                     Perform associations of alert to return a set of trajectories candidates.
  solve_orbit                      Resolve a dynamical inverse problem to return a set of orbital elements from
                                   the set of trajectories candidates.
  stats                            Print statistics about trajectories detected by assocations, the old observations
                                   and, if exists, the orbital elements for some trajectories.
  mpc                              Return the associations on the solar system mpc alerts (only for tests purpose).
  candidates                       Run the associations on the solar system candidates alerts.
  local                            Run the orbital solver in local mode. Use multiprocessing to speed-up the computation.
  cluster                          Run the orbital solver in cluster mode. Use a Spark cluster to significantly speed-up the computation.
                                   The cluster mode need to be launch on a system where pyspark are installed and a spark cluster manager are setup.
  -n <date> --night <date>         Specify the night to request sso alerts from fink broker.
                                   Format is yyyy-mm-dd as yyyy = year, mm = month, dd = day.
                                   Example : 2022-03-04 for the 2022 march 04.
                                   [intervall of day between the day starting at night midday until night midday + 1]
  -r --reset                       Remove the file containing the trajectories and the old observations.
  -h --help                        Show help and quit.
  --version                        Show version.
  --config FILE                    Specify the config file
  --output PATH                    Specify the out directory. A default path is set in the default fink_fat.conf
  --verbose                        Print information and progress bar during the process
"""

from collections import Counter
from collections import OrderedDict
import subprocess
from docopt import docopt
import os
import pandas as pd
import numpy as np
import time as t
import datetime
from astropy import units as u
from bin.orbit_cli import cluster_mode, get_orbital_data, intro_reset_orbit, yes_orbit_reset
from bin.utils_cli import get_class, init_cli, string_to_bool, yes_or_no

import fink_fat
from fink_fat.associations.inter_night_associations import night_to_night_association
from fink_fat.others.utils import cast_obs_data
from fink_fat.orbit_fitting.orbfit_local import compute_df_orbit_param
from bin.association_cli import get_data, get_last_sso_alert, intro_reset, no_reset, yes_reset


def main():

    # parse the command line and return options provided by the user.
    arguments = docopt(__doc__, version=fink_fat.__version__)

    config, output_path = init_cli(arguments)

    if arguments["associations"]:

        output_path, object_class = get_class(arguments, output_path)

        tr_df_path = os.path.join(output_path, "trajectory_df.parquet")
        obs_df_path = os.path.join(output_path, "old_obs.parquet")

        # remove the save data from previous associations if the user say yes
        if arguments["--reset"]:
            yes_or_no(
                intro_reset,
                yes_reset,
                no_reset,
                yes_args=(arguments, tr_df_path, obs_df_path),
            )

        last_night = datetime.datetime.now() - datetime.timedelta(days=1)
        last_night = last_night.strftime("%Y-%m-%d")
        if arguments["--night"]:
            last_night = arguments["--night"]

        t_before = t.time()
        new_alerts = get_last_sso_alert(
            object_class, last_night, arguments["--verbose"]
        )
        if arguments["--verbose"]:
            print("time taken to retrieve alerts from fink broker: {}".format(t.time() - t_before))
            print()

        if len(new_alerts) == 0:
            print("no alerts available for the night of {}".format(last_night))
            exit()

        trajectory_df, old_obs_df, last_nid, next_nid = get_data(new_alerts, tr_df_path, obs_df_path)

        if arguments["--verbose"]:
            print("started associations...")

        trajectory_df, old_obs_df, _ = night_to_night_association(
            trajectory_df,
            old_obs_df,
            new_alerts,
            last_nid,
            next_nid,
            int(config["TW_PARAMS"]["trajectory_keep_limit"]),
            int(config["TW_PARAMS"]["old_observation_keep_limit"]),
            int(config["TW_PARAMS"]["trajectory_2_points_keep_limit"]),
            float(config["ASSOC_PARAMS"]["intra_night_separation"]) * u.arcsecond,
            float(config["ASSOC_PARAMS"]["intra_night_magdiff_limit_same_fid"]),
            float(config["ASSOC_PARAMS"]["intra_night_magdiff_limit_diff_fid"]),
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

        cast_obs_data(trajectory_df).to_parquet(tr_df_path)
        cast_obs_data(old_obs_df).to_parquet(obs_df_path)

        if arguments["--verbose"]:
            print("Association done")

    elif arguments["solve_orbit"]:

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

            if arguments["local"]:
                # return orbit results from local mode
                orbit_results = compute_df_orbit_param(
                    traj_to_orbital,
                    int(config["SOLVE_ORBIT_PARAMS"]["cpu_count"]),
                    config["SOLVE_ORBIT_PARAMS"]["ram_dir"],
                ).drop("provisional designation", axis=1)

            elif arguments["cluster"]:
                # return orbit results from cluster mode
                orbit_results = cluster_mode(config, traj_to_orbital)

            if len(orbit_results) > 0:

                # write the trajectory_df without the trajectories with more than orbfit_limit point
                # delay the writing of trajectory_df in case of orbfit fail.
                traj_no_orb.to_parquet(tr_df_path)

                # get only the trajectories with orbital elements
                traj_with_orb_elem = orbit_results[orbit_results["a"] != -1.0]

                # get the observations of trajectories with orbital elements
                obs_with_orb = traj_to_orbital[
                    traj_to_orbital["trajectory_id"].isin(
                        traj_with_orb_elem["trajectory_id"]
                    )
                ]

                if os.path.exists(orb_res_path):
                    # if a save of orbit exists, append the new trajectories to it.
                    orb_df = pd.read_parquet(orb_res_path)
                    orb_df = pd.concat([orb_df, traj_with_orb_elem])
                    orb_df.to_parquet(orb_res_path)
                else:
                    # else we create the save of orbital elements
                    orbit_results.to_parquet(orb_res_path)

                if os.path.exists(traj_orb_path):
                    # we save the observations of trajectories with orbital elements
                    traj_orb_df = pd.read_parquet(traj_orb_path)
                    traj_orb_df = pd.concat([traj_orb_df, obs_with_orb])
                    traj_orb_df.to_parquet(traj_orb_path)
                else:
                    obs_with_orb.to_parquet(traj_orb_path)

                if arguments["--verbose"]:
                    print("Orbital elements saved")

            else:
                if arguments["--verbose"]:
                    print("No orbital elements found.")

        else:
            print("No trajectory with enough points to send to orbfit.")
            print("Wait more night to produce trajectories with more points")

    elif arguments["stats"]:

        output_path, object_class = get_class(arguments, output_path)
        tr_df_path = os.path.join(output_path, "trajectory_df.parquet")
        orb_res_path = os.path.join(output_path, "orbital.parquet")
        obs_df_path = os.path.join(output_path, "old_obs.parquet")

        if os.path.exists(tr_df_path):
            trajectory_df = pd.read_parquet(tr_df_path)
            print(
                "Number of observations, all trajectories combined: {}".format(
                    len(trajectory_df)
                )
            )
            print(
                "Number of trajectories detected: {}".format(
                    len(np.unique(trajectory_df["trajectory_id"]))
                )
            )
            gb = trajectory_df.groupby(["trajectory_id"]).count()["ra"]
            print("Trajectories size distribution:")
            c = Counter(gb)
            for size, number_size in OrderedDict(sorted(c.items())).items():
                print(
                    "\tsize: {}, number of trajectories: {}".format(size, number_size)
                )
            print()
        else:
            print(
                "Trajectory file doesn't exist, run 'fink_fat association (mpc | candidates)' to create it."
            )

        if os.path.exists(obs_df_path):
            old_obs_df = pd.read_parquet(obs_df_path)
            print("Number of old observations: {}".format(len(old_obs_df)))
            print()
        else:
            print("No old observations exists.")

        if os.path.exists(orb_res_path):
            orb_df = pd.read_parquet(orb_res_path)
            print(
                "number of trajectories with orbital elements: {}".format(len(orb_df))
            )
        else:
            print("No trajectories with orbital elements found")

    else:
        exit()
