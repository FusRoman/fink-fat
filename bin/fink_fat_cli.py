"""
Usage:
    fink_fat associations (mpc | candidates) [--night <date> --save] [options]
    fink_fat solve_orbit (mpc | candidates) (local | cluster) [options]
    fink_fat offline (mpc | candidates) (local | cluster) <end> [<start> --save] [options]
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
  -r --reset                       Remove the file containing the trajectories candidates, the old observations and the orbits.
  -s --save                        Save the alerts sent by Fink for statistics.
  -h --help                        Show help and quit.
  --version                        Show version.
  --config FILE                    Specify the config file
  --output PATH                    Specify the out directory. A default path is set in the default fink_fat.conf
  --verbose                        Print information and progress bar during the process
"""

from collections import Counter
from collections import OrderedDict
from docopt import docopt
import os
import pandas as pd
import numpy as np
import time as t
import datetime
from astropy import units as u
from bin.offline_cli import offline_intro_reset, offline_yes_reset
from bin.orbit_cli import (
    cluster_mode,
    get_orbital_data,
    intro_reset_orbit,
    yes_orbit_reset,
)
from bin.utils_cli import get_class, init_cli, string_to_bool, yes_or_no

import fink_fat
from fink_fat.associations.inter_night_associations import night_to_night_association
from fink_fat.others.utils import cast_obs_data
from fink_fat.orbit_fitting.orbfit_local import compute_df_orbit_param
from bin.association_cli import (
    get_data,
    get_last_sso_alert,
    intro_reset,
    no_reset,
    yes_reset,
)


def main():

    # parse the command line and return options provided by the user.
    arguments = docopt(__doc__, version=fink_fat.__version__)

    config, output_path = init_cli(arguments)

    if arguments["associations"]:

        # get the path according to the class mpc or candidates
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

        if arguments["--save"]:
            save_path = os.path.join(output_path, "save", "")
            if not os.path.isdir(save_path):
                os.mkdir(save_path)
            new_alerts.to_parquet(os.path.join(save_path, "alert_{}".format(last_night)))

        if arguments["--verbose"]:
            print(
                "time taken to retrieve alerts from fink broker: {}".format(
                    t.time() - t_before
                )
            )
            print()

        if len(new_alerts) == 0:
            print("no alerts available for the night of {}".format(last_night))
            exit()

        trajectory_df, old_obs_df, last_nid, next_nid = get_data(
            new_alerts, tr_df_path, obs_df_path
        )

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

        if "last_assoc_date" in trajectory_df:
            trajectory_df["last_assoc_date"] = last_night
        else:
            trajectory_df.insert(
                len(trajectory_df.columns), "last_assoc_date", last_night
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
                t_before = t.time()

                # return orbit results from local mode
                orbit_results = compute_df_orbit_param(
                    traj_to_orbital,
                    int(config["SOLVE_ORBIT_PARAMS"]["cpu_count"]),
                    config["SOLVE_ORBIT_PARAMS"]["ram_dir"],
                ).drop("provisional designation", axis=1)

                if arguments["--verbose"]:
                    print("time taken to get orbit: {}".format(t.time() - t_before))

            elif arguments["cluster"]:
                t_before = t.time()

                # return orbit results from cluster mode
                orbit_results = cluster_mode(config, traj_to_orbital)

                if arguments["--verbose"]:
                    print("time taken to get orbit: {}".format(t.time() - t_before))

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
                "Number of observations, all trajectories candidates combined: {}".format(
                    len(trajectory_df)
                )
            )
            print(
                "Number of trajectories candidates: {}".format(
                    len(np.unique(trajectory_df["trajectory_id"]))
                )
            )
            gb = trajectory_df.groupby(["trajectory_id"]).count()["ra"]
            print("Trajectories size distribution:")
            c = Counter(gb)
            for size, number_size in OrderedDict(sorted(c.items())).items():
                print(
                    "\tsize: {}, number of trajectories candidates: {}".format(
                        size, number_size
                    )
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
            print("Number of detected orbit: {}".format(len(orb_df)))
        else:
            print("No trajectories with orbital elements found")

    elif arguments["offline"]:
        print("offline mode")

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
            "dcmag",
            "candid",
            "not_updated",
            "ssnamenr",
            "trajectory_id",
        ]
        trajectory_df = pd.DataFrame(columns=trajectory_columns)
        old_obs_df = pd.DataFrame(columns=trajectory_columns)

        delta_day = datetime.timedelta(days=1)

        # Default: begin the offline mode from the last night
        current_date = datetime.datetime.now() - delta_day

        # test if the trajectory_df and old_obs_df exists in the output directory.
        if os.path.exists(tr_df_path) and os.path.exists(obs_df_path):
            if arguments["<start>"] is not None:
                print("A save of trajectories candidates already exists.")
                print(
                    "Remove the <start> argument if you want to continue with the save"
                )
                print(
                    "or use the -r options to restart the associations from your <start> date."
                )
                print("Abort offline mode.")
                exit()

            trajectory_df = pd.read_parquet(tr_df_path)
            old_obs_df = pd.read_parquet(obs_df_path)

            # first case: trajectories already exists: begin the offline mode with the last associations date + 1
            current_date = datetime.datetime.strptime(
                trajectory_df["last_assoc_date"].values[0], "%Y-%m-%d"
            )
            current_date += delta_day

        # last case: <start> options given by the user, start the offline mode from this date.
        if arguments["<start>"] is not None:
            current_date = datetime.datetime.strptime(arguments["<start>"], "%Y-%m-%d")

        # stop date
        stop_date = datetime.datetime.strptime(arguments["<end>"], "%Y-%m-%d")

        # tomorrow
        today = datetime.datetime.now().date()

        if current_date.date() > stop_date.date():
            print("Error !!! Start date is greater than stop date.")
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

        while True:
            if arguments["--verbose"]:
                print("current processing date: {}".format(current_date))
                print()

            t_before = t.time()
            new_alerts = get_last_sso_alert(
                object_class, current_date.strftime("%Y-%m-%d"), arguments["--verbose"]
            )

            if arguments["--save"]:
                new_alerts.to_parquet(os.path.join(save_path, "alert_{}".format(current_date.strftime("%Y-%m-%d"))))

            # if no alerts are available
            if len(new_alerts) == 0:
                current_date += delta_day

                if current_date == stop_date + delta_day:
                    break
                if current_date.date() == today:
                    print(
                        "The current processing day is greater than today. Out of the offline loop."
                    )
                    break

                continue

            if arguments["--verbose"]:
                print(
                    "time taken to retrieve alerts from fink broker: {}".format(
                        t.time() - t_before
                    )
                )
                print()

            next_nid = new_alerts["nid"][0]
            last_nid = np.max([np.max(trajectory_df["nid"]), np.max(old_obs_df["nid"])])

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
                string_to_bool(
                    config["ASSOC_SYSTEM"]["tracklets_with_old_observations"]
                ),
                string_to_bool(
                    config["ASSOC_SYSTEM"]["new_observations_with_old_observations"]
                ),
                arguments["--verbose"],
            )

            # get trajectories with a number of points greater than the orbfit limit
            gb = trajectory_df.groupby(["trajectory_id"]).count().reset_index()
            traj = gb[gb["ra"] >= int(config["SOLVE_ORBIT_PARAMS"]["orbfit_limit"])][
                "trajectory_id"
            ]
            test_orb = trajectory_df["trajectory_id"].isin(traj)
            traj_to_orbital = trajectory_df[test_orb]
            trajectory_df = trajectory_df[~test_orb]

            if len(traj_to_orbital) > 0:

                if arguments["--verbose"]:
                    print()
                    print(
                        "Number of trajectories candidates send to solve orbit: {}".format(
                            len(np.unique(traj_to_orbital["trajectory_id"]))
                        )
                    )
                    print("Solve orbit...")

                if arguments["local"]:
                    t_before = t.time()

                    # return orbit results from local mode
                    orbit_results = compute_df_orbit_param(
                        traj_to_orbital,
                        int(config["SOLVE_ORBIT_PARAMS"]["cpu_count"]),
                        config["SOLVE_ORBIT_PARAMS"]["ram_dir"],
                    ).drop("provisional designation", axis=1)

                    if arguments["--verbose"]:
                        print("time taken to get orbit: {}".format(t.time() - t_before))

                elif arguments["cluster"]:
                    t_before = t.time()

                    # return orbit results from cluster mode
                    orbit_results = cluster_mode(config, traj_to_orbital)

                    if arguments["--verbose"]:
                        print("time taken to get orbit: {}".format(t.time() - t_before))

                if len(orbit_results) > 0:

                    # get only the trajectories with orbital elements
                    current_traj_with_orb_elem = orbit_results[
                        orbit_results["a"] != -1.0
                    ]

                    # get the observations of trajectories with orbital elements
                    current_obs_with_orb = traj_to_orbital[
                        traj_to_orbital["trajectory_id"].isin(
                            current_traj_with_orb_elem["trajectory_id"]
                        )
                    ]

                    orb_df = pd.concat([orb_df, current_traj_with_orb_elem])
                    traj_orb_df = pd.concat([traj_orb_df, current_obs_with_orb])

            current_date += delta_day

            if current_date == stop_date + delta_day:
                break
            if current_date.date() == today:
                print(
                    "The current processing day is greater than today. Out of the offline loop."
                )
                break

        # save the new data computed by the online mode
        cast_obs_data(trajectory_df).to_parquet(tr_df_path)
        cast_obs_data(old_obs_df).to_parquet(obs_df_path)

        orb_df.to_parquet(orb_res_path)
        traj_orb_df.to_parquet(traj_orb_path)

        print("Offline mode ended")

    else:
        exit()
