"""
Usage:
    fink_fat associations (mpc | candidates) [--night <date>] [options]
    fink_fat solve_orbit (mpc | candidates) (local | cluster) [options]
    fink_fat merge_orbit (mpc | candidates) [options]
    fink_fat offline (mpc | candidates) (local | cluster) <end> [<start>] [options]
    fink_fat stats (mpc | candidates) [--mpc-data <path>] [options]
    fink_fat -h | --help
    fink_fat --version

Options:
  associations                     Perform associations of alert to return a set of trajectories candidates.
  solve_orbit                      Resolve a dynamical inverse problem to return a set of orbital elements from
                                   the set of trajectories candidates.
  merge_orbit                      Merge the orbit candidates if the both trajectories can belong to the same solar system objects.
  offline                          Associate the alerts to form trajectories candidates then solve the orbit
                                   until the end parameters. Starts from saved data or from the start parameters
                                   if provided.
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
  -m <path> --mpc-data <path>      Compute statistics according to the minor planet center database.
                                   <path> of the mpc database file.
                                   The mpc database can be downloaded by pasting this url in your browser: https://minorplanetcenter.net/Extended_Files/mpcorb_extended.json.gz
  -r --reset                       Remove the file containing the trajectories candidates, the old observations and the orbits.
  -s --save                        Save the alerts sent by Fink before the associations for statistics purposes.
                                   Save also additional statistics : computation time, number of alerts from the current days, number of candidates trajectories, number of old observations.
  -h --help                        Show help and quit.
  --version                        Show version.
  --config FILE                    Specify the config file
  --verbose                        Print information and progress bar during the process
"""

from collections import Counter
from collections import OrderedDict
import json
from docopt import docopt
import os
import pandas as pd
import numpy as np
import time as t
import datetime
import glob
from astropy import units as u
from terminaltables import DoubleTable, AsciiTable, SingleTable
from bin.offline_cli import offline_intro_reset, offline_yes_reset
from bin.orbit_cli import (
    cluster_mode,
    get_orbital_data,
    intro_reset_orbit,
    yes_orbit_reset,
)
from bin.stat_cli import compute_residue, test_detectable
from bin.utils_cli import (
    get_class,
    init_cli,
    string_to_bool,
    yes_or_no,
    align_trajectory_id,
    save_additional_stats,
)

import fink_fat
from fink_fat.associations.inter_night_associations import night_to_night_association
from fink_fat.others.utils import cast_obs_data
from fink_fat.orbit_fitting.orbfit_local import compute_df_orbit_param
from fink_fat.orbit_fitting.orbfit_merger import orbit_identification
from bin.association_cli import (
    get_data,
    get_last_sso_alert,
    intro_reset,
    no_reset,
    yes_reset,
)
from bin.stat_cli import print_assoc_table


def main():

    # parse the command line and return options provided by the user.
    arguments = docopt(__doc__, version=fink_fat.__version__)

    config, output_path = init_cli(arguments)

    if arguments["associations"]:

        # get the path according to the class mpc or candidates
        output_path, object_class = get_class(arguments, output_path)

        tr_df_path = os.path.join(output_path, "trajectory_df.parquet")
        obs_df_path = os.path.join(output_path, "old_obs.parquet")

        # get the path of the orbit database to compute properly the trajectory_id baseline.
        orb_res_path = os.path.join(output_path, "orbital.parquet")

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
            print("Number of alerts retrieve from fink: {}".format(len(new_alerts)))

        if arguments["--save"]:
            save_path = os.path.join(output_path, "save", "")
            if not os.path.isdir(save_path):
                os.mkdir(save_path)
            if len(new_alerts) > 0:
                new_alerts.to_parquet(
                    os.path.join(save_path, "alert_{}".format(last_night))
                )

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

        trajectory_df, old_obs_df, last_nid, next_nid, last_trajectory_id = get_data(
            new_alerts, tr_df_path, obs_df_path, orb_res_path
        )

        if arguments["--verbose"]:
            print("started associations...")

        # for additional statistics
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
        new_stats = {
            "assoc_time": assoc_time,
            "nb_traj": nb_traj,
            "nb_old_obs": nb_old_obs,
            "nb_new_alerts": nb_new_alerts,
        }

        if arguments["--save"]:
            save_additional_stats(
                os.path.join(save_path, "stats.json"), last_night, new_stats
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
                orbfit_time = t.time() - t_before

                if arguments["--verbose"]:
                    print("time taken to get orbit: {}".format(orbfit_time))

            elif arguments["cluster"]:

                t_before = t.time()
                # return orbit results from cluster mode
                orbit_results = cluster_mode(config, traj_to_orbital)
                orbfit_time = t.time() - t_before

                if arguments["--verbose"]:
                    print("time taken to get orbit: {}".format(orbfit_time))

            nb_orb = 0
            if len(orbit_results) > 0:
                # get only the trajectories with orbital elements
                traj_with_orb_elem = orbit_results[orbit_results["a"] != -1.0]
                nb_orb = len(traj_with_orb_elem)

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

                    orb_df = pd.concat([orb_df, traj_with_orb_elem])
                    traj_orb_df = pd.concat([traj_orb_df, obs_with_orb])

                    traj_no_orb, orb_df, traj_orb_df = align_trajectory_id(
                        traj_no_orb, orb_df, traj_orb_df
                    )

                    orb_df.to_parquet(orb_res_path)
                    traj_orb_df.to_parquet(traj_orb_path)

                    # write the trajectory_df without the trajectories with more than orbfit_limit point
                    traj_no_orb.to_parquet(tr_df_path)
                else:

                    traj_no_orb, traj_with_orb_elem, obs_with_orb = align_trajectory_id(
                        traj_no_orb, traj_with_orb_elem, obs_with_orb
                    )

                    # else we create the save of orbital elements and the associated observations
                    traj_with_orb_elem.to_parquet(orb_res_path)
                    obs_with_orb.to_parquet(traj_orb_path)

                    # write the trajectory_df without the trajectories with more than orbfit_limit point
                    traj_no_orb.to_parquet(tr_df_path)

                if arguments["--verbose"]:
                    print("Orbital elements saved")

            else:
                if arguments["--verbose"]:
                    print("No orbital elements found.")

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
                    print(
                        "No stats file exists. Run fink-fat in associations mode with the options --save to add it. "
                    )

        else:
            print("No trajectory with enough points to send to orbfit.")
            print("Wait more night to produce trajectories with more points")

    elif arguments["merge_orbit"]:

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
            merge_traj_orb, merger_orb_df = orbit_identification(
                traj_orb_df,
                orb_df,
                config["SOLVE_ORBIT_PARAMS"]["ram_dir"],
                int(config["MERGE_ORBIT_PARAMS"]["neighbor"]),
                int(config["SOLVE_ORBIT_PARAMS"]["cpu_count"]),
            )

            merge_traj_orb.to_parquet(traj_orb_path)
            merger_orb_df.to_parquet(orb_res_path)

            if arguments["--verbose"]:
                print("Merging of the trajectories done !")
                print("elapsed time: {:.3f}".format(t.time() - t_before))

        else:
            print("No orbital elements found !")
            print("Abort merging !")
            exit()

    elif arguments["stats"]:

        output_path, object_class = get_class(arguments, output_path)
        tr_df_path = os.path.join(output_path, "trajectory_df.parquet")
        orb_res_path = os.path.join(output_path, "orbital.parquet")
        obs_df_path = os.path.join(output_path, "old_obs.parquet")
        traj_orb_path = os.path.join(output_path, "trajectory_orb.parquet")

        if os.path.exists(tr_df_path):
            trajectory_df = pd.read_parquet(tr_df_path)

            if len(trajectory_df) == 0:
                print("No trajectories detected.")
                exit()

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
            c = Counter(gb)
            table_data = [["Size", "Number of trajectories candidates"]]
            table_data += [
                [size, number_size]
                for size, number_size in OrderedDict(sorted(c.items())).items()
            ]
            table_instance = AsciiTable(
                table_data, "Trajectories candidates size distribution"
            )
            table_instance.justify_columns[1] = "right"
            print()
            print(table_instance.table)
            print()

            print_assoc_table(trajectory_df)

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

        if os.path.exists(orb_res_path) and os.path.exists(traj_orb_path):
            orb_df = pd.read_parquet(orb_res_path)
            traj_orb_df = pd.read_parquet(traj_orb_path)

            if len(orb_df) == 0 or len(traj_orb_df) == 0:
                print("No trajectories with orbital elements found.")
                exit()

        else:
            print("No trajectories with orbital elements found")
            exit()

        # trajectories with orbits size comparation
        trajectories_gb = traj_orb_df.groupby(["trajectory_id"]).agg(
            count=("ra", len), tags=("assoc_tag", list)
        )

        trajectories_size = Counter(trajectories_gb["count"])
        table_data = [["Size", "Number of orbits candidates"]]
        table_data += [
            [size, number_size]
            for size, number_size in OrderedDict(
                sorted(trajectories_size.items())
            ).items()
        ]
        table_instance = AsciiTable(table_data, "Orbits candidates size distribution")
        table_instance.justify_columns[1] = "right"
        print()
        print(table_instance.table)
        print()

        print_assoc_table(traj_orb_df)

        # orbital type statistics
        orb_stats = (
            orb_df[["a", "e", "i", "long. node", "arg. peric", "mean anomaly"]]
            .describe()
            .round(decimals=3)
        )

        print("Number of orbit candidates: {}".format(int(orb_stats["a"]["count"])))

        orbit_distrib_data = (
            ("orbital elements", "Metrics", "Values"),
            ("semi-major-axis (AU)", "mean", orb_stats["a"]["mean"]),
            ("", "std", orb_stats["a"]["std"]),
            ("", "min", orb_stats["a"]["min"]),
            ("", "max", orb_stats["a"]["max"]),
            ("eccentricity", "mean", orb_stats["e"]["mean"]),
            ("", "std", orb_stats["e"]["std"]),
            ("", "min", orb_stats["e"]["min"]),
            ("", "max", orb_stats["e"]["max"]),
            ("inclination (degrees)", "mean", orb_stats["i"]["mean"]),
            ("", "std", orb_stats["i"]["std"]),
            ("", "min", orb_stats["i"]["min"]),
            ("", "max", orb_stats["i"]["max"]),
            ("long. node (degrees)", "mean", orb_stats["long. node"]["mean"]),
            ("", "std", orb_stats["long. node"]["std"]),
            ("", "min", orb_stats["long. node"]["min"]),
            ("", "max", orb_stats["long. node"]["max"]),
            ("arg. peric (degrees)", "mean", orb_stats["arg. peric"]["mean"]),
            ("", "std", orb_stats["arg. peric"]["std"]),
            ("", "min", orb_stats["arg. peric"]["min"]),
            ("", "max", orb_stats["arg. peric"]["max"]),
            ("mean anomaly (degrees)", "mean", orb_stats["mean anomaly"]["mean"]),
            ("", "std", orb_stats["mean anomaly"]["std"]),
            ("", "min", orb_stats["mean anomaly"]["min"]),
            ("", "max", orb_stats["mean anomaly"]["max"]),
        )

        orb_table = SingleTable(orbit_distrib_data, "orbit candidates distribution")
        print()
        print(orb_table.table)
        print()

        main_belt_candidates = orb_df[(orb_df["a"] <= 4.5) & (orb_df["a"] >= 1.7)]
        distant_main_belt = orb_df[orb_df["a"] > 4.5]
        close_asteroids = orb_df[orb_df["a"] < 1.7]
        earth_crosser = close_asteroids[
            (close_asteroids["a"] < 1.7) & (close_asteroids["e"] > 0.1)
        ]
        no_earth_crosser = close_asteroids[
            (close_asteroids["a"] < 1.7) & (close_asteroids["e"] <= 0.1)
        ]

        orbit_type_data = (
            ("Orbit type", "Number of candidates", "Notes"),
            (
                "Main belt",
                len(main_belt_candidates),
                "Main belt asteroids are asteroids with a semi major axis between 1.7 AU and 4.5 AU",
            ),
            (
                "Distant",
                len(distant_main_belt),
                "Distant asteroids are asteroids with a semi major axis greater than 4.5 AU",
            ),
            (
                "Earth crosser",
                len(earth_crosser),
                "An asteroids is considered as an earth crosser when his semi major axis is less than 1.7 and his eccentricity is greater than 0.1",
            ),
            (
                "No earth crosser",
                len(no_earth_crosser),
                "Asteroids with a semi major axis less than 1.7 and an eccentricity less than 0.1",
            ),
        )
        orb_type_table = SingleTable(orbit_type_data, "orbit candidates type")
        print()
        print(orb_type_table.table)
        print()

        # subtraction with the mean of each rms computed with
        # the trajectories from MPC.
        orb_df["rms_dist"] = np.linalg.norm(
            orb_df[
                [
                    "rms_a",
                    "rms_e",
                    "rms_i",
                    "rms_long. node",
                    "rms_arg. peric",
                    "rms_mean anomaly",
                ]
            ].values
            - [0.018712, 0.009554, 0.170369, 0.383595, 4.314636, 3.791175],
            axis=1,
        )

        orb_df["chi_dist"] = np.abs(orb_df["chi_reduced"].values - 1)

        orb_df["score"] = np.linalg.norm(
            orb_df[["rms_dist", "chi_dist"]].values - [0, 0], axis=1
        )

        orb_df = orb_df.sort_values(["score"]).reset_index(drop=True)
        best_orb = orb_df.loc[:9]

        best_orbit_data = (
            [
                [
                    "Trajectory id",
                    "Orbit ref epoch",
                    "a (AU)",
                    "error",
                    "e",
                    "error",
                    "i (deg)",
                    "error",
                    "Long. node (deg)",
                    "error",
                    "Arg. peri (deg)",
                    "error",
                    "Mean Anomaly (deg)",
                    "error",
                    "chi",
                    "score",
                ],
            ]
            + np.around(
                best_orb[
                    [
                        "trajectory_id",
                        "ref_epoch",
                        "a",
                        "rms_a",
                        "e",
                        "rms_e",
                        "i",
                        "rms_i",
                        "long. node",
                        "rms_long. node",
                        "arg. peric",
                        "rms_arg. peric",
                        "mean anomaly",
                        "rms_mean anomaly",
                        "chi_reduced",
                        "score",
                    ]
                ].values,
                3,
            ).tolist()
        )

        best_table = DoubleTable(best_orbit_data, "Best orbit")

        print(best_table.table)
        print("* a: Semi major axis, e: eccentricity, i: inclination")
        print()

        if arguments["mpc"]:
            print()
            path_alert = os.path.join(output_path, "save", "")
            if os.path.exists(path_alert):
                all_path_alert = glob.glob(os.path.join(path_alert, "alert_*"))
                alerts_pdf = pd.DataFrame()
                for path in all_path_alert:
                    pdf = pd.read_parquet(path)
                    alerts_pdf = pd.concat([alerts_pdf, pdf])

                alerts_pdf["ssnamenr"] = alerts_pdf["ssnamenr"].astype("string")

                gb = (
                    alerts_pdf.sort_values(["jd"])
                    .groupby(["ssnamenr"])
                    .agg(
                        trajectory_size=("candid", lambda x: len(list(x))),
                        nid=("nid", list),
                        diff_night=("nid", lambda x: list(np.diff(list(x)))),
                    )
                    .reset_index()
                )

                detectable_test = gb["trajectory_size"] >= int(
                    config["SOLVE_ORBIT_PARAMS"]["orbfit_limit"]
                )

                trivial_detectable_sso = gb[detectable_test]
                trivial_detectable_sso.insert(
                    len(trivial_detectable_sso.columns),
                    "detectable",
                    trivial_detectable_sso.apply(
                        test_detectable,
                        axis=1,
                        args=(
                            int(config["TW_PARAMS"]["trajectory_keep_limit"]),
                            int(config["SOLVE_ORBIT_PARAMS"]["orbfit_limit"]),
                        ),
                    ),
                )

                detectable_sso = trivial_detectable_sso[
                    trivial_detectable_sso["detectable"]
                ]

                obs_with_orb = orb_df.merge(traj_orb_df, on="trajectory_id")

                true_cand = (
                    obs_with_orb.groupby(["trajectory_id"])
                    .agg(
                        error=("ssnamenr", lambda x: len(np.unique(x))),
                        ssnamenr=("ssnamenr", list),
                    )
                    .reset_index()
                    .explode(["ssnamenr"])
                )

                true_orbit = true_cand[true_cand["error"] == 1]

                orb_cand = len(orb_df)
                pure_orb = len(np.unique(true_orbit["trajectory_id"]))
                purity = np.round_((pure_orb / orb_cand) * 100, decimals=2)

                detectable = len(np.unique(detectable_sso["ssnamenr"]))
                detected = len(np.unique(true_orbit["ssnamenr"]))
                efficiency = np.round_((detected / detectable) * 100, decimals=2)

                table_data = (
                    ("Metrics", "Values", "Notes"),
                    (
                        "True SSO",
                        len(np.unique(alerts_pdf["ssnamenr"])),
                        "Number of solar system objects (SSO) observed by ZTF since the first associations date with fink_fat.",
                    ),
                    (
                        "Detectable True SSO",
                        detectable,
                        "Number of SSO detectable with fink_fat according to the config file.\n(trajectory_keep_limit={} days / orbfit_limit={} points.".format(
                            config["TW_PARAMS"]["trajectory_keep_limit"],
                            config["SOLVE_ORBIT_PARAMS"]["orbfit_limit"],
                        ),
                    ),
                    (
                        "Orbit candidates",
                        orb_cand,
                        "Number of orbit detected with fink_fat",
                    ),
                    (
                        "Pure objects orbit",
                        pure_orb,
                        "Number of orbit candidates that contains only observations of the same SSO.",
                    ),
                    (
                        "Detected SSO",
                        detected,
                        "Number of unique SSO detected with fink_fat.\n(removes the SSO seen multiple time with fink_fat)",
                    ),
                    (
                        "Purity",
                        "{} %".format(purity),
                        "ratio between the number of orbit candidates and the number of pure orbits",
                    ),
                    (
                        "Efficiency",
                        "{} %".format(efficiency),
                        "ratio between the number of detectable sso and the number of detected sso with fink_fat.",
                    ),
                )

                table_instance = DoubleTable(table_data, "fink_fat performances")
                table_instance.justify_columns[2] = "right"
                print(table_instance.table)

                if arguments["--mpc-data"] is not None:

                    if os.path.exists(arguments["--mpc-data"]):
                        print()
                        print()
                        print("Load mpc database...")
                        mpc_data = pd.read_json(arguments["--mpc-data"])
                        mpc_data["Number"] = (
                            mpc_data["Number"].astype("string").str[1:-1]
                        )

                        sub_set_mpc = alerts_pdf.merge(
                            mpc_data, left_on="ssnamenr", right_on="Number", how="inner"
                        )

                        detectable_mpc = sub_set_mpc[
                            sub_set_mpc["ssnamenr"].isin(detectable_sso["ssnamenr"])
                        ].drop_duplicates(subset=["ssnamenr"])
                        pure_mpc = sub_set_mpc[
                            sub_set_mpc["ssnamenr"].isin(true_orbit["ssnamenr"])
                        ].drop_duplicates(subset=["ssnamenr"])

                        count_detect_orbit = Counter(detectable_mpc["Orbit_type"])
                        count_pure_orbit = Counter(pure_mpc["Orbit_type"])
                        table_rows = [["Orbit type", "Recovery"]]
                        for detect_key, detect_value in count_detect_orbit.items():
                            if detect_key in count_pure_orbit:
                                pure_value = count_pure_orbit[detect_key]
                            else:
                                pure_value = 0

                            table_rows.append(
                                [
                                    detect_key,
                                    "{} %".format(
                                        np.round_(
                                            (pure_value / detect_value) * 100,
                                            decimals=2,
                                        )
                                    ),
                                ]
                            )

                        orbit_type_table = DoubleTable(
                            table_rows, "Orbit type recovery performance"
                        )
                        print()
                        print(orbit_type_table.table)
                        print(
                            "\t*Ratio computed between the detectable object and the pure detected objects with fink_fat."
                        )

                        true_obs = obs_with_orb[
                            obs_with_orb["trajectory_id"].isin(
                                true_cand["trajectory_id"]
                            )
                        ]
                        detect_orb_with_mpc = true_obs.drop_duplicates(
                            subset=["trajectory_id"]
                        ).merge(
                            sub_set_mpc,
                            left_on="ssnamenr",
                            right_on="Number",
                            how="inner",
                        )

                        orbital_residue = compute_residue(detect_orb_with_mpc)[
                            ["da", "de", "di", "dNode", "dPeri", "dM"]
                        ]
                        residue_stats = orbital_residue.describe().round(decimals=3)

                        orbit_residue_data = (
                            ("orbital elements", "Metrics", "Values"),
                            (
                                "residue semi-major-axis (AU) (%)",
                                "mean",
                                residue_stats["da"]["mean"],
                            ),
                            ("", "std", residue_stats["da"]["std"]),
                            ("", "min", residue_stats["da"]["min"]),
                            ("", "max", residue_stats["da"]["max"]),
                            (
                                "residue eccentricity (%)",
                                "mean",
                                residue_stats["de"]["mean"],
                            ),
                            ("", "std", residue_stats["de"]["std"]),
                            ("", "min", residue_stats["de"]["min"]),
                            ("", "max", residue_stats["de"]["max"]),
                            (
                                "residue inclination (degrees) (%)",
                                "mean",
                                residue_stats["di"]["mean"],
                            ),
                            ("", "std", residue_stats["di"]["std"]),
                            ("", "min", residue_stats["di"]["min"]),
                            ("", "max", residue_stats["di"]["max"]),
                            (
                                "residue long. node (degrees) (%)",
                                "mean",
                                residue_stats["dNode"]["mean"],
                            ),
                            ("", "std", residue_stats["dNode"]["std"]),
                            ("", "min", residue_stats["dNode"]["min"]),
                            ("", "max", residue_stats["dNode"]["max"]),
                            (
                                "residue arg. peric (degrees) (%)",
                                "mean",
                                residue_stats["dPeri"]["mean"],
                            ),
                            ("", "std", residue_stats["dPeri"]["std"]),
                            ("", "min", residue_stats["dPeri"]["min"]),
                            ("", "max", residue_stats["dPeri"]["max"]),
                            (
                                "residue mean anomaly (degrees) (%)",
                                "mean",
                                residue_stats["dM"]["mean"],
                            ),
                            ("", "std", residue_stats["dM"]["std"]),
                            ("", "min", residue_stats["dM"]["min"]),
                            ("", "max", residue_stats["dM"]["max"]),
                        )

                        residue_table = SingleTable(
                            orbit_residue_data, "orbit residuals"
                        )
                        print(residue_table.table)
                        print(
                            "\t*Residues computed between the orbital elements from the pure detected objets and the orbital elements from the mpc database for the corresponding object."
                        )

                    else:
                        print()
                        print("The indicated path for the mpc database doesn't exist.")
                        exit()

                print(
                    "\t**Reminder: These performance statistics exists as fink_fat has been run in mpc mode."
                )

                exit()

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
                str(trajectory_df["last_assoc_date"].values[0].astype("datetime64[D]")),
                "%Y-%m-%d",
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

        stats_dict = {}

        while True:
            if arguments["--verbose"]:
                print("current processing date: {}".format(current_date))
                print()

            t_before = t.time()
            new_alerts = get_last_sso_alert(
                object_class, current_date.strftime("%Y-%m-%d"), arguments["--verbose"]
            )

            if arguments["--verbose"]:
                print("Number of alerts retrieve from fink: {}".format(len(new_alerts)))

            if arguments["--save"]:
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

            # get the last trajectory_id as baseline for new trajectories
            last_trajectory_id = 0
            if len(trajectory_df) > 0:
                if len(orb_df) > 0:
                    last_trajectory_id = np.max(
                        np.union1d(
                            trajectory_df["trajectory_id"], orb_df["trajectory_id"]
                        )
                    )
                else:
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
                    print()
                    print(
                        "Number of trajectories candidates send to solve orbit: {}".format(
                            nb_traj_to_orbfit
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
                    orbfit_time = t.time() - t_before

                    if arguments["--verbose"]:
                        print("time taken to get orbit: {}".format(orbfit_time))

                elif arguments["cluster"]:

                    t_before = t.time()
                    # return orbit results from cluster mode
                    orbit_results = cluster_mode(config, traj_to_orbital)
                    orbfit_time = t.time() - t_before

                    if arguments["--verbose"]:
                        print("time taken to get orbit: {}".format(orbfit_time))

                if len(orbit_results) > 0:

                    # get only the trajectories with orbital elements
                    # the other one are discards
                    current_traj_with_orb_elem = orbit_results[
                        orbit_results["a"] != -1.0
                    ]
                    nb_orb = len(current_traj_with_orb_elem)

                    # get the observations of trajectories with orbital elements
                    current_obs_with_orb = traj_to_orbital[
                        traj_to_orbital["trajectory_id"].isin(
                            current_traj_with_orb_elem["trajectory_id"]
                        )
                    ]

                    orb_df = pd.concat([orb_df, current_traj_with_orb_elem])
                    traj_orb_df = pd.concat([traj_orb_df, current_obs_with_orb])

                    (trajectory_df, orb_df, traj_orb_df,) = align_trajectory_id(
                        trajectory_df, orb_df, traj_orb_df
                    )

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
                print(
                    "The current processing day is greater than today. Out of the offline loop."
                )
                break

        if "last_assoc_date" in trajectory_df:
            trajectory_df["last_assoc_date"] = current_date
        else:
            trajectory_df.insert(
                len(trajectory_df.columns), "last_assoc_date", current_date
            )

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

        print("Offline mode ended")

    else:
        exit()
