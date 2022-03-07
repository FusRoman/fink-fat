"""
Usage:
    fink_fat associations (mpc | candidates) [--night <date>] [options]
    fink_fat solve_orbit (mpc | candidates) [options]
    fink_fat stats (mpc | candidates) [options]
    fink_fat -h | --help
    fink_fat --version

Options:
  mpc                              Return the associations on the solar system mpc alerts (only for tests purpose).
  candidates                       Run the associations on the solar system candidates alerts.
  stats                            Print statistics about trajectories detected by assocations, the old observations
                                   and, if exists, the orbital elements for some trajectories.
  -n <date> --night <date>         Specify the night to request sso alerts from fink broker.
                                   Format is yyyy-mm-dd as yyyy = year, mm = month, dd = day.
                                   Example : 2022-03-04 for the 2022 march 04.
                                   [intervall of day between the day starting at night midday until night midday + 1]
  -r --reset                       Remove the file containing the trajectories and the old observations.
  -h --help                        Show help and quit.
  --version                        Show version.
  --config FILE                    Specify the config file [default: conf/fink_fat.conf]
  --output PATH                    Specify the out directory. A default path is set in the default fink_fat.conf
  --verbose                        Print information and progress bar during the process
"""

from collections import Counter
from collections import OrderedDict
from docopt import docopt
import configparser
import os
import pandas as pd
import numpy as np
import datetime
import requests
from fink_science.conversion import dc_mag
from astropy import units as u

import fink_fat
from fink_fat.associations.inter_night_associations import night_to_night_association
from fink_fat.others.utils import cast_obs_data
from fink_fat.orbit_fitting.orbfit_management import compute_df_orbit_param


def string_to_bool(str):
    if str.casefold() == "false".casefold():
        return False
    else:
        return True


def get_last_sso_alert(object_class, date, verbose=False):
    startdate = datetime.datetime.strptime(date, "%Y-%m-%d")
    stopdate = startdate + datetime.timedelta(days=1)

    if verbose:
        print(
            "Query fink broker to get sso alerts for the night between {} and {}".format(
                startdate.strftime("%Y-%m-%d"), stopdate.strftime("%Y-%m-%d")
            )
        )

    r = requests.post(
        "https://fink-portal.org/api/v1/latests",
        json={
            "class": object_class,
            "n": "100",
            "startdate": str(startdate),
            "stopdate": str(stopdate),
        },
    )
    pdf = pd.read_json(r.content)

    required_columns = [
        "ra",
        "dec",
        "jd",
        "nid",
        "fid",
        "dcmag",
        "candid",
        "not_updated",
    ]
    translate_columns = {
        "i:ra": "ra",
        "i:dec": "dec",
        "i:jd": "jd",
        "i:nid": "nid",
        "i:fid": "fid",
        "i:candid": "candid",
    }
    if object_class == "Solar System MPC":
        required_columns.append("ssnamenr")
        translate_columns["i:ssnamenr"] = "ssnamenr"

    _dc_mag = np.array(
        pdf.apply(
            lambda x: dc_mag(
                x["i:fid"],
                x["i:magpsf"],
                x["i:sigmapsf"],
                x["i:magnr"],
                x["i:sigmagnr"],
                x["i:magzpsci"],
                x["i:isdiffpos"],
            ),
            axis=1,
            result_type="expand",
        ).values
    )

    pdf = pdf.rename(columns=translate_columns)
    if len(_dc_mag) > 0:
        pdf.insert(len(pdf.columns), "dcmag", _dc_mag[:, 0])
    else:
        return pd.DataFrame(columns=required_columns)

    pdf.insert(len(pdf.columns), "not_updated", np.ones(len(pdf), dtype=np.bool_))
    return pdf[required_columns]


def init_cli(arguments):
    # read the config file
    config = configparser.ConfigParser()
    config.read(arguments["--config"])
    output_path = config["OUTPUT"]["association_output_file"]

    if arguments["--output"]:
        output_path = arguments["--output"]

    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    return config, output_path


def get_class(arguments, path):
    if arguments["mpc"]:
        path = os.path.join(path, "mpc", "")
        if not os.path.isdir(path):
            os.mkdir(path)
        object_class = "Solar System MPC"

    elif arguments["candidates"]:
        path = os.path.join(path, "candidates", "")
        if not os.path.isdir(path):
            os.mkdir(path)
        object_class = "Solar System candidate"

    return path, object_class


if __name__ == "__main__":

    # parse the command line and return options provided by the user.
    arguments = docopt(__doc__, version=fink_fat.__version__)

    config, output_path = init_cli(arguments)

    if arguments["associations"]:

        output_path, object_class = get_class(arguments, output_path)

        tr_df_path = os.path.join(output_path, "trajectory_df.parquet")
        obs_df_path = os.path.join(output_path, "old_obs.parquet")

        if arguments["--reset"]:
            print("WARNING !!!")
            print(
                "you will loose the trajectory done by previous association, Continue ? [Y/n]"
            )
            answer = ""
            while answer.upper() not in ["Y", "YES", "N", "NO"]:
                answer = input("Continue?")
                if answer.upper() in ["Y", "YES"]:
                    if os.path.exists(tr_df_path) and os.path.exists(obs_df_path):
                        print(
                            "Removing files :\n\t{}\n\t{}".format(
                                tr_df_path, obs_df_path
                            )
                        )
                        try:
                            os.remove(tr_df_path)
                            os.remove(obs_df_path)
                        except OSError as e:
                            if arguments["--verbose"]:
                                print("Failed with:", e.strerror)
                                print("Error code:", e.code)
                    else:
                        print("File trajectory and old observations not exists.")
                elif answer.upper() in ["N", "NO"]:
                    print("Abort reset.")
                else:
                    print("please, answer with y or n.")

        last_night = datetime.datetime.now() - datetime.timedelta(days=1)
        last_night = last_night.strftime("%Y-%m-%d")
        if arguments["--night"]:
            last_night = arguments["--night"]

        new_alerts = get_last_sso_alert(
            object_class, last_night, arguments["--verbose"]
        )

        if len(new_alerts) == 0:
            print("no alerts available for this night")
            exit()

        last_nid = next_nid = new_alerts["nid"][0]
        trajectory_df = pd.DataFrame(columns=new_alerts.columns)
        old_obs_df = pd.DataFrame(columns=new_alerts.columns)

        # test if the trajectory_df and old_obs_df exists in the output directory.
        if os.path.exists(tr_df_path) and os.path.exists(obs_df_path):

            trajectory_df = pd.read_parquet(tr_df_path)
            old_obs_df = pd.read_parquet(obs_df_path)
            last_nid = np.max([np.max(trajectory_df["nid"]), np.max(old_obs_df["nid"])])
            if last_nid == next_nid:
                print()
                print("ERROR !!!")
                print("Association already done for this night.")
                print("Wait a next observation night to do new association")
                print("or run 'fink_fat solve_orbit' to get orbital_elements.")
                exit()
            if last_nid > next_nid:
                print()
                print("ERROR !!!")
                print(
                    "Query alerts from a night before the last night in the recorded trajectory/old_observations."
                )
                print(
                    "Maybe try with a more recent night or reset the associations with 'fink_fat association -r'"
                )
                exit()

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
            print("WARNING !!!")
            print(
                "you will loose the previously computed orbital elements and all the associated observations, Continue ? [Y/n]"
            )
            answer = ""
            while answer.upper() not in ["Y", "YES", "N", "NO"]:
                answer = input("Continue?")
                if answer.upper() in ["Y", "YES"]:
                    if os.path.exists(orb_res_path) and os.path.exists(traj_orb_path):
                        print(
                            "Removing files :\n\t{}\n\t{}".format(
                                orb_res_path, traj_orb_path
                            )
                        )
                        try:
                            os.remove(orb_res_path)
                            os.remove(traj_orb_path)
                        except OSError as e:
                            if arguments["--verbose"]:
                                print("Failed with:", e.strerror)
                                print("Error code:", e.code)
                    else:
                        print("File with orbital elements not exists.")
                elif answer.upper() in ["N", "NO"]:
                    print("Abort reset.")
                else:
                    print("please, answer with y or n.")

        # test if the trajectory_df exist in the output directory.
        if os.path.exists(tr_df_path):
            trajectory_df = pd.read_parquet(tr_df_path)
        else:
            print(
                "Trajectory file doesn't exist, run 'fink_fat association (mpc | candidates)' to create it."
            )
            exit()

        # get trajectories with a number of points greater than the orbfit limit
        gb = trajectory_df.groupby(["trajectory_id"]).count().reset_index()
        traj = gb[gb["ra"] >= int(config["SOLVE_ORBIT_PARAMS"]["orbfit_limit"])][
            "trajectory_id"
        ]
        traj_to_orbital = trajectory_df[trajectory_df["trajectory_id"].isin(traj)]

        if len(traj_to_orbital) > 0:
            orbit_results = compute_df_orbit_param(
                traj_to_orbital,
                int(config["SOLVE_ORBIT_PARAMS"]["cpu_count"]),
                config["SOLVE_ORBIT_PARAMS"]["ram_dir"],
            )

            if len(orbit_results) > 0:
                traj_with_orb = orbit_results["trajectory_id"]
                test_orb = trajectory_df["trajectory_id"].isin(traj_with_orb)

                obs_with_orb = trajectory_df[test_orb]
                obs_without_orb = trajectory_df[~test_orb]

                if os.path.exists(orb_res_path):
                    orb_df = pd.read_parquet(orb_res_path)
                    orb_df = pd.concat([orb_df, orbit_results])
                    orb_df.to_parquet(orb_res_path)
                else:
                    orbit_results.to_parquet(orb_res_path)

                if os.path.exists(traj_orb_path):
                    traj_orb_df = pd.read_parquet(traj_orb_path)
                    traj_orb_df = pd.concat([traj_orb_df, obs_with_orb])
                    traj_orb_df.to_parquet(traj_orb_path)
                else:
                    obs_with_orb.to_parquet(traj_orb_path)

                obs_without_orb.to_parquet(tr_df_path)

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
