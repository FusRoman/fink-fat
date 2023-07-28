import numpy as np
import pandas as pd
import os
import datetime
import time as t

import astropy.units as u

from fink_fat.associations.inter_night_associations import night_to_night_association

from fink_fat.command_line.utils_cli import (
    get_class,
    string_to_bool,
    yes_or_no,
    save_additional_stats,
)

from fink_fat.command_line.association_cli import (
    get_data,
    get_last_sso_alert,
    intro_reset,
    no_reset,
    yes_reset,
)

from fink_fat.others.utils import cast_obs_data


def cli_associations(arguments, config, output_path):
    """
    Perform the intra_night and inter_night associations of fink-fat from the command_line

    Parameters
    ----------
    arguments : dict
        command_line arguments
    config : ConfigParser
        object containing the data from the config file
    output_path : string
        path where are located the fink-fat data
    """

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

    trajectory_df, old_obs_df, last_trajectory_id = get_data(tr_df_path, obs_df_path)

    if len(trajectory_df) > 0:
        last_tr_date = pd.to_datetime(
            trajectory_df["last_assoc_date"], format="%Y-%m-%d"
        )

        last_obs_date = pd.to_datetime(old_obs_df["last_assoc_date"], format="%Y-%m-%d")

        last_request_date = max(last_tr_date.max(), last_obs_date.max())
        current_date = datetime.datetime.strptime(last_night, "%Y-%m-%d")

        if last_request_date == current_date:
            print()
            print("ERROR !!!")
            print("Association already done for this night.")
            print("Wait a next observation night to do new association")
            print("or run 'fink_fat solve_orbit' to get orbital_elements.")
            exit()
        if last_request_date > current_date:
            print()
            print("ERROR !!!")
            print(
                "Query alerts from a night before the last night in the recorded trajectory/old_observations."
            )
            print(
                "Maybe try with a more recent night or reset the associations with 'fink_fat association -r'"
            )
            exit()

    t_before = t.time()
    new_alerts = get_last_sso_alert(object_class, last_night, arguments["--verbose"])
    if len(new_alerts) == 0:
        print("no alerts available for the night of {}".format(last_night))
        exit()

    last_nid = next_nid = new_alerts["nid"][0]
    if len(trajectory_df) > 0 and len(old_obs_df) > 0:
        last_nid = np.max([np.max(trajectory_df["nid"]), np.max(old_obs_df["nid"])])

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
        string_to_bool(config["ASSOC_PARAMS"]["use_dbscan"]),
        float(config["ASSOC_PARAMS"]["inter_night_separation"]) * u.degree,
        float(config["ASSOC_PARAMS"]["inter_night_magdiff_limit_same_fid"]),
        float(config["ASSOC_PARAMS"]["inter_night_magdiff_limit_diff_fid"]),
        float(config["ASSOC_PARAMS"]["maximum_angle"]),
        string_to_bool(config["ASSOC_PERF"]["store_kd_tree"]),
        int(config["SOLVE_ORBIT_PARAMS"]["orbfit_limit"]),
        string_to_bool(config["ASSOC_SYSTEM"]["tracklets_with_trajectories"]),
        string_to_bool(config["ASSOC_SYSTEM"]["trajectories_with_new_observations"]),
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

    cast_obs_data(trajectory_df).to_parquet(tr_df_path)
    cast_obs_data(old_obs_df).to_parquet(obs_df_path)

    if arguments["--verbose"]:
        print("Association done")
