import configparser
import json
import os

import numpy as np
import pandas as pd

import fink_fat


def string_to_bool(bool_str):
    if bool_str.casefold() == "false".casefold():
        return False
    else:
        return True


def init_cli(arguments):

    # read the config file
    config = configparser.ConfigParser()

    if arguments["--config"]:
        config.read(arguments["--config"])
    else:
        config_path = os.path.join(
            os.path.dirname(fink_fat.__file__), "data", "fink_fat.conf"
        )
        config.read(config_path)

    output_path = config["OUTPUT"]["association_output_file"]

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


def yes_or_no(
    intro_function, yes_function, no_function, intro_args=(), yes_args=(), no_args=()
):

    intro_function(*intro_args)

    answer = ""
    while answer.upper() not in ["Y", "YES", "N", "NO"]:
        answer = input("Continue?")
        if answer.upper() in ["Y", "YES"]:
            yes_function(*yes_args)
        elif answer.upper() in ["N", "NO"]:
            no_function(*no_args)
        else:
            print("please, answer with y or n.")


def save_additional_stats(save_path, date, stats):
    """
    Save the additional statistics

    Parameters
    ----------
    save_path : string
        json statistics file path
    date : string
        the computed statistics date
    stats : dict
        the additional statistics
    """
    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            stats_dict = json.load(f)
    else:
        stats_dict = {}

    stats_dict[date] = stats

    with open(save_path, "w") as f:
        json.dump(stats_dict, f, indent=4, sort_keys=True)


def align_trajectory_id(trajectory_df, orbit_df, obs_orbit_df):
    """
    Reasign the trajectory_id of the trajectories dataframe from 0 to the number of trajectories.
    Reasign also the trajectories with orbital elements with no overlapping with the trajectory_df.

    Parameters
    ----------
    trajectories : dataframe
        The set of observations belonging to the trajectories
    orbit_df : dataframe
        The set of the orbital elements.
    obs_orbit_df : dataframe
        The set of observations for the trajectories with orbital elements.

    Returns
    -------
    trajectories : dataframe
        Same as the input except that the trajectory_id column is in [0, nb trajectories[ .
    orbit_df : dataframe
        Same as the input except that the trajectory_id column is in [0, nb trajectories[ .
    obs_orbit_df : dataframe
        Same as the input except that the trajectory_id column is in [0, nb trajectories[ .

    Examples
    --------

    >>> trajectories = pd.DataFrame({
    ... "candid" : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
    ... "trajectory_id" : [2, 2, 4, 5, 4, 4, 6, 6, 5, 6, 2, 3, 3, 7, 7, 3, 7, 5]
    ... })

    >>> orb_df = pd.DataFrame({
    ... "a" : [1.2, 3.5, 2.7, 5.87, 6.23, 3.42],
    ... "trajectory_id" : [8, 10, 12, 24, 52, 41]
    ... })

    >>> obs_orbit_df = pd.DataFrame({
    ... "candid" : [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35],
    ... "trajectory_id" : [8, 10, 8, 12, 24, 12, 8, 41, 52, 12, 10, 24, 10, 41, 52, 24, 52, 41]
    ... })

    >>> tr_res, orb_res, obs_res = align_trajectory_id(trajectories, orb_df, obs_orbit_df)

    >>> tr_res
        candid  trajectory_id
    0        0              0
    1        1              0
    2        2              2
    3        3              3
    4        4              2
    5        5              2
    6        6              4
    7        7              4
    8        8              3
    9        9              4
    10      10              0
    11      11              1
    12      12              1
    13      13              5
    14      14              5
    15      15              1
    16      16              5
    17      17              3

    >>> orb_res
          a  trajectory_id
    0  1.20              6
    1  3.50              7
    2  2.70              8
    3  5.87              9
    4  6.23             11
    5  3.42             10

    >>> obs_res
        candid  trajectory_id
    0       18              6
    1       19              7
    2       20              6
    3       21              8
    4       22              9
    5       23              8
    6       24              6
    7       25             10
    8       26             11
    9       27              8
    10      28              7
    11      29              9
    12      30              7
    13      31             10
    14      32             11
    15      33              9
    16      34             11
    17      35             10

    >>> trajectories = pd.DataFrame(columns=["trajectory_id", "candid"])

    >>> tr_res, orb_res, obs_res = align_trajectory_id(trajectories, orb_df, obs_orbit_df)

    >>> tr_res
    Empty DataFrame
    Columns: [trajectory_id, candid]
    Index: []

    >>> orb_res
          a  trajectory_id
    0  1.20              0
    1  3.50              1
    2  2.70              2
    3  5.87              3
    4  6.23              5
    5  3.42              4

    >>> obs_res
        candid  trajectory_id
    0       18              0
    1       19              1
    2       20              0
    3       21              2
    4       22              3
    5       23              2
    6       24              0
    7       25              4
    8       26              5
    9       27              2
    10      28              1
    11      29              3
    12      30              1
    13      31              4
    14      32              5
    15      33              3
    16      34              5
    17      35              4
    """

    tr_id = np.union1d(
        np.unique(trajectory_df["trajectory_id"]), np.unique(orbit_df["trajectory_id"])
    )
    translate_tr = {
        old_id: new_id for old_id, new_id in zip(tr_id, np.arange(len(tr_id)))
    }

    with pd.option_context("mode.chained_assignment", None):
        if len(orbit_df) > 0:
            orbit_df["trajectory_id"] = orbit_df.apply(
                lambda x: translate_tr[x["trajectory_id"]], axis=1
            )

        if len(obs_orbit_df) > 0:
            obs_orbit_df["trajectory_id"] = obs_orbit_df.apply(
                lambda x: translate_tr[x["trajectory_id"]], axis=1
            )

        if len(trajectory_df) > 0:
            trajectory_df["trajectory_id"] = trajectory_df.apply(
                lambda x: translate_tr[x["trajectory_id"]], axis=1
            )

    return trajectory_df, orbit_df, obs_orbit_df


if __name__ == "__main__":  # pragma: no cover
    import sys
    import doctest
    from pandas.testing import assert_frame_equal  # noqa: F401
    import fink_fat.test.test_sample as ts  # noqa: F401
    from unittest import TestCase  # noqa: F401

    if "unittest.util" in __import__("sys").modules:
        # Show full diff in self.assertEqual.
        __import__("sys").modules["unittest.util"]._MAX_LENGTH = 999999999

    sys.exit(doctest.testmod()[0])
