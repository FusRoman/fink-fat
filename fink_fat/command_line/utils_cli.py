import configparser
import json
import os

import numpy as np
import pandas as pd

import fink_fat
from fink_fat.others.id_tags import generate_tags
from fink_fat.others.utils import init_logging
from fink_fat.roid_fitting.utils_roid_fit import fit_traj, predict_equ
from typing import Tuple
import warnings
from astropy.coordinates import SkyCoord
from astropy.time import Time


def string_to_bool(bool_str):
    """
    Convert a string to a boolean.

    Note: Raise ValueError exception if the parameter is not 'true' or 'false'.

    Parameters
    ----------
    bool_str : string
        a string containing either 'true' or 'false'

    Returns
    bool_results : boolean
        True is bool_str is 'true' else 'false'

    Examples
    --------
    >>> string_to_bool("true")
    True
    >>> string_to_bool("false")
    False
    >>> string_to_bool("tRuE")
    True
    >>> string_to_bool("FaLse")
    False
    """
    if bool_str.casefold() == "false".casefold():
        return False
    elif bool_str.casefold() == "true".casefold():
        return True
    else:  # pragma: no cover
        raise ValueError(
            "the parameter is not a boolean string, should be 'true' or 'false'"
        )


class EnvInterpolation(configparser.BasicInterpolation):
    """Interpolation which expands environment variables in values."""

    def before_get(self, parser, section, option, value, defaults):
        value = super().before_get(parser, section, option, value, defaults)
        return os.path.expandvars(value)


def init_cli(arguments: dict) -> Tuple[configparser.ConfigParser, str]:
    """
    Read the fink_fat configuration file of fink_fat specified by the --config argument

    Parameters
    ----------
    arguments : dictionnary
        The arguments read from the command line and parse by docopt

    Returns
    -------
    config : dictionnary
        the options from the configuration file
    output_path : string
        the output path where to store the product of fink_fat

    Examples
    --------
    >>> arguments = {
    ...    "--config" : "fink_fat/test/cli_test/test.conf"
    ... }

    >>> config, output_path = init_cli(arguments)

    >>> config.sections()
    ['TW_PARAMS', 'ASSOC_PARAMS', 'ASSOC_PERF', 'SOLVE_ORBIT_PARAMS', 'ASSOC_SYSTEM', 'OUTPUT']
    >>> output_path
    'fink_fat/test/cli_test/fink_fat_out'

    >>> config, output_path = init_cli({"--config":""})

    >>> config.sections()
    ['TW_PARAMS', 'ASSOC_PARAMS', 'ASSOC_PERF', 'SOLVE_ORBIT_PARAMS', 'ASSOC_SYSTEM', 'OUTPUT']
    >>> output_path
    'fink_fat_out'

    >>> os.rmdir("fink_fat_out")
    """
    # read the config file
    config = configparser.ConfigParser(os.environ, interpolation=EnvInterpolation())

    if arguments["--config"]:
        if os.path.exists(arguments["--config"]):
            config.read(arguments["--config"])
        else:  # pragma: no cover
            logger = init_logging()
            logger.info(
                "config file does not exist from this path: {} !!".format(
                    arguments["--config"]
                )
            )
            exit(1)
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
    """
    Return the fink object name corresponding to the arguments given by the users
    'mpc' for the known objects and 'candidates' for the sso candidates.

    Parameters
    ----------
    arguments : dictionnary
        The arguments read from the command line and parse by docopt
    path : string
        path where to store the fink_fat product read from the config file.

    Returns
    -------
    path : string
        path where to store the fink_fat product corresponding to the object class.
    object_class : string
        contains 'Solar System MPC' if the argument is 'mpc'
            or 'Solar System candidate' if the argument is 'candidates'.

    Examples
    --------
    >>> arguments = {
    ...    "--config" : "fink_fat/test/cli_test/test.conf",
    ...    "mpc" : True
    ... }
    >>> config, output_path = init_cli(arguments)
    >>> path, object_class = get_class(arguments, output_path)

    >>> path
    'fink_fat/test/cli_test/fink_fat_out/mpc/'
    >>> object_class
    'Solar System MPC'

    >>> arguments = {
    ...    "--config" : "fink_fat/test/cli_test/test.conf",
    ...    "mpc" : False,
    ...    "candidates" : True
    ... }
    >>> config, output_path = init_cli(arguments)
    >>> path, object_class = get_class(arguments, output_path)

    >>> path
    'fink_fat/test/cli_test/fink_fat_out/candidates/'
    >>> object_class
    'Solar System candidate'

    >>> shutil.rmtree("fink_fat/test/cli_test/fink_fat_out")
    """
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
    elif arguments["fitroid"]:
        path = os.path.join(path, "fitroid", "")
        if not os.path.isdir(path):
            os.mkdir(path)
        object_class = "SSO fitroid"
    else:  # pragma: no cover
        raise ValueError(
            "Class does not correspond to a sso class from fink\nargument keys: {}".format(
                arguments
            )
        )

    return path, object_class


def yes_or_no(
    intro_function, yes_function, no_function, intro_args=(), yes_args=(), no_args=()
):  # pragma: no cover
    """
    Function for the -r options.
    Ask to the user a questions from the intro_function and depending of the answer, apply the yes_function or
    the no_function.

    Parameters
    ----------
    intro_function : Callable
    yes_function : Callable
    no_function : Callable
    intro_args : tuples
        arguments of the intro_function
    yes_args : tuples
        arguments of the yes_function
    no_args : tuples
        arguments of the no function

    Returns
    -------
    None

    """

    intro_function(*intro_args)

    answer = ""
    while answer.upper() not in ["Y", "YES", "N", "NO"]:
        answer = input("Continue?")
        if answer.upper() in ["Y", "YES"]:
            yes_function(*yes_args)
        elif answer.upper() in ["N", "NO"]:
            no_function(*no_args)
        else:
            logger = init_logging()
            logger.info("please, answer with y or n.")


def save_additional_stats(save_path, date, stats):  # pragma: no cover
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

    Returns
    -------
    None
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


def assig_tags(
    orb_df: pd.DataFrame, traj_orb_df: pd.DataFrame, start_tags: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Assign the ssoCandId to the orbital and traj_orb dataframe

    Parameters
    ----------
    orb_df : DataFrame
        contains the orbital elements
    traj_orb_df : DataFrame
        contains the observations
    start_tags : integer
        start generation tags number, the tags will be between start_tags and len(orb_df)

    Returns
    -------
    orb_df : DataFrame
        same as inputs with a new column called ssoCandId, the trajectory_id column has been dropped
    traj_orb_df : DataFrame
        same as inputs with a new column called ssoCandId, the trajectory_id column has been dropped

    Examples
    --------
    >>> orb = pd.DataFrame({
    ... "trajectory_id": [0, 1, 2, 3, 4, 5],
    ... "a": [1, 1.5, 1.6, 2.8, 35.41, 265.32],
    ... "ref_epoch": [2460235.42, 2460412.42, 2460842.42, 2460137.42, 2460131.42, 2460095.42]
    ... })
    >>> traj = pd.DataFrame({
    ... "trajectory_id": [0, 0, 0, 0, 0, 0, 1, 2, 1, 2, 1, 3, 3, 4, 5, 3, 5, 4, 5, 4, 3, 1, 2, 3, 4, 5, 5, 2],
    ... "candid": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27],
    ... "ra": [0, 0, 0, 0, 0, 0, 1, 2, 1, 2, 1, 3, 3, 4, 5, 3, 5, 4, 5, 4, 3, 1, 2, 3, 4, 5, 5, 2]
    ... })
    >>> new_orb, new_traj = assig_tags(orb, traj, 0)

    >>> orb_test = pd.read_parquet("fink_fat/test/utils_cli_test_orb.parquet").reset_index(drop=True)
    >>> traj_test = pd.read_parquet("fink_fat/test/utils_cli_test_traj.parquet").reset_index(drop=True)

    >>> assert_frame_equal(orb_test, new_orb.reset_index(drop=True))
    >>> assert_frame_equal(traj_test, new_traj.reset_index(drop=True))
    """
    orb_df = orb_df.sort_values("ref_epoch")

    all_tags = generate_tags(start_tags, start_tags + len(orb_df), orb_df["ref_epoch"])
    int_id_to_tags = {
        tr_id: tag for tr_id, tag in zip(orb_df["trajectory_id"], all_tags)
    }

    assert len(np.unique(orb_df["trajectory_id"])) == len(int_id_to_tags)
    with pd.option_context("mode.chained_assignment", None):
        orb_df["ssoCandId"] = orb_df["trajectory_id"].map(int_id_to_tags)
        traj_orb_df["ssoCandId"] = traj_orb_df["trajectory_id"].map(int_id_to_tags)

    orb_df = orb_df.drop("trajectory_id", axis=1)
    traj_orb_df = traj_orb_df.drop("trajectory_id", axis=1)
    return orb_df, traj_orb_df


def chi_square(ra: np.ndarray, dec: np.ndarray, jd: np.ndarray)->float:
    """
    Compute chi-square of a trajectory fitted using a polynomial function

    Parameters
    ----------
    ra : np.ndarray
        right ascension
    dec : np.ndarray
        declination
    jd : np.ndarray
        julian date of observations

    Returns
    -------
    float
        chi-square computed on the trajectory
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        popt = fit_traj(ra, dec, jd)
    prediction = predict_equ(popt[0], popt[1], popt[2], jd)

    true = SkyCoord(ra, dec, unit="deg")
    return np.sum(true.separation(prediction).value ** 2) / 3


def chi_filter(
        trajectory_df: pd.DataFrame,
        fit_df: pd.DataFrame,
        chi_limit: float
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter the trajectories based on the chi square.
    Trajectories with a chi square below the chi_limit are discarded.

    Parameters
    ----------
    trajectory_df : pd.DataFrame
        trajectories dataframe containing ra, dec and jd columns
    chi_limit : float
        chi filtering limit

    Returns
    -------
    pd.DataFrame
        trajectory dataframe filtered using chi square values
    """
    chi = trajectory_df.groupby("trajectory_id").apply(
        lambda x: chi_square(x["ra"], x["dec"], x["jd"])
    )

    traj_filt = trajectory_df[trajectory_df["trajectory_id"].isin(chi[chi <= chi_limit].index)]
    fit_filt = fit_df[fit_df["trajectory_id"].isin(chi[chi <= chi_limit].index)]

    return traj_filt, fit_filt

def time_window(
        trajectory_df: pd.DataFrame,
        fit_df: pd.DataFrame,
        current_time: float, 
        time_window: int
    )->Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove trajectories based on a time window.
    The time delay is computed between the last observations of each trajectories and the current time.
    Each trajectories with a delay greater than the time window are discarded.

    Parameters
    ----------
    trajectory_df : pd.DataFrame
        trajectories dataframe containing ra, dec and jd columns
    current_time : float
        the current processing time of fink-fat
    time_window : int
        time window limit

    Returns
    -------
    pd.DataFrame
        trajectories dataframe filtered using the time window threshold
    """
    last_pdf = trajectory_df.sort_values(["trajectory_id", "jd"]).groupby("trajectory_id").agg(
        last_jd=("jd", lambda x: list(x)[-1])
    )


    last_time = Time(last_pdf["last_jd"].round(decimals=0), format="jd").jd

    traj_window = trajectory_df[
        trajectory_df["trajectory_id"].isin(
            last_pdf[(current_time - last_time) <= time_window].index
    )]
    fit_df = fit_df[
        fit_df["trajectory_id"].isin(
            last_pdf[(current_time - last_time) <= time_window].index
    )]

    return traj_window, fit_df


if __name__ == "__main__":  # pragma: no cover
    import sys
    import doctest
    from pandas.testing import assert_frame_equal  # noqa: F401
    import fink_fat.test.test_sample as ts  # noqa: F401
    from unittest import TestCase  # noqa: F401
    import shutil  # noqa: F401
    import datetime  # noqa: F401

    if "unittest.util" in __import__("sys").modules:
        # Show full diff in self.assertEqual.
        __import__("sys").modules["unittest.util"]._MAX_LENGTH = 999999999

    sys.exit(doctest.testmod()[0])
