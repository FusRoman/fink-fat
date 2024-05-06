import datetime
import os
import numpy as np
import pandas as pd
import requests
import shutil
from io import BytesIO

from astropy.time import Time

def request_fink(
    object_class,
    n_sso,
    startdate,
    stopdate,
    request_columns,
    verbose,
    nb_tries,
    current_tries,
):
    """
    Get the alerts corresponding to the object_class from Fink with the API.

    Parameters
    ----------
    object_class : string
        should be either 'Solar System MPC' or 'Solar System candidates'
    n_sso : integer
        number of alerts to retrieve
    startdate : datetime
        start date of the request
    stopdate : datetime
        stop date of the request
    request_columns : string
        the request will return only the columns specified in this string.
        The columns name are comma-separated.
    verbose : boolean
        print some informations during the process
    nb_tries : integer
        the maximum number of trials if the request failed.
    current_tries : integer
        the current number of trials / the number of failed requests.

    Returns
    -------
    alert_pdf : dataframe
        the alerts get from fink for the corresponding interval of time.

    Examples
    --------
    >>> request_fink(
    ... 'Solar System MPC',
    ... 10,
    ... datetime.datetime.strptime("2022-06-22", "%Y-%m-%d"),
    ... datetime.datetime.strptime("2022-06-23", "%Y-%m-%d"),
    ... "i:ra, i:dec, i:jd",
    ... False,
    ... 1,
    ... 0
    ... )
           i:dec          i:jd        i:ra
    0  12.543168  2.459753e+06  356.113631
    1  12.907248  2.459753e+06  356.054645
    2  14.686620  2.459753e+06  353.888462
    3  15.305179  2.459753e+06  354.357292
    4  10.165192  2.459753e+06  353.209892
    5  10.296633  2.459753e+06  353.197138
    6  10.504157  2.459753e+06  353.403702
    7  10.305569  2.459753e+06  358.151255
    8  10.270319  2.459753e+06  357.861407
    9  10.307019  2.459753e+06  358.332623
    """

    if current_tries == nb_tries:
        return pd.DataFrame(
            columns=[
                "ra",
                "dec",
                "jd",
                "nid",
                "fid",
                "magpsf",
                "sigmapsf",
                "candid",
                "not_updated",
            ]
        )

    # +- 1 hour due to hbase issue
    r = requests.post(
        "https://fink-portal.org/api/v1/latests",
        json={
            "class": object_class,
            "n": "{}".format(n_sso),
            "startdate": str((startdate - datetime.timedelta(hours=1))),
            "stopdate": str((stopdate + datetime.timedelta(hours=1))),
            "columns": request_columns,
        },
    )

    try:
        pdf = pd.read_json(BytesIO(r.content))
        if len(pdf) != n_sso:
            if verbose:
                print(
                    "error when trying to get fink alerts !!!\n\t number of alerts get from the API call ({}) is different from the real number of alerts ({}) \n\ttry again !".format(
                        len(pdf), n_sso
                    )
                )
            return request_fink(
                object_class,
                n_sso,
                startdate,
                stopdate,
                request_columns,
                verbose,
                nb_tries,
                current_tries + 1,
            )
        else:
            return pdf

    except ValueError:  # pragma: no cover
        if verbose:
            print("error when trying to get fink alerts, try again !")

        return request_fink(
            object_class,
            n_sso,
            startdate,
            stopdate,
            request_columns,
            verbose,
            nb_tries,
            current_tries + 1,
        )


def get_n_sso(object_class, date):
    """
    Get the exact number of object for the given date by using the fink statistics API.

    Parameters
    ----------
    object_class : string
        should be either 'Solar System MPC' or 'Solar System candidates'
    startdate : datetime
        start date of the request

    Returns
    -------
    n_sso : integer
        the number of alerts for the corresponding class and date.

    Examples
    --------
    >>> get_n_sso(
    ... 'Solar System MPC',
    ... "2022-06-22".replace("-", ""),
    ... )
    10536

    >>> get_n_sso(
    ... 'Solar System MPC',
    ... "2020-05-21".replace("-", ""),
    ... )
    0
    """
    r = requests.post(
        "https://fink-portal.org/api/v1/statistics",
        json={"date": str(date), "output-format": "json"},
    )

    pdf = pd.read_json(BytesIO(r.content))

    if len(pdf) == 0 or "class:Solar System candidate" not in pdf:
        return 0

    return pdf["class:{}".format(object_class)].values[0]


def get_last_sso_alert_from_file(filepath, verbose=False):
    """
    Read single night alert data from a file. The file header **must contain** at least:
    ra, dec, jd, magpsf, sigmapsf

    other fields can exist, and will remain attached.

    Parameters
    ----------
    filepath: str
        Path to a file containing measurements
    verbose: bool
        If True, print extra information. Default is False.

    Returns
    -------
    pdf : pd.DataFrame
        the alerts from Fink with the following columns:
            ra, dec, jd, nid, fid, magpsf, sigmapsf, candid, not_updated

    Examples
    --------
    >>> data_path = "fink_fat/data/sample_euclid.txt"
    >>> pdf = get_last_sso_alert_from_file(data_path)
    >>> assert len(pdf) == 2798
    >>> assert 'objectId' in pdf.columns
    """
    pdf = pd.read_parquet(filepath)

    required_header = ['ra', 'dec', 'jd', 'magpsf', 'sigmapsf']
    msg = """
    The header of {} must contain at least the following fields:
    ra dec jd magpsf sigmapsf
    """.format(filepath)
    assert set(required_header) - set(pdf.columns) == set(), AssertionError(msg)

    if 'objectId' not in pdf.columns:
        pdf['objectId'] = range(len(pdf))

    pdf['candid'] = range(10, len(pdf) + 10)
    pdf['nid'] = 0
    pdf['fid'] = 0

    required_columns = [
        "objectId",
        "candid",
        "ra",
        "dec",
        "jd",
        "nid",
        "fid",
        "magpsf",
        "sigmapsf",
        "not_updated",
        "last_assoc_date",
    ]

    if len(pdf) > 0:
        date = Time(pdf['jd'].values[0], format='jd').iso.split(' ')[0]
        pdf.insert(len(pdf.columns), "not_updated", np.ones(len(pdf), dtype=np.bool_))
        pdf.insert(len(pdf.columns), "last_assoc_date", date)
    else:
        return pd.DataFrame(columns=required_columns)

    return pdf[required_columns]

def get_last_sso_alert(object_class, date, verbose=False):
    """
    Get the alerts from Fink corresponding to the object_class for the given date.

    Parameters
    ----------
    object_class : string
        the class of the requested alerts
    date : string
        the requested date of the alerts, format is YYYY-MM-DD

    Returns
    -------
    pdf : pd.DataFrame
        the alerts from Fink with the following columns:
            ra, dec, jd, nid, fid, magpsf, sigmapsf, candid, not_updated

    Examples
    --------
    >>> res_request = get_last_sso_alert(
    ... 'Solar System candidate',
    ... '2020-06-29'
    ... )

    >>> pdf_test = pd.read_parquet("fink_fat/test/cli_test/get_sso_alert_test.parquet")
    >>> assert_frame_equal(res_request, pdf_test)

    >>> res_request = get_last_sso_alert(
    ... 'Solar System candidate',
    ... '2020-05-21'
    ... )
    >>> assert_frame_equal(res_request, pd.DataFrame(columns=["objectId", "candid", "ra", "dec", "jd", "nid", "fid", "magpsf", "sigmapsf", "not_updated", "last_assoc_date"]))
    """
    startdate = datetime.datetime.strptime(date, "%Y-%m-%d")
    stopdate = startdate + datetime.timedelta(days=1)

    if verbose:  # pragma: no cover
        print(
            "Query fink broker to get sso alerts for the night between {} and {}".format(
                startdate.strftime("%Y-%m-%d"), stopdate.strftime("%Y-%m-%d")
            )
        )

    request_columns = (
        "i:objectId,i:candid,i:ra,i:dec,i:jd,i:nid,i:fid,i:magpsf,i:sigmapsf"
    )
    if object_class == "Solar System MPC":  # pragma: no cover
        request_columns += ", i:ssnamenr"

    n_sso = get_n_sso(object_class, date.replace("-", ""))

    pdf = request_fink(
        object_class, n_sso, startdate, stopdate, request_columns, verbose, 5, 0
    )

    translate_columns = {
        "i:objectId": "objectId",
        "i:candid": "candid",
        "i:ra": "ra",
        "i:dec": "dec",
        "i:jd": "jd",
        "i:nid": "nid",
        "i:fid": "fid",
        "i:magpsf": "magpsf",
        "i:sigmapsf": "sigmapsf",
    }

    required_columns = [
        "objectId",
        "candid",
        "ra",
        "dec",
        "jd",
        "nid",
        "fid",
        "magpsf",
        "sigmapsf",
        "not_updated",
        "last_assoc_date",
    ]

    if object_class == "Solar System MPC":  # pragma: no cover
        required_columns.append("ssnamenr")
        translate_columns["i:ssnamenr"] = "ssnamenr"

    pdf = pdf.rename(columns=translate_columns)
    if len(pdf) > 0:
        pdf.insert(len(pdf.columns), "not_updated", np.ones(len(pdf), dtype=np.bool_))
        pdf.insert(len(pdf.columns), "last_assoc_date", date)
    else:
        return pd.DataFrame(columns=required_columns)

    return pdf[required_columns]


def intro_reset():  # pragma: no cover
    print("WARNING !!!")
    print(
        "you will loose the trajectory done by previous association, Continue ? [Y/n]"
    )


def yes_reset(arguments, tr_df_path, obs_df_path):  # pragma: no cover
    if os.path.exists(tr_df_path) and os.path.exists(obs_df_path):
        print("Removing files :\n\t{}\n\t{}".format(tr_df_path, obs_df_path))
        try:
            os.remove(tr_df_path)
            os.remove(obs_df_path)
        except OSError as e:
            if arguments["--verbose"]:
                print("Failed with:", e.strerror)
                print("Error code:", e.code)
    else:
        print("File trajectory and old observations not exists.")

    dirname = os.path.dirname(tr_df_path)
    save_path = os.path.join(dirname, "save", "")
    if os.path.exists(save_path):
        shutil.rmtree(save_path)


def no_reset():  # pragma: no cover
    print("Abort reset.")


def get_data(tr_df_path, obs_df_path):
    """
    Load the trajectory and old observations save by the previous call of fink_fat

    Parameters
    ----------
    tr_df_path : string
        path where are saved the trajectory observations
    obs_df_path : string
        path where are saved the old observations

    Returns
    -------
    trajectory_df : dataframe
        the trajectory observations
    old_obs_df : dataframe
        the old observations
    last_trajectory_id : integer
        the last trajectory identifier given to a trajectory.

    Examples
    --------
    >>> data_path = "fink_fat/test/cli_test/fink_fat_out_test/mpc/"
    >>> tr_df, old_obs_df, last_tr_id = get_data(
    ... data_path + "trajectory_df.parquet",
    ... data_path + "old_obs.parquet"
    ... )

    >>> len(tr_df)
    12897
    >>> len(old_obs_df)
    5307
    >>> last_tr_id
    7866

    >>> tr_df, old_obs_df, last_tr_id = get_data(
    ... data_path + "trajectory_df.parquet",
    ... data_path + "old_obs.parquet"
    ... )

    >>> len(tr_df)
    12897
    >>> len(old_obs_df)
    5307
    >>> last_tr_id
    7866
    """
    tr_columns = [
        "ra",
        "dec",
        "jd",
        "nid",
        "fid",
        "magpsf",
        "sigmapsf",
        "candid",
        "not_updated",
        "last_assoc_date",
    ]
    # last_nid = next_nid = new_alerts["nid"][0]
    trajectory_df = pd.DataFrame(columns=tr_columns + ["trajectory_id"])
    old_obs_df = pd.DataFrame(columns=tr_columns)

    last_trajectory_id = 0

    # test if the trajectory_df and old_obs_df exists in the output directory.
    if os.path.exists(tr_df_path) and os.path.exists(obs_df_path):

        trajectory_df = pd.read_parquet(tr_df_path)
        old_obs_df = pd.read_parquet(obs_df_path)
        # last_nid = np.max([np.max(trajectory_df["nid"]), np.max(old_obs_df["nid"])])
        # if last_nid == next_nid:
        #     print()
        #     print("ERROR !!!")
        #     print("Association already done for this night.")
        #     print("Wait a next observation night to do new association")
        #     print("or run 'fink_fat solve_orbit' to get orbital_elements.")
        #     exit()
        # if last_nid > next_nid:
        #     print()
        #     print("ERROR !!!")
        #     print(
        #         "Query alerts from a night before the last night in the recorded trajectory/old_observations."
        #     )
        #     print(
        #         "Maybe try with a more recent night or reset the associations with 'fink_fat association -r'"
        #     )
        #     exit()

        last_trajectory_id = np.max(trajectory_df["trajectory_id"])

    return trajectory_df, old_obs_df, last_trajectory_id


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
