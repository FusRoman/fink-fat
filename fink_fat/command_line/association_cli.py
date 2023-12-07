import datetime
import os
import numpy as np
import pandas as pd
import requests
import shutil
from io import BytesIO
from fink_fat.others.utils import init_logging, LoggerNewLine

from astropy.time import Time
import fink_fat
import subprocess
from pyspark.sql.functions import col
from fink_fat.command_line.utils_cli import string_to_bool
import configparser
import pathlib


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

    logger = init_logging()

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
                logger.info(
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
            logger.info("error when trying to get fink alerts, try again !")

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
    pdf = pd.read_csv(filepath, header=0, sep=r"\s+", index_col=False)

    required_header = ["ra", "dec", "jd", "magpsf", "sigmapsf"]
    msg = """
    The header of {} must contain at least the following fields:
    ra dec jd magpsf sigmapsf
    """.format(
        filepath
    )
    assert set(required_header) - set(pdf.columns) == set(), AssertionError(msg)

    if "objectId" not in pdf.columns:
        pdf["objectId"] = range(len(pdf))

    pdf["candid"] = range(10, len(pdf) + 10)
    pdf["nid"] = 0
    pdf["fid"] = 0

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
        date = Time(pdf["jd"].values[0], format="jd").iso.split(" ")[0]
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
    logger = init_logging()
    startdate = datetime.datetime.strptime(date, "%Y-%m-%d")
    stopdate = startdate + datetime.timedelta(days=1)

    if verbose:  # pragma: no cover
        logger.info(
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


def get_last_roid_streaming_alert(
    config: configparser.ConfigParser,
    last_night: str,
    output_path: str,
    is_mpc: bool,
    verbose: bool = False,
    logger: LoggerNewLine = None,
):
    assert verbose and logger is not None, "logger is None while verbose is True"
    input_path = config["OUTPUT"]["roid_module_output"]
    year, month, day = last_night.split("-")
    input_path = os.path.join(
        input_path,
        f"year={year}",
        f"month={month}",
        f"day={day}",
    )

    mode = str(config["OUTPUT"]["roid_path_mode"])

    if mode == "local":
        if verbose:
            logger.info("start to get data in local mode")
        # load alerts from local
        sso_night = pd.read_parquet(input_path)
        if "candidate" in sso_night:
            candidate_pdf = pd.json_normalize(sso_night["candidate"]).drop(
                "candid", axis=1
            )
            sso_night = pd.concat(
                [sso_night, candidate_pdf],
                axis=1,
            )
    elif mode == "spark":
        if verbose:
            logger.info("start to get data in spark mode")
        output_path_spark = os.path.join(
            output_path,
            f"year={year}",
            f"month={month}",
            f"day={day}",
        )
        if not os.path.isdir(output_path_spark):
            pathlib.Path(output_path_spark).mkdir(parents=True)

        # load alerts from spark
        master_manager = config["SOLVE_ORBIT_PARAMS"]["manager"]
        principal_group = config["SOLVE_ORBIT_PARAMS"]["principal"]
        secret = config["SOLVE_ORBIT_PARAMS"]["secret"]
        role = config["SOLVE_ORBIT_PARAMS"]["role"]
        executor_env = config["SOLVE_ORBIT_PARAMS"]["exec_env"]
        driver_mem = config["SOLVE_ORBIT_PARAMS"]["driver_memory"]
        exec_mem = config["SOLVE_ORBIT_PARAMS"]["executor_memory"]
        max_core = config["SOLVE_ORBIT_PARAMS"]["max_core"]
        exec_core = config["SOLVE_ORBIT_PARAMS"]["executor_core"]

        application = os.path.join(
            os.path.dirname(fink_fat.__file__),
            "command_line",
            "association_cli.py prod",
        )

        application += " " + master_manager
        application += " " + input_path
        application += " " + output_path_spark
        application += " " + str(is_mpc)
        application += " " + year
        application += " " + month
        application += " " + day

        # FIXME
        # temporary dependencies (only during the performance test phase)
        FINK_FAT = "/home/roman.le-montagner/home_big_storage/Doctorat/Asteroids/fink-fat/dist/fink_fat-1.0.0-py3.9.egg"
        FINK_SCIENCE = "/home/roman.le-montagner/home_big_storage/Doctorat/fink-science/dist/fink_science-4.4-py3.7.egg"

        spark_submit = f"spark-submit \
            --master {master_manager} \
            --conf spark.mesos.principal={principal_group} \
            --conf spark.mesos.secret={secret} \
            --conf spark.mesos.role={role} \
            --conf spark.executorEnv.HOME={executor_env} \
            --driver-memory {driver_mem}G \
            --executor-memory {exec_mem}G \
            --conf spark.cores.max={max_core} \
            --conf spark.executor.cores={exec_core} \
            --conf spark.driver.maxResultSize=6G\
            --conf spark.sql.execution.arrow.pyspark.enabled=true\
            --conf spark.sql.execution.arrow.maxRecordsPerBatch=1000000\
            --conf spark.kryoserializer.buffer.max=512m\
            --py-files {FINK_FAT},{FINK_SCIENCE}\
            {application}"

        if verbose:
            logger.info("run recovering of data with spark")
        process = subprocess.run(spark_submit, shell=True)
        if process.returncode != 0:
            logger = init_logging()
            logger.info(process.stderr)
            logger.info(process.stdout)
            exit()

        if verbose:
            logger.info("data recovered from spark")
        read_path = os.path.join(output_path_spark, "tmp_ast.parquet")
        sso_night = pd.read_parquet(read_path)
        os.remove(read_path)

    else:
        raise ValueError(f"mode {mode} not exist")

    roid_pdf = pd.json_normalize(sso_night["ff_roid"])
    sso_night = pd.concat(
        [sso_night, roid_pdf],
        axis=1,
    )
    sso_night = sso_night.explode(["estimator_id", "ffdistnr"])
    cols_to_keep = [
        "objectId",
        "candid",
        "ra",
        "dec",
        "jd",
        "magpsf",
        "sigmapsf",
        "fid",
        "ssnamenr",
        "roid",
        "estimator_id",
        "ffdistnr",
    ]
    return sso_night[cols_to_keep]


def intro_reset():  # pragma: no cover
    logger = init_logging()
    logger.info("WARNING !!!")
    logger.info(
        "you will loose the trajectory done by previous association, Continue ? [Y/n]"
    )


def yes_reset(arguments, tr_df_path, obs_df_path):  # pragma: no cover
    logger = init_logging()
    if os.path.exists(tr_df_path) and os.path.exists(obs_df_path):
        logger.info("Removing files :\n\t{}\n\t{}".format(tr_df_path, obs_df_path))
        try:
            os.remove(tr_df_path)
            os.remove(obs_df_path)
        except OSError as e:
            if arguments["--verbose"]:
                logger.info("Failed with:", e.strerror)
                logger.info("Error code:", e.code)
    else:
        logger.info("File trajectory and old observations not exists.")

    dirname = os.path.dirname(tr_df_path)
    save_path = os.path.join(dirname, "save", "")
    if os.path.exists(save_path):
        shutil.rmtree(save_path)


def no_reset():  # pragma: no cover
    logger = init_logging()
    logger.info("Abort reset.")


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
        last_trajectory_id = np.max(trajectory_df["trajectory_id"])

    return trajectory_df, old_obs_df, last_trajectory_id


if __name__ == "__main__":  # pragma: no cover
    import sys

    if sys.argv[1] == "test":
        import doctest
        from pandas.testing import assert_frame_equal  # noqa: F401
        import fink_fat.test.test_sample as ts  # noqa: F401
        from unittest import TestCase  # noqa: F401

        if "unittest.util" in __import__("sys").modules:
            # Show full diff in self.assertEqual.
            __import__("sys").modules["unittest.util"]._MAX_LENGTH = 999999999

        sys.exit(doctest.testmod()[0])
    elif sys.argv[1] == "prod":
        from pyspark.sql import SparkSession

        logger = init_logging()
        master_adress = str(sys.argv[2])
        read_path = str(sys.argv[3])
        output_path = str(sys.argv[4])
        is_mpc = string_to_bool(str(sys.argv[5]))
        year = sys.argv[6]
        month = sys.argv[7]
        day = sys.argv[8]

        spark = (
            SparkSession.builder.master(master_adress)
            .appName(f"FINK-FAT_recover_stream_data_{year}{month}{day}")
            .getOrCreate()
        )
        df = spark.read.load(read_path)
        df = df.select(
            "objectId",
            "candid",
            "candidate.ra",
            "candidate.dec",
            "candidate.jd",
            "candidate.magpsf",
            "candidate.sigmapsf",
            "candidate.fid",
            "candidate.ssnamenr",
            "ff_roid",
        )
        roid_flag = [3, 4, 5] if is_mpc else [1, 2, 4, 5]
        df = df.filter(col("ff_roid.roid").isin(roid_flag))
        df_local = df.toPandas()
        df_local.to_parquet(os.path.join(output_path, "tmp_ast.parquet"), index=False)
