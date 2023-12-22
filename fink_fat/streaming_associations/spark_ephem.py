import configparser
import os
import subprocess

import fink_fat
from fink_fat.others.utils import init_logging

def launch_spark_ephem(
    config: configparser.ConfigParser,
    orbital_path: str,
    ephem_output_path: str,
    year: str,
    month: str,
    day: str,
    verbose: bool = False,
):
    """
    Launch the spark job computing the ephemeries

    Parameters
    ----------
    config : configparser.ConfigParser
        fink-fat configuration
    orbital_path : str
        path where are stored the orbits
    ephem_output_path : str
        path where the ephemerides will be stored
    year : str
        current processing year, ephemerides will be generate the day after
    month : str
        current processing month, ephemerides will be generate the day after
    day : str
        current processing day, ephemerides will be generate the day after

    Examples
    --------
    >>> from fink_fat.command_line.utils_cli import init_cli
    >>> import pandas as pd
    >>> config, _ = init_cli({"--config": ""})

    >>> launch_spark_ephem(
    ... config,
    ... "fink_fat/test/ephem_test/distribute_ephem_test.parquet",
    ... "ephem.parquet",
    ... "2023", "08", "12", False
    ... )

    >>> pd.read_parquet("ephem.parquet").sort_values(["epoch_jd"])
                  ssoCandId          RA        DEC      epoch_jd
    0     FF20230801aaaadar   92.166557  24.420229  2.460170e+06
    1440  FF20230809aaaadbj  230.341629  -1.378862  2.460170e+06
    1441  FF20230809aaaadbj  230.341705  -1.378949  2.460170e+06
    1     FF20230801aaaadar   92.168213  24.420206  2.460170e+06
    2     FF20230801aaaadar   92.169868  24.420185  2.460170e+06
    ...                 ...         ...        ...           ...
    1437  FF20230801aaaadar   94.647072  24.447813  2.460170e+06
    2878  FF20230809aaaadbj  230.463974  -1.503022  2.460170e+06
    1438  FF20230801aaaadar   94.648766  24.447826  2.460170e+06
    1439  FF20230801aaaadar   94.650459  24.447840  2.460170e+06
    2879  FF20230809aaaadbj  230.464050  -1.503109  2.460170e+06
    <BLANKLINE>
    [2880 rows x 4 columns]

    >>> os.remove("ephem.parquet")
    """
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
        "others",
        "spark_ephem_utils.py",
    )

    application += " " + orbital_path
    application += " " + ephem_output_path
    application += " " + config["TW_PARAMS"]["orbit_keep_limit"]
    application += " " + config["ASSOC_PARAMS"]["ephem_obs_site"]
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

    process = subprocess.run(spark_submit, shell=True, capture_output=True)
    logger = init_logging()
    if verbose:
        logger.info(
            process.stdout.decode("utf-8")
            if process.stdout is not None
            else "no std output"
        )
    if process.returncode != 0:
        logger.error(
            f"ephem spark generation exited with a non-zero status code, status_code={process.returncode}"
        )
        logger.newline(2)
        logger.info(
            process.stdout.decode("utf-8")
            if process.stdout is not None
            else "no std output"
        )
        logger.newline(2)
        logger.info(
            process.stderr.decode("utf-8")
            if process.stderr is not None
            else "no err output"
        )
        logger.newline(2)
        exit()


if __name__ == "__main__":
    """Execute the test suite"""
    from fink_science.tester import spark_unit_tests

    globs = globals()

    # Run the test suite
    spark_unit_tests(globs)
