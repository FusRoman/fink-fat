import configparser
import subprocess

from fink_fat.others.utils import init_logging


def build_spark_submit(
    config: configparser.ConfigParser
) -> str:
    """
    Build the spark-submit application command

    Parameters
    ----------
    config : configparser.ConfigParser
        the config entries from the config file

    Returns
    -------
    str
        spark-submit command without the application
    """
    master_manager = config["SPARK"]["manager"]
    principal_group = config["SPARK"]["principal"]
    secret = config["SPARK"]["secret"]
    role = config["SPARK"]["role"]
    executor_env = config["SPARK"]["exec_env"]
    driver_mem = config["SPARK"]["driver_memory"]
    exec_mem = config["SPARK"]["executor_memory"]
    max_core = config["SPARK"]["max_core"]
    exec_core = config["SPARK"]["executor_core"]
    py_files = config["SPARK"]["py_files"]

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
    --conf spark.kryoserializer.buffer.max=512m"

    if py_files != "":
        spark_submit += f" --py-files {py_files}"

    # add the orbfit path as an environment variable in the spark env
    orbfit_path = config["SOLVE_ORBIT_PARAMS"]["orbfit_path"]
    if orbfit_path != "":
        spark_submit += f" --conf spark.executorEnv.ORBFIT_HOME={orbfit_path}"

    return spark_submit


def spark_submit_application(spark_cmd: str, application: str) -> str:
    """
    Merge the spark submit command with the application

    Parameters
    ----------
    spark_cmd : str
        spark submit command
    application : str
        application with the parameters

    Returns
    -------
    str
        spark submit command concatenate with the application
    """
    return f"{spark_cmd} {application}"


def run_spark_submit(spark_cmd: str, verbose: bool) -> subprocess.CompletedProcess:
    """
    Run the spark_submit command within a subprocess

    Parameters
    ----------
    spark_cmd : str
        spark submit command
    verbose : bool
        if true, print logs

    Returns
    -------
    subprocess.CompletedProcess
        the process running the spark command
    """
    logger = init_logging()
    if verbose:
        logger.info(
            f"""
                    run a spark_submit command

                    cmd:
                    {spark_cmd}
                """
        )
    process = subprocess.run(spark_cmd, shell=True)

    return process
