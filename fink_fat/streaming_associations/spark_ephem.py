import numpy as np
import pandas as pd
import configparser
from astropy.time import Time
import os
import subprocess

import fink_fat
from pyspark.sql.functions import pandas_udf, lit
from pyspark.sql.types import ArrayType, FloatType, StructField, StructType
from fink_fat.streaming_associations.orbit_assoc import compute_ephem
from fink_utils.broker.sparkUtils import init_sparksession
from fink_fat.others.utils import init_logging

ephem_schema = StructType(
    [
        StructField("RA", ArrayType(FloatType()), True),
        StructField("DEC", ArrayType(FloatType()), True),
    ]
)


@pandas_udf(ephem_schema)
def distribute_ephem(
    a,
    e,
    i,
    long_node,
    arg_peric,
    mean_anomaly,
    ref_epoch,
    ssoCandId,
    start_time,
    end_time,
    step,
    location,
):
    orbit_pdf = pd.DataFrame(
        {
            "a": a,
            "e": e,
            "i": i,
            "long. node": long_node,
            "arg. peric": arg_peric,
            "mean anomaly": mean_anomaly,
            "ref_epoch": ref_epoch,
            "ssoCandId": ssoCandId,
        }
    )

    time_ephem = np.arange(
        Time(start_time.values[0]).jd, Time(end_time.values[0]).jd, step.values[0]
    )

    cols_to_keep = ["targetname", "RA", "DEC"]
    ephem_pdf = compute_ephem(orbit_pdf, time_ephem, location)[cols_to_keep]
    ephem_gb = ephem_pdf.groupby("targetname").agg(list)

    return ephem_gb


def spark_ephem(
    orbital_path: str,
    ephem_output_path: str,
    start_ephem: str,
    stop_ephem: str,
    step: float,
    location: str,
    year: str,
    month: str,
    day: str,
):
    orbit_pdf = pd.read_parquet(orbital_path)

    spark = init_sparksession(
        f"FINK-FAT_ephemerides_{int(year):04d}{int(month):02d}{int(day):02d}"
    )

    orbit_pdf = orbit_pdf.rename(
        {
            "long. node": "long_node",
            "arg. peric": "arg_peric",
        },
        axis=1,
    )

    orbit_spark = spark.createDataFrame(orbit_pdf)

    ephem_spark = orbit_spark.withColumn(
        "ephem",
        distribute_ephem(
            "a",
            "e",
            "i",
            "long_node",
            "arg_peric",
            "mean anomaly",
            "ref_epoch",
            "ssoCandId",
            lit(start_ephem),
            lit(stop_ephem),
            lit(step),
            lit(location),
        ),
    )

    ephem_spark = ephem_spark.select(
        [
            "ssoCandId",
            ephem_spark["ephem"]["RA"].alias("RA"),
            ephem_spark["ephem"]["DEC"].alias("DEC"),
        ]
    )

    local_ephem = ephem_spark.toPandas()
    local_ephem = local_ephem.explode(["RA", "DEC"])

    local_ephem.to_parquet(ephem_output_path, index=False)


def launch_spark_ephem(
    config: configparser.ConfigParser,
    orbital_path: str,
    ephem_output_path: str,
    year: str,
    month: str,
    day: str,
):
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
        "streaming_associations",
        "spark_ephem.py",
    )

    application += " " + orbital_path
    application += " " + ephem_output_path
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

    process = subprocess.run(spark_submit, shell=True)
    if process.returncode != 0:
        logger = init_logging()
        logger.info(
            process.stderr.decode("utf-8")
            if process.stderr is not None
            else "no err output"
        )
        logger.info(
            process.stdout.decode("utf-8")
            if process.stdout is not None
            else "no std output"
        )
        exit()


if __name__ == "__main__":
    import sys

    year, month, day = (
        sys.argv[4],
        sys.argv[5],
        sys.argv[6],
    )
    start_time = Time(f"{year}-{month}-{day} 03:00:00").jd
    stop_time = Time(f"{year}-{month}-{day} 12:50:00").jd

    start_time = Time(start_time + 1, format="jd").iso
    stop_time = Time(stop_time + 1, format="jd").iso

    spark_ephem(
        sys.argv[1],
        sys.argv[2],
        start_time,
        stop_time,
        30 / 24 / 3600,
        sys.argv[3],
        year,
        month,
        day,
    )
