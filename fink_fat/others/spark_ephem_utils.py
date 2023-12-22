import numpy as np
import pandas as pd
from astropy.time import Time

from pyspark.sql.functions import pandas_udf, lit
from pyspark.sql.types import ArrayType, FloatType, StructField, StructType, DoubleType
from fink_fat.streaming_associations.orbit_assoc import compute_ephem
from fink_utils.broker.sparkUtils import init_sparksession


ephem_schema = StructType(
    [
        StructField("RA", ArrayType(FloatType()), True),
        StructField("DEC", ArrayType(FloatType()), True),
        StructField("epoch_jd", ArrayType(DoubleType()), True)
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
    """
    Compute the ephemerides of orbits in a distributed context

    Parameters
    ----------
    a : spark columns
        semi major axis
    e : spark columns
        eccentricity
    i : spark columns
        inclination
    long_node : spark columns
        longitude of ascending node
    arg_peric : spark columns
        argument of perihelion
    mean_anomaly : spark columns
        mean anomaly
    ref_epoch : spark columns
        orbit reference epoch
    ssoCandId : spark columns
        fink-fat trajectories identifier
    start_time : spark columns
        start ephemeries date
    end_time : spark columns
        end ephemeries date
    step : spark columns
        step between the ephemeries
    location : spark columns
        mpc code observer location, ZTF is I41 as an example

    Returns
    -------
    pd.dataframe
        dataframe containing the ephemeries
    """
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

    cols_to_keep = ["targetname", "RA", "DEC", "epoch"]
    ephem_pdf = compute_ephem(orbit_pdf, time_ephem, location)[cols_to_keep]
    ephem_pdf["epoch_jd"] = Time(ephem_pdf["epoch"].values).jd
    ephem_pdf = ephem_pdf.drop("epoch", axis=1)
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
    """
    Compute the ephemerides with spark

    Parameters
    ----------
    orbital_path : str
        path where are stored the orbits
    ephem_output_path : str
        path where the ephemerides will be stored
    start_ephem : str
        start ephemeries date
    stop_ephem : str
        end ephemeries date
    step : float
        step between the ephemeries
    location : str
        mpc code observer location, ZTF is I41 as an example
    year : str
        current processing year, used for the spark job name
    month : str
        current processing month, used for the spark job name
    day : str
        current processing day, used for the spark job name
    """
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
            ephem_spark["ephem"]["epoch_jd"].alias("epoch_jd"),
        ]
    )

    local_ephem = ephem_spark.toPandas()
    local_ephem = local_ephem.explode(["RA", "DEC", "epoch_jd"])

    local_ephem.to_parquet(ephem_output_path, index=False)

if __name__ == "__main__":
    import sys

    year, month, day = (
        sys.argv[4],
        sys.argv[5],
        sys.argv[6],
    )
    start_time = Time(f"{year}-{month}-{day} 02:00:00").jd
    stop_time = Time(f"{year}-{month}-{day} 14:00:00").jd

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
