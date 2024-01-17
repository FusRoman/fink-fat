from pyspark.sql import functions as F
import os
import sys
import time as t

from fink_utils.broker.sparkUtils import init_sparksession
from fink_utils.spark.utils import concat_col

from fink_science.asteroids.processor import roid_catcher

from fink_fat.command_line.utils_cli import init_cli
from pyspark.sql import SparkSession


def addFileToSpark(
    spark: SparkSession, fitroid_path: str, orbit_path: str, ephem_path: str
):
    if os.path.exists(orbit_path):
        spark.sparkContext.addFile(orbit_path)
    if os.path.exists(fitroid_path):
        spark.sparkContext.addFile(fitroid_path)
    if os.path.exists(ephem_path):
        spark.sparkContext.addFile(ephem_path)


if __name__ == "__main__":
    year, month, day, path_config = sys.argv[1:]
    config, output_path = init_cli({"--config": path_config})

    path_orbit = os.path.join(output_path, "fitroid", "orbital.parquet")
    path_fit_roid = os.path.join(output_path, "fitroid", "fit_roid.parquet")
    path_ephem = os.path.join(output_path, "fitroid", "ephem.parquet")

    datapath = config["OFFLINE"]["stream_datapath"]
    path_sso = os.path.join(
        datapath, f"year={int(year):04d}/month={int(month):02d}/day={int(day):02d}"
    )

    spark = init_sparksession(
        f"FINK-FAT_roid_module_{int(year):04d}{int(month):02d}{int(day):02d}"
    )
    addFileToSpark(spark, path_fit_roid, path_orbit, path_ephem)

    userschema = spark.read.parquet(path_sso).schema

    df = (
        spark.read.format("parquet")
        .schema(userschema)
        .option("basePath", path_sso)
        .option("path", path_sso)
        .load()
    )

    what = ["jd", "magpsf"]
    prefix = "c"
    what_prefix = [prefix + i for i in what]
    for colname in what:
        df = concat_col(df, colname, prefix=prefix)

    args = [
        "candidate.ra",
        "candidate.dec",
        "candidate.jd",
        "candidate.magpsf",
        "candidate.candid",
        "cjd",
        "cmagpsf",
        "candidate.fid",
        "candidate.ndethist",
        "candidate.sgscore1",
        "candidate.ssdistnr",
        "candidate.distpsnr1",
        F.lit(float(config["ASSOC_PARAMS"]["error_radius"])),
        F.lit(float(config["ASSOC_PARAMS"]["inter_night_magdiff_limit_same_fid"])),
        F.lit(float(config["ASSOC_PARAMS"]["inter_night_magdiff_limit_diff_fid"])),
        F.lit(int(config["TW_PARAMS"]["orbit_keep_limit"])),
        F.lit(float(config["ASSOC_PARAMS"]["orbit_assoc_radius"])),
        F.lit(True),
    ]
    df = df.withColumn("ff_roid", roid_catcher(*args))
    df = df.drop(*what_prefix)

    df = df.select(
        [
            col
            for col in df.columns
            if col not in ["cutoutScience", "cutoutDifference", "cutoutTemplate"]
        ]
    )

    t_before = t.time()
    df.write.mode("ignore").parquet(
        os.path.join(
            config["OUTPUT"]["roid_module_output"],
            f"year={int(year):04d}",
            f"month={int(month):02d}",
            f"day={int(day):02d}",
        )
    )
    print(f"ROID SCIENCE MODULE ELAPSED TIME: {t.time() - t_before:.5f}")
