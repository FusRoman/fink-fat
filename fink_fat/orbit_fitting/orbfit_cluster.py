import os
import shutil
import sys
import numpy as np
import pandas as pd

from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import ArrayType, DoubleType
from pyspark.sql import functions as F
from pyspark.sql import SparkSession

import fink_fat.orbit_fitting.orbfit_files as of
import fink_fat.orbit_fitting.orbfit_local as ol
import fink_fat.orbit_fitting.mpcobs_files as mf


def orbit_wrapper(
    ra,
    dec,
    dcmag,
    band,
    date,
    traj_id,
    ram_dir,
    n_triplets,
    noise_ntrials,
    prop_epoch=None,
    verbose=1,
):
    """
    Computation of OrbFit in cluster mode.

    Parameters
    ----------
    ra : spark dataframe columns
        Right ascension columns of the observations
    dec : spark dataframe columns
        Declination columns of the observations
    dcmag : spark dataframe columns
        The apparent magnitude computed with the dc_mag function in fink_science.conversion
    band : spark dataframe columns
        The filter used during the observations
    date : spark dataframe columns
        The observations date in jd
    traj_id : spark dataframe columns
        The trajectory_id of each trajectories
    ram_dir : string
        the path where to write the file
    n_triplets : integer
        max number of triplets of observations to be tried for the initial orbit determination
    noise_ntrials : integer
        number of trials for each triplet for the initial orbit determination
    prop_epoch : float
        Epoch at which output orbital elements in JD.
    verbose : integer
        Verbosity levels of Orbfit
        1 = summary information on the solution found
        2 = summary information on all trials
        3 = debug

    Return
    ------
    res : Series
        The orbital elements and their RMS computed by orbfit for each trajectories.


    Examples
    --------
    >>> sparkDF = spark.read.format('parquet').load(traj_sample)
    >>> spark_gb = (
    ... sparkDF.groupby("trajectory_id")
    ... .agg(
    ...     F.sort_array(
    ...             F.collect_list(F.struct("jd", "ra", "dec", "fid", "dcmag"))
    ...         ).alias("collected_list")
    ...     )
    ...     .withColumn("ra", F.col("collected_list.ra"))
    ...     .withColumn("dec", F.col("collected_list.dec"))
    ...     .withColumn("fid", F.col("collected_list.fid"))
    ...     .withColumn("dcmag", F.col("collected_list.dcmag"))
    ...     .withColumn("jd", F.col("collected_list.jd"))
    ...     .drop("collected_list")
    ... )


    >>> spark_column = spark_gb.withColumn(
    ...     "orbital_elements",
    ...     orbit_wrapper(
    ...         spark_gb.ra,
    ...         spark_gb.dec,
    ...         spark_gb.dcmag,
    ...         spark_gb.fid,
    ...         spark_gb.jd,
    ...         spark_gb.trajectory_id,
    ...         "",
    ...         30, 20
    ...     ),
    ... )

    >>> orb_pdf = spark_column.toPandas()

    >>> orbital_columns = [
    ... "ref_epoch",
    ... "a",
    ... "e",
    ... "i",
    ... "long. node",
    ... "arg. peric",
    ... "mean anomaly",
    ... "rms_a",
    ... "rms_e",
    ... "rms_i",
    ... "rms_long. node",
    ... "rms_arg. peric",
    ... "rms_mean anomaly",
    ... "chi_reduced",
    ... ]

    >>> split_df = pd.DataFrame(
    ... orb_pdf["orbital_elements"].tolist(), columns=orbital_columns
    ... )

    >>> orbit_results = pd.concat([orb_pdf["trajectory_id"], split_df], axis=1)
    >>> orbit_test = pd.read_parquet("fink_fat/test/cluster_test/res_orb_cluster.parquet")
    
    >>> assert_frame_equal(orbit_results.round(decimals=5), orbit_test.round(decimals=5))
    """

    @pandas_udf(ArrayType(DoubleType()))  # noqa: F405
    def get_orbit_element(ra, dec, dcmag, band, date, traj_id):  # pragma: no cover
        _pid = os.getpid()
        current_ram_path = os.path.join(ram_dir, str(_pid), "")
        if not os.path.isdir(current_ram_path):
            os.mkdir(current_ram_path)

        of.prep_orbitfit(current_ram_path)

        res = []
        for c_ra, c_dec, c_dcmag, c_band, c_date, c_traj_id in zip(
            ra, dec, dcmag, band, date, traj_id
        ):
            df_tmp_traj = pd.DataFrame(
                {
                    "trajectory_id": np.ones(len(c_ra), dtype=int) * c_traj_id,
                    "ra": c_ra,
                    "dec": c_dec,
                    "dcmag": c_dcmag,
                    "fid": c_band,
                    "jd": c_date,
                }
            )

            prov_desig = mf.write_observation_file(current_ram_path, df_tmp_traj)
            of.write_inp(current_ram_path, prov_desig)

            if prop_epoch is None:
                of.write_oop(
                    current_ram_path,
                    prov_desig,
                    prop_epoch="JD  {} UTC".format(df_tmp_traj["jd"].values[-1]),
                    n_triplets=n_triplets,
                    noise_ntrials=noise_ntrials,
                    verbose=verbose,
                )
            else:
                of.write_oop(
                    current_ram_path,
                    prov_desig,
                    prop_epoch="JD  {} UTC".format(prop_epoch),
                    n_triplets=n_triplets,
                    noise_ntrials=noise_ntrials,
                    verbose=verbose,
                )

            ol.call_orbitfit(current_ram_path, prov_desig)
            orb_elem = of.read_oel(current_ram_path, prov_desig)

            chi_values = of.read_rwo(current_ram_path, prov_desig, len(c_ra))
            # reduced the chi values
            chi_reduced = np.sum(np.array(chi_values)) / len(c_ra)

            res.append(orb_elem + [chi_reduced])

            of.obs_clean(current_ram_path, prov_desig)

        of.final_clean(current_ram_path)
        shutil.rmtree(current_ram_path)
        res = [[float(el) for el in i] for i in res]
        return pd.Series(res)

    return get_orbit_element(ra, dec, dcmag, band, date, traj_id)


if __name__ == "__main__":
    if sys.argv[1] == "test":
        from fink_science.tester import spark_unit_tests
        from pandas.testing import assert_frame_equal  # noqa: F401

        globs = globals()

        traj_sample = "fink_fat/test/cluster_test/traj_sample.parquet"
        globs["traj_sample"] = traj_sample

        # Run the test suite
        spark_unit_tests(globs)
    elif sys.argv[1] == "prod":

        master_adress = sys.argv[2]
        ram_dir = sys.argv[3]
        n_triplets = sys.argv[4]
        noise_ntrials = sys.argv[5]
        prop_epoch = sys.argv[6]

        spark = spark = (
            SparkSession.builder.master(master_adress)
            .appName("Fink-FAT_solve_orbit")
            .getOrCreate()
        )

        # read the input from local parquet file
        traj_df = pd.read_parquet("tmp_traj.parquet")
        # transform the local pandas dataframe into a spark dataframe
        sparkDF = spark.createDataFrame(traj_df)

        spark_gb = (
            sparkDF.groupby("trajectory_id")
            .agg(
                F.sort_array(
                    F.collect_list(F.struct("jd", "ra", "dec", "fid", "dcmag"))
                ).alias("collected_list")
            )
            .withColumn("ra", F.col("collected_list.ra"))
            .withColumn("dec", F.col("collected_list.dec"))
            .withColumn("fid", F.col("collected_list.fid"))
            .withColumn("dcmag", F.col("collected_list.dcmag"))
            .withColumn("jd", F.col("collected_list.jd"))
            .drop("collected_list")
        )

        max_core = int(dict(spark.sparkContext.getConf().getAll())["spark.cores.max"])
        spark_gb = spark_gb.repartition(max_core * 100)

        print("begin compute orbital elem on spark")
        spark_column = spark_gb.withColumn(
            "orbital_elements",
            orbit_wrapper(
                spark_gb.ra,
                spark_gb.dec,
                spark_gb.dcmag,
                spark_gb.fid,
                spark_gb.jd,
                spark_gb.trajectory_id,
                ram_dir,
                n_triplets,
                noise_ntrials,
                prop_epoch,
                verbose=3,
            ),
        )

        orb_pdf = spark_column.toPandas()
        orb_pdf.to_parquet("res_orb.parquet")
        sys.exit(0)
