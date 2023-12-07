from pyspark.sql import SparkSession
import os
from fink_fat import __file__


def add_roid_datatest(spark: SparkSession):
    """
    Load the files used for the roid test

    Parameters
    ----------
    spark : SparkSession
        spark session
    is_processor : bool, optional
        if True, load the test file for the processor, by default False
    """
    path = os.path.dirname(__file__)

    orbit_sample = "file://{}/test/streaming_data_test/orbital.parquet".format(path)
    kalman_sample = "file://{}/test/streaming_data_test/kalman.pkl".format(path)
    fitroid_sample = "file://{}/test/streaming_data_test/fit_roid.parquet".format(path)

    spark.sparkContext.addFile(orbit_sample)
    spark.sparkContext.addFile(kalman_sample)
    spark.sparkContext.addFile(fitroid_sample)
