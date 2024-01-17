# YEAR=$1
# MONTH=$2
# DAY=$3

# spark-submit \
#     --master mesos://vm-75063.lal.in2p3.fr:5050 \
#     --conf spark.mesos.principal=lsst \
#     --conf spark.mesos.secret=secret \
#     --conf spark.mesos.role=lsst \
#     --conf spark.executorEnv.HOME='/home/roman.le-montagner'\
#     --conf spark.driver.maxResultSize=6G\
#     --conf spark.sql.execution.arrow.pyspark.enabled=true\
#     --conf spark.sql.execution.arrow.maxRecordsPerBatch=1000000\
#     --conf spark.kryoserializer.buffer.max=512m\
#     --driver-memory 6G --executor-memory 4G --conf spark.cores.max=16 --conf spark.executor.cores=2\
#     --py-files ${FINK_FAT},${FINK_SCIENCE}\
#     $4/run_roid.py $YEAR $MONTH $DAY


if __name__ == "__main__":
    import os
    import sys
    import subprocess
    from fink_fat.others.utils import init_logging
    from fink_fat.command_line.utils_cli import init_cli

    logger = init_logging()

    year, month, day = sys.argv[1], sys.argv[2], sys.argv[3]
    path_offline = sys.argv[4]
    path_config=sys.argv[5]
    verbose = sys.argv[6]

    config, output_path = init_cli(
        {"--config": path_config}
    )

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
        path_offline,
        "run_roid.py",
    )

    application += " " + year
    application += " " + month
    application += " " + day
    application += " " + path_config

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
        {application}"

    # --py-files {FINK_FAT},{FINK_SCIENCE}\

    if verbose:
        logger.info("run recovering of data with spark")
    process = subprocess.run(spark_submit, shell=True)
    if process.returncode != 0:
        logger = init_logging()
        logger.info(process.stderr)
        logger.info(process.stdout)
