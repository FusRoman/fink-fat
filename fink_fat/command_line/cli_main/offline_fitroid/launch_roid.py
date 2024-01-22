if __name__ == "__main__":
    import os
    import sys
    from fink_fat.others.utils import init_logging
    from fink_fat.command_line.utils_cli import init_cli
    import fink_fat.others.launch_spark as spark

    logger = init_logging()

    year, month, day = sys.argv[1], sys.argv[2], sys.argv[3]
    path_offline = sys.argv[4]
    path_config = sys.argv[5]
    verbose = sys.argv[6]

    config, output_path = init_cli({"--config": path_config})

    application = os.path.join(
        path_offline,
        "run_roid.py",
    )

    application += " " + year
    application += " " + month
    application += " " + day
    application += " " + path_config

    spark_submit = spark.build_spark_submit(config)
    spark_app = spark.spark_submit_application(spark_submit, application)
    process = spark.run_spark_submit(spark_app, verbose)

    if process.returncode != 0:
        logger = init_logging()
        logger.error(f"""
    Offline launch roid spark_submit exited with a non-zero return code: {process.returncode}
""")
        logger.info(process.stderr)
        logger.info(process.stdout)
