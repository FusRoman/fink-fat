import subprocess
from fink_fat.others.utils import init_logging
from fink_fat.command_line.fink_fat_cli import main_test
from datetime import date, timedelta
import os
import pandas as pd
import numpy as np


def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)


def run_roid(year, month, day):
    process = subprocess.run(
        f"spark-submit fink_fat/test/cli_test/run_roid.py {year} {month} {day}",
        shell=True,
    )

    if process.returncode != 0:
        logger = init_logging()
        logger.error(process.stderr)
        logger.info(process.stdout)
        exit()


if __name__ == "__main__":
    data_test_path = "fink_fat/test/cli_test/fink_fat_out_test/"
    data_current_path = "fink_fat/test/cli_test/fink_fat_out/"
    datapath = "fink_fat/test/cli_test/small_sso_dataset"

    start_date = date(2021, 6, 1)
    end_date = date(2021, 6, 15)
    logger = init_logging()

    # test successive call of fink-fat 1.0 as Fink normal operation
    # between first of june and 15 of june
    logger.info("TEST SUCCESSIVE FINK-FAT FITROID CALL")
    logger.newline()
    for sso_date in daterange(start_date, end_date):
        logger.info(f"start to process date {sso_date}")
        logger.newline()

        path_sso = os.path.join(
            datapath,
            f"year={sso_date.year:04d}/month={sso_date.month:02d}/day={sso_date.day:02d}",
        )
        if os.path.exists(path_sso):
            # --------------------------------------------------------------------------------#
            # roid runner (fink science side)
            logger.info("start roid processor")
            run_roid(
                sso_date.year,
                sso_date.month,
                sso_date.day,
            )

            # --------------------------------------------------------------------------------#
            # fink fat command line association
            main_test(
                [
                    "associations",
                    "fitroid",
                    "--night",
                    f"{sso_date.year:04d}-{sso_date.month:02d}-{sso_date.day:02d}",
                    "--config",
                    "fink_fat/test/cli_test/fitroid_test.conf",
                    "--verbose",
                ]
            )
    
    logger.info("TEST FINK-FAT OFFLINE FITROID")
    logger.newline()
    # --------------------------------------------------------------------------------#
    # fink fat offline command line association
    # test fink-fat 1.0 offline mode
    main_test(
        [
            "offline",
            "fitroid",
            "2021-06-30",
            "--config",
            "fink_fat/test/cli_test/fitroid_test.conf",
            "--verbose",
        ]
    )


    true_traj = pd.read_parquet("fink_fat/test/cli_test/small_sso_dataset/")
    tr_orb = pd.read_parquet(
        "fink_fat/test/cli_test/fink_fat_out/fitroid/trajectory_orb.parquet"
    )

    gb = tr_orb.groupby("ssoCandId").agg(
        sso_id=("ssnamenr", "unique"),
        nb_uniq_id=("ssnamenr", lambda x: len(np.unique(x))),
    )
    assert (gb["nb_uniq_id"] == 1).all()

    print(
        f"number of trajectories in the input dataset: {len(true_traj['candidate'].str['ssnamenr'].unique())}"
    )
    print(f"number of reconstructed trajectories: {len(tr_orb['ssnamenr'].unique())}")
