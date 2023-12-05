import subprocess
from fink_fat.others.utils import init_logging
from fink_fat.command_line.fink_fat_cli import main_test
from datetime import date, timedelta
import os

def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)

def run_roid(year, month, day):
    process = subprocess.run(
        f"spark-submit fink_fat/test/cli_test/run_roid.py {year} {month} {day}", shell=True
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
    end_date = date(2021, 6, 30)
    logger = init_logging()

    i = 0
    max = 3

    for sso_date in daterange(start_date, end_date):

        logger.info(f"start to process date {sso_date}")
        logger.newline()

        path_sso = os.path.join(
            datapath, f"year={sso_date.year:04d}/month={sso_date.month:02d}/day={sso_date.day:02d}"
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
        
        i += 1
        if i == max:
            break
