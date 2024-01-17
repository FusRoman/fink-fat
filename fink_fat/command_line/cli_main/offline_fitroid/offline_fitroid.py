import datetime
import os
import subprocess
from fink_fat.others.utils import LoggerNewLine
import fink_fat
import configparser
import pathlib


def offline_fitroid(
    config: configparser.ConfigParser,
    path_config: str,
    start_date: datetime,
    end_date: datetime,
    logger: LoggerNewLine,
    verbose: bool,
):
    if verbose:
        logger.info(
            f"""
 --- START FITROID OFFLINE ---
 start date: {start_date}
 end date: {end_date}
    """
        )
        logger.newline()

    ff_path = os.path.dirname(fink_fat.__file__)
    offline_path = os.path.join(ff_path, "command_line", "cli_main", "offline_fitroid")

    log_path = config["OFFLINE"]["log_path"]
    if not os.path.isdir(log_path):
        pathlib.Path(log_path).mkdir(parents=True)

    proc = subprocess.run(
        os.path.join(
            offline_path,
            f"run_offline_fitroid.sh {str(start_date)} {str(end_date)} {offline_path} {path_config} {log_path} {verbose}",
        ),
        shell=True,
    )
    return
