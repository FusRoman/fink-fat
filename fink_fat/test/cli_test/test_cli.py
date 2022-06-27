import pandas as pd
from pandas.testing import assert_frame_equal
import shutil
import sys

import traceback
import logging

from bin.fink_fat_cli import main_test


if __name__ == "__main__":
    data_test_path = "fink_fat/test/cli_test/fink_fat_out_test/"
    data_current_path = "fink_fat/test/cli_test/fink_fat_out/"

    main_test(
        [
            "associations",
            "mpc",
            "--night",
            "2020-05-04",
            "--config",
            "fink_fat/test/cli_test/test.conf",
            "--verbose",
        ]
    )

    main_test(
        [
            "solve_orbit",
            "mpc",
            "local",
            "--config",
            "fink_fat/test/cli_test/test.conf",
            "--verbose",
        ]
    )

    main_test(
        [
            "associations",
            "mpc",
            "--night",
            "2020-05-05",
            "--config",
            "fink_fat/test/cli_test/test.conf",
            "--verbose",
        ]
    )

    main_test(
        [
            "solve_orbit",
            "mpc",
            "local",
            "--config",
            "fink_fat/test/cli_test/test.conf",
            "--verbose",
        ]
    )

    old_obs = pd.read_parquet("{}mpc/old_obs.parquet".format(data_current_path))
    trajectory_df = pd.read_parquet(
        "{}mpc/trajectory_df.parquet".format(data_current_path)
    )
    orb = pd.read_parquet("{}mpc/orbital.parquet".format(data_current_path))
    obs_orb = pd.read_parquet("{}mpc/trajectory_orb.parquet".format(data_current_path))

    old_obs_test = pd.read_parquet("{}mpc/old_obs.parquet".format(data_test_path))
    trajectory_df_test = pd.read_parquet(
        "{}mpc/trajectory_df.parquet".format(data_test_path)
    )
    orb_test = pd.read_parquet("{}mpc/orbital.parquet".format(data_test_path))
    obs_orb_test = pd.read_parquet(
        "{}mpc/trajectory_orb.parquet".format(data_test_path)
    )

    try:
        assert_frame_equal(old_obs, old_obs_test)
        assert_frame_equal(trajectory_df, trajectory_df_test)
        assert_frame_equal(orb_test, orb)
        assert_frame_equal(obs_orb_test, obs_orb)

        shutil.rmtree("fink_fat/test/cli_test/fink_fat_out")

        sys.exit(0)

    except Exception as e:  # pragma: no cover
        print(e)
        print()
        logging.error(traceback.format_exc())
        print()
        shutil.rmtree("fink_fat/test/cli_test/fink_fat_out")
        sys.exit(1)
