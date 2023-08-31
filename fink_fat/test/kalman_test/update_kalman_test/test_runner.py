import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal
from fink_fat.associations.association_kalman import kalman_association

from fink_fat.test.kalman_test.update_kalman_test import data as d


def assert_test(
    new_traj: pd.DataFrame,
    new_kalman: pd.DataFrame,
    result_traj: pd.DataFrame,
    result_kalman: pd.DataFrame,
):
    assert len(new_traj["trajectory_id"].unique()) == len(new_kalman)
    assert len(new_kalman) == len(new_kalman["trajectory_id"].unique())

    test_id = np.array(
        [
            kf.kf_id == tr_id
            for kf, tr_id in zip(new_kalman["kalman"], new_kalman["trajectory_id"])
        ]
    )
    assert test_id.all()

    assert_frame_equal(
        new_traj.fillna(-1.0).reset_index(drop=True),
        result_traj.fillna(-1.0).reset_index(drop=True),
        check_index_type=False,
        check_dtype=False,
    )

    assert_frame_equal(
        new_kalman.fillna(-1.0)[new_kalman.columns[:-1]].reset_index(drop=True),
        result_kalman.fillna(-1.0).reset_index(drop=True),
        check_index_type=False,
        check_dtype=False,
    )


def aux_test_runner(f):
    trajectory_df, kalman_pdf, new_alerts = f()
    new_traj, new_kalman = kalman_association(
        trajectory_df, kalman_pdf, new_alerts, True
    )

    # print("   FUNCTION RESULTS   ")
    # print(new_traj.fillna(-1.0))
    # print()
    # print()
    # print(new_kalman.fillna(-1.0)[new_kalman.columns[:-1]])
    # print("####")

    path_start = "fink_fat/test/kalman_test/update_kalman_test"
    # new_traj.to_parquet(
    #     f"{path_start}/data_test/new_traj_{f.__name__}.parquet", index=False
    # )
    # new_kalman[new_kalman.columns[:-1]].to_parquet(
    #     f"{path_start}/data_test/new_kalman_{f.__name__}.parquet", index=False
    # )

    result_traj = pd.read_parquet(
        f"{path_start}/data_test/new_traj_{f.__name__}.parquet"
    )
    result_kalman = pd.read_parquet(
        f"{path_start}/data_test/new_kalman_{f.__name__}.parquet"
    )

    # print("   TEST RESULTS   ")
    # print(result_traj)
    # print(result_kalman)

    assert_test(new_traj, new_kalman, result_traj, result_kalman)


def test_1():
    aux_test_runner(d.data_test_1)


def test_2():
    aux_test_runner(d.data_test_2)


def test_3():
    aux_test_runner(d.data_test_3)


def test_4():
    aux_test_runner(d.data_test_4)


def test_5():
    aux_test_runner(d.data_test_5)


def test_6():
    aux_test_runner(d.data_test_6)


def test_7():
    aux_test_runner(d.data_test_7)


test_1()
test_2()
test_3()
test_4()
test_5()
test_6()
test_7()
