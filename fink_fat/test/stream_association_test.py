from fink_fat.associations.stream_association import (
    merge_trajectory_alerts,
    merge_trajectory_cluster,
    stream_association,
)
import pandas as pd
from fink_fat.others.utils import init_logging
import numpy as np
from pandas.testing import assert_frame_equal

# test merge traj and cluster

traj = pd.DataFrame(
    {
        "trajectory_id": [0, 0, 1, 1],
        "candid": [0, 1, 2, 3],
        "ra": [0, 1, 5, 6],
        "dec": [0, 1, 5, 6],
        "jd": [0, 1, 0, 1],
        "magpsf": [0, 1, 2, 3],
        "fid": [1, 2, 2, 1],
    }
)

new_alerts = pd.DataFrame(
    {
        "trajectory_id": [0, 0],
        "estimator_id": [0, 0],
        "roid": [4, 4],
        "candid": [4, 5],
        "jd": [2, 3],
        "ra": [8, 9],
        "dec": [8, 9],
        "magpsf": [8, 9],
        "fid": [2, 2],
    }
)

res_traj, res_fit = merge_trajectory_cluster(
    traj, pd.DataFrame(), 2, new_alerts, init_logging(), False
)

test_traj = pd.DataFrame(
    {
        "trajectory_id": [2, 2, 2, 2],
        "candid": [0, 1, 4, 5],
        "ra": [0, 1, 8, 9],
        "dec": [0, 1, 8, 9],
        "jd": [0, 1, 2, 3],
        "magpsf": [0, 1, 8, 9],
        "fid": [1, 2, 2, 2],
        "estimator_id": [0.0, 0.0, 0.0, 0.0],
        "roid": [0.0, 0.0, 4.0, 4.0],
    }
)

test_fit = pd.DataFrame(
    {
        "trajectory_id": [2],
        "ra_0": [8],
        "dec_0": [8],
        "ra_1": [9],
        "dec_1": [9],
        "jd_0": [2],
        "jd_1": [3],
        "mag_1": [9],
        "fid_1": [2],
        "xp": [np.array([-0.0011995, -0.00564948, 1.0016361])],
        "yp": [np.array([-0.00018998, 0.05895939, -0.01032991])],
        "zp": [np.array([-4.77605893e-05, 5.92456907e-02, -1.04363809e-02])],
    }
)

assert_frame_equal(
    res_traj.fillna(0).reset_index(drop=True),
    test_traj.reset_index(drop=True),
    check_like=True,
)
assert_frame_equal(
    res_fit.fillna(0).reset_index(drop=True),
    test_fit.reset_index(drop=True),
    check_like=True,
)


traj = pd.DataFrame(
    {
        "trajectory_id": [0, 0, 1, 1],
        "candid": [0, 1, 2, 3],
        "ra": [0, 1, 5, 6],
        "dec": [0, 1, 5, 6],
        "jd": [0, 1, 0, 1],
        "magpsf": [0, 1, 2, 3],
        "fid": [1, 2, 2, 1],
    }
)

new_alerts = pd.DataFrame(
    {
        "trajectory_id": [0, 0, 5, 5],
        "estimator_id": [0, 0, 1, 1],
        "roid": [4, 4, 4, 4],
        "candid": [4, 5, 6, 7],
        "jd": [2, 3, 4, 5],
        "ra": [8, 9, 11, 12],
        "dec": [8, 9, 11, 12],
        "magpsf": [8, 9, 8, 9],
        "fid": [2, 2, 1, 2],
    }
)

res_traj, res_fit = merge_trajectory_cluster(
    traj, pd.DataFrame(), 2, new_alerts, init_logging(), False
)

test_traj = pd.DataFrame(
    {
        "trajectory_id": [2, 2, 2, 2, 3, 3, 3, 3],
        "candid": [0, 1, 4, 5, 2, 3, 6, 7],
        "ra": [0, 1, 8, 9, 5, 6, 11, 12],
        "dec": [0, 1, 8, 9, 5, 6, 11, 12],
        "jd": [0, 1, 2, 3, 0, 1, 4, 5],
        "magpsf": [0, 1, 8, 9, 2, 3, 8, 9],
        "fid": [1, 2, 2, 2, 2, 1, 1, 2],
        "estimator_id": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
        "roid": [0.0, 0.0, 4.0, 4.0, 0.0, 0.0, 4.0, 4.0],
    }
)
test_fit = pd.DataFrame(
    {
        "trajectory_id": [2, 3],
        "ra_0": [8, 11],
        "dec_0": [8, 11],
        "ra_1": [9, 12],
        "dec_1": [9, 12],
        "jd_0": [2, 4],
        "jd_1": [3, 5],
        "mag_1": [9, 9],
        "fid_1": [2, 2],
        "xp": [
            np.array([-0.0011995, -0.00564948, 1.0016361]),
            np.array([-4.36140290e-04, -5.30757377e-03, 9.93308991e-01]),
        ],
        "yp": [
            np.array([-0.00018998, 0.05895939, -0.01032991]),
            np.array([-0.00013334, 0.02515975, 0.08386359]),
        ],
        "zp": [
            np.array([-4.77605893e-05, 5.92456907e-02, -1.04363809e-02]),
            np.array([-3.37531348e-05, 2.55399815e-02, 8.41056773e-02]),
        ],
    }
)

assert_frame_equal(
    res_traj.fillna(0).reset_index(drop=True),
    test_traj.reset_index(drop=True),
    check_like=True,
)
assert_frame_equal(
    res_fit.fillna(0).reset_index(drop=True),
    test_fit.reset_index(drop=True),
    check_like=True,
)


traj = pd.DataFrame(
    {
        "trajectory_id": [0, 0, 1, 1, 2, 2, 2],
        "candid": [0, 1, 2, 3, 4, 5, 6],
        "ra": [0, 1, 5, 6, 0, 1, 2],
        "dec": [0, 1, 5, 6, 0, 1, 2],
        "jd": [0, 1, 0, 1, 0, 1, 2],
        "magpsf": [0, 1, 2, 3, 0, 1, 2],
        "fid": [1, 2, 2, 1, 1, 2, 1],
    }
)

new_alerts = pd.DataFrame(
    {
        "trajectory_id": [0, 0, 5, 5, 8, 8, 8],
        "estimator_id": [0, 0, 1, 1, 2, 2, 1],
        "roid": [4, 4, 4, 4, 4, 4, 4],
        "candid": [40, 50, 60, 70, 66, 77, 88],
        "jd": [2, 3, 4, 5, 3, 4, 5],
        "ra": [8, 9, 11, 12, 5, 6, 7],
        "dec": [8, 9, 11, 12, 5, 6, 7],
        "magpsf": [8, 9, 8, 9, 3, 2, 1],
        "fid": [2, 2, 1, 2, 2, 1, 1],
    }
)

res_traj, res_fit = merge_trajectory_cluster(
    traj, pd.DataFrame(), 3, new_alerts, init_logging(), False
)

test_traj = pd.DataFrame(
    {
        "trajectory_id": [3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6],
        "candid": [0, 1, 40, 50, 2, 3, 60, 70, 2, 3, 66, 77, 88, 4, 5, 6, 66, 77, 88],
        "ra": [0, 1, 8, 9, 5, 6, 11, 12, 5, 6, 5, 6, 7, 0, 1, 2, 5, 6, 7],
        "dec": [0, 1, 8, 9, 5, 6, 11, 12, 5, 6, 5, 6, 7, 0, 1, 2, 5, 6, 7],
        "jd": [0, 1, 2, 3, 0, 1, 4, 5, 0, 1, 3, 4, 5, 0, 1, 2, 3, 4, 5],
        "magpsf": [0, 1, 8, 9, 2, 3, 8, 9, 2, 3, 3, 2, 1, 0, 1, 2, 3, 2, 1],
        "fid": [1, 2, 2, 2, 2, 1, 1, 2, 2, 1, 2, 1, 1, 1, 2, 1, 2, 1, 1],
        "estimator_id": [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
            0.0,
            0.0,
            2.0,
            2.0,
            1.0,
            0.0,
            0.0,
            0.0,
            2.0,
            2.0,
            1.0,
        ],
        "roid": [
            0.0,
            0.0,
            4.0,
            4.0,
            0.0,
            0.0,
            4.0,
            4.0,
            0.0,
            0.0,
            4.0,
            4.0,
            4.0,
            0.0,
            0.0,
            0.0,
            4.0,
            4.0,
            4.0,
        ],
    }
)
test_fit = pd.DataFrame(
    {
        "trajectory_id": [3, 4, 5, 6],
        "ra_0": [8, 11, 6, 6],
        "dec_0": [8, 11, 6, 6],
        "ra_1": [9, 12, 7, 7],
        "dec_1": [9, 12, 7, 7],
        "jd_0": [2, 4, 4, 4],
        "jd_1": [3, 5, 5, 5],
        "mag_1": [9, 9, 1, 1],
        "fid_1": [2, 2, 1, 1],
        "xp": [
            np.array([-0.0011995, -0.00564948, 1.0016361]),
            np.array([-4.36140290e-04, -5.30757377e-03, 9.93308991e-01]),
            np.array([-4.84540438e-04, 1.40388349e-03, 9.90912859e-01]),
            np.array([-4.95955442e-04, -7.34613431e-04, 1.00056662e00]),
        ],
        "yp": [
            np.array([-0.00018998, 0.05895939, -0.01032991]),
            np.array([-0.00013334, 0.02515975, 0.08386359]),
            np.array([0.00215592, -0.00601496, 0.0941527]),
            np.array([-6.08956711e-05, 2.64835893e-02, -4.97261832e-03]),
        ],
        "zp": [
            np.array([-4.77605893e-05, 5.92456907e-02, -1.04363809e-02]),
            np.array([-3.37531348e-05, 2.55399815e-02, 8.41056773e-02]),
            np.array([0.00219643, -0.00613609, 0.09459778]),
            np.array([-1.52698300e-05, 2.64429530e-02, -4.98316688e-03]),
        ],
    }
)

assert_frame_equal(
    res_traj.fillna(0).reset_index(drop=True),
    test_traj.reset_index(drop=True),
    check_like=True,
)
assert_frame_equal(
    res_fit.fillna(0).reset_index(drop=True),
    test_fit.reset_index(drop=True),
    check_like=True,
)


traj = pd.DataFrame(
    {
        "trajectory_id": [0, 0, 1, 1],
        "candid": [0, 1, 2, 3],
        "ra": [0, 1, 5, 6],
        "dec": [0, 1, 5, 6],
        "jd": [0, 1, 0, 1],
        "magpsf": [0, 1, 2, 3],
        "fid": [1, 2, 2, 1],
    }
)

new_alerts = pd.DataFrame(
    {
        "trajectory_id": [0, 0, 1, 1],
        "estimator_id": [0, 0, 0, 0],
        "roid": [4, 4, 4, 4],
        "candid": [40, 50, 60, 70],
        "jd": [2, 3, 4, 5],
        "ra": [8, 9, 10, 11],
        "dec": [8, 9, 10, 11],
        "magpsf": [8, 9, 10, 11],
        "fid": [2, 2, 1, 2],
    }
)

res_traj, res_fit = merge_trajectory_cluster(
    traj, pd.DataFrame(), 3, new_alerts, init_logging(), False
)

test_traj = pd.DataFrame(
    {
        "trajectory_id": [3, 3, 3, 3, 4, 4, 4, 4],
        "candid": [0, 1, 40, 50, 0, 1, 60, 70],
        "ra": [0, 1, 8, 9, 0, 1, 10, 11],
        "dec": [0, 1, 8, 9, 0, 1, 10, 11],
        "jd": [0, 1, 2, 3, 0, 1, 4, 5],
        "magpsf": [0, 1, 8, 9, 0, 1, 10, 11],
        "fid": [1, 2, 2, 2, 1, 2, 1, 2],
        "estimator_id": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "roid": [0.0, 0.0, 4.0, 4.0, 0.0, 0.0, 4.0, 4.0],
    }
)
test_fit = pd.DataFrame(
    {
        "trajectory_id": [3, 4],
        "ra_0": [8, 10],
        "dec_0": [8, 10],
        "ra_1": [9, 11],
        "dec_1": [9, 11],
        "jd_0": [2, 4],
        "jd_1": [3, 5],
        "mag_1": [9, 11],
        "fid_1": [2, 2],
        "xp": [
            np.array([-0.0011995, -0.00564948, 1.0016361]),
            np.array([-7.43724577e-04, -4.26924985e-03, 1.00176565e00]),
        ],
        "yp": [
            np.array([-0.00018998, 0.05895939, -0.01032991]),
            np.array([-0.00014457, 0.04181687, -0.00908346]),
        ],
        "zp": [
            np.array([-4.77605893e-05, 5.92456907e-02, -1.04363809e-02]),
            np.array([-3.64485910e-05, 4.20243691e-02, -9.20081779e-03]),
        ],
    }
)

assert_frame_equal(
    res_traj.fillna(0).reset_index(drop=True),
    test_traj.reset_index(drop=True),
    check_like=True,
)
assert_frame_equal(
    res_fit.fillna(0).reset_index(drop=True),
    test_fit.reset_index(drop=True),
    check_like=True,
)


traj = pd.DataFrame(
    {
        "trajectory_id": [0, 0, 1, 1],
        "candid": [0, 1, 2, 3],
        "ra": [0, 1, 5, 6],
        "dec": [0, 1, 5, 6],
        "jd": [0, 1, 0, 1],
        "magpsf": [0, 1, 2, 3],
        "fid": [1, 2, 2, 1],
    }
)

new_alerts = pd.DataFrame(
    {
        "trajectory_id": [0, 0, 1, 1],
        "estimator_id": [0, 0, 1, 1],
        "roid": [4, 4, 4, 4],
        "candid": [40, 50, 40, 50],
        "jd": [2, 3, 2, 3],
        "ra": [8, 9, 8, 9],
        "dec": [8, 9, 8, 9],
        "magpsf": [8, 9, 8, 9],
        "fid": [2, 2, 2, 2],
    }
)

res_traj, res_fit = merge_trajectory_cluster(
    traj, pd.DataFrame(), 3, new_alerts, init_logging(), False
)


test_traj = pd.DataFrame(
    {
        "trajectory_id": [3, 3, 3, 3, 4, 4, 4, 4],
        "candid": [0, 1, 40, 50, 2, 3, 40, 50],
        "ra": [0, 1, 8, 9, 5, 6, 8, 9],
        "dec": [0, 1, 8, 9, 5, 6, 8, 9],
        "jd": [0, 1, 2, 3, 0, 1, 2, 3],
        "magpsf": [0, 1, 8, 9, 2, 3, 8, 9],
        "fid": [1, 2, 2, 2, 2, 1, 2, 2],
        "estimator_id": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
        "roid": [0.0, 0.0, 4.0, 4.0, 0.0, 0.0, 4.0, 4.0],
    }
)
test_fit = pd.DataFrame(
    {
        "trajectory_id": [3, 4],
        "ra_0": [8, 8],
        "dec_0": [8, 8],
        "ra_1": [9, 9],
        "dec_1": [9, 9],
        "jd_0": [2, 2],
        "jd_1": [3, 3],
        "mag_1": [9, 9],
        "fid_1": [2, 2],
        "xp": [
            np.array([-0.0011995, -0.00564948, 1.0016361]),
            np.array([-4.43128421e-04, -4.57759549e-03, 9.92826538e-01]),
        ],
        "yp": [
            np.array([-0.00018998, 0.05895939, -0.01032991]),
            np.array([-0.00011048, 0.02402306, 0.08512888]),
        ],
        "zp": [
            np.array([-4.77605893e-05, 5.92456907e-02, -1.04363809e-02]),
            np.array([-2.78391100e-05, 2.43315978e-02, 8.54229832e-02]),
        ],
    }
)

assert_frame_equal(
    res_traj.fillna(0).reset_index(drop=True),
    test_traj.reset_index(drop=True),
    check_like=True,
)
assert_frame_equal(
    res_fit.fillna(0).reset_index(drop=True),
    test_fit.reset_index(drop=True),
    check_like=True,
)


traj = pd.DataFrame(
    {
        "trajectory_id": [0, 0, 1, 1],
        "candid": [0, 1, 2, 3],
        "ra": [0, 1, 5, 6],
        "dec": [0, 1, 5, 6],
        "jd": [0, 1, 0, 1],
        "magpsf": [0, 1, 2, 3],
        "fid": [1, 2, 2, 1],
    }
)

new_alerts = pd.DataFrame(
    {
        "trajectory_id": [0, 0, 0, 0],
        "estimator_id": [0, 1, 0, 1],
        "roid": [4, 4, 4, 4],
        "candid": [4, 5, 4, 5],
        "jd": [2, 3, 2, 3],
        "ra": [8, 9, 8, 9],
        "dec": [8, 9, 8, 9],
        "magpsf": [8, 9, 8, 9],
        "fid": [2, 2, 2, 2],
    }
)

res_traj, res_fit = merge_trajectory_cluster(
    traj, pd.DataFrame(), 2, new_alerts, init_logging(), False
)

test_traj = pd.DataFrame(
    {
        "trajectory_id": [2, 2, 2, 2, 3, 3, 3, 3],
        "candid": [0, 1, 4, 5, 2, 3, 4, 5],
        "ra": [0, 1, 8, 9, 5, 6, 8, 9],
        "dec": [0, 1, 8, 9, 5, 6, 8, 9],
        "jd": [0, 1, 2, 3, 0, 1, 2, 3],
        "magpsf": [0, 1, 8, 9, 2, 3, 8, 9],
        "fid": [1, 2, 2, 2, 2, 1, 2, 2],
        "estimator_id": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "roid": [0.0, 0.0, 4.0, 4.0, 0.0, 0.0, 4.0, 4.0],
    }
)
test_fit = pd.DataFrame(
    {
        "trajectory_id": [2, 3],
        "ra_0": [8, 8],
        "dec_0": [8, 8],
        "ra_1": [9, 9],
        "dec_1": [9, 9],
        "jd_0": [2, 2],
        "jd_1": [3, 3],
        "mag_1": [9, 9],
        "fid_1": [2, 2],
        "xp": [
            np.array([-0.0011995, -0.00564948, 1.0016361]),
            np.array([-4.43128421e-04, -4.57759549e-03, 9.92826538e-01]),
        ],
        "yp": [
            np.array([-0.00018998, 0.05895939, -0.01032991]),
            np.array([-0.00011048, 0.02402306, 0.08512888]),
        ],
        "zp": [
            np.array([-4.77605893e-05, 5.92456907e-02, -1.04363809e-02]),
            np.array([-2.78391100e-05, 2.43315978e-02, 8.54229832e-02]),
        ],
    }
)

assert_frame_equal(
    res_traj.fillna(0).reset_index(drop=True),
    test_traj.reset_index(drop=True),
    check_like=True,
)
assert_frame_equal(
    res_fit.fillna(0).reset_index(drop=True),
    test_fit.reset_index(drop=True),
    check_like=True,
)


# ======================================================================== #
# test merge traj and single alert


traj = pd.DataFrame(
    {
        "trajectory_id": [0, 0, 1, 1],
        "candid": [0, 1, 2, 3],
        "ra": [0, 1, 5, 6],
        "dec": [0, 1, 5, 6],
        "jd": [0, 1, 0, 1],
        "magpsf": [0, 1, 2, 3],
        "fid": [1, 2, 2, 1],
    }
)

new_alerts = pd.DataFrame(
    {
        "trajectory_id": [-1, -1],
        "estimator_id": [0, 1],
        "roid": [4, 4],
        "candid": [4, 5],
        "jd": [2, 3],
        "ra": [8, 9],
        "dec": [8, 9],
        "magpsf": [8, 9],
        "fid": [2, 2],
    }
)

res_traj, res_fit = merge_trajectory_alerts(
    traj, pd.DataFrame(), 2, new_alerts, init_logging(), False
)

test_traj = pd.DataFrame(
    {
        "trajectory_id": [2, 2, 3, 3, 2, 3],
        "candid": [0, 1, 2, 3, 4, 5],
        "ra": [0, 1, 5, 6, 8, 9],
        "dec": [0, 1, 5, 6, 8, 9],
        "jd": [0, 1, 0, 1, 2, 3],
        "magpsf": [0, 1, 2, 3, 8, 9],
        "fid": [1, 2, 2, 1, 2, 2],
        "estimator_id": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        "roid": [0.0, 0.0, 0.0, 0.0, 4.0, 4.0],
    }
)
test_fit = pd.DataFrame(
    {
        "trajectory_id": [2, 3],
        "ra_0": [1, 6],
        "dec_0": [1, 6],
        "ra_1": [8, 9],
        "dec_1": [8, 9],
        "jd_0": [1, 1],
        "jd_1": [2, 3],
        "mag_1": [8, 9],
        "fid_1": [2, 2],
        "xp": [
            np.array([-0.00937999, 0.0090754, 1.0]),
            np.array([-0.00114756, -0.00218251, 0.99240388]),
        ],
        "yp": [
            np.array([5.14595906e-02, -3.40098423e-02, -7.49515362e-18]),
            np.array([0.00271486, 0.0144169, 0.08682409]),
        ],
        "zp": [
            np.array([5.21341440e-02, -3.46817376e-02, 1.49650697e-17]),
            np.array([0.00286009, 0.01451263, 0.08715574]),
        ],
    }
)

assert_frame_equal(
    res_traj.fillna(0).reset_index(drop=True),
    test_traj.reset_index(drop=True),
    check_like=True,
)
assert_frame_equal(
    res_fit.fillna(0).reset_index(drop=True),
    test_fit.reset_index(drop=True),
    check_like=True,
)


traj = pd.DataFrame(
    {
        "trajectory_id": [0, 0, 1, 1, 2, 2],
        "candid": [0, 1, 2, 3, 4, 5],
        "ra": [0, 1, 5, 6, 0, 1],
        "dec": [0, 1, 5, 6, 0, 1],
        "jd": [0, 1, 0, 1, 0, 1],
        "magpsf": [0, 1, 2, 3, 1, 5],
        "fid": [1, 2, 2, 1, 2, 1],
    }
)

new_alerts = pd.DataFrame(
    {
        "trajectory_id": [-1, -1, -1, -1, -1, -1],
        "estimator_id": [0, 1, 0, 1, 1, 0],
        "roid": [4, 4, 4, 4, 4, 4],
        "candid": [40, 50, 60, 70, 80, 90],
        "jd": [2, 3, 4, 2, 4, 5],
        "ra": [8, 9, 10, 11, 14, 15],
        "dec": [8, 9, 10, 11, 14, 15],
        "magpsf": [8, 9, 10, 5, 8, 9],
        "fid": [2, 2, 1, 1, 1, 2],
    }
)

res_traj, res_fit = merge_trajectory_alerts(
    traj, pd.DataFrame(), 2, new_alerts, init_logging(), False
)

test_traj = pd.DataFrame(
    {
        "trajectory_id": [2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 2, 3, 4, 5, 6, 7],
        "candid": [0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 40, 60, 90, 50, 70, 80],
        "ra": [0, 1, 0, 1, 0, 1, 5, 6, 5, 6, 5, 6, 8, 10, 15, 9, 11, 14],
        "dec": [0, 1, 0, 1, 0, 1, 5, 6, 5, 6, 5, 6, 8, 10, 15, 9, 11, 14],
        "jd": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 4, 5, 3, 2, 4],
        "magpsf": [0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 8, 10, 9, 9, 5, 8],
        "fid": [1, 2, 1, 2, 1, 2, 2, 1, 2, 1, 2, 1, 2, 1, 2, 2, 1, 1],
        "estimator_id": [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
            1.0,
        ],
        "roid": [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            4.0,
            4.0,
            4.0,
            4.0,
            4.0,
            4.0,
        ],
    }
)
test_fit = pd.DataFrame(
    {
        "trajectory_id": [2, 3, 4, 5, 6, 7],
        "ra_0": [1, 1, 1, 6, 6, 6],
        "dec_0": [1, 1, 1, 6, 6, 6],
        "ra_1": [8, 10, 15, 9, 11, 14],
        "dec_1": [8, 10, 15, 9, 11, 14],
        "jd_0": [1, 1, 1, 1, 1, 1],
        "jd_1": [2, 4, 5, 3, 2, 4],
        "mag_1": [8, 10, 9, 9, 5, 8],
        "fid_1": [2, 1, 2, 2, 1, 1],
        "xp": [
            np.array([-0.00937999, 0.0090754, 1.0]),
            np.array([-0.00241128, 0.00210669, 1.0]),
            np.array([-0.00327322, 0.00296863, 1.0]),
            np.array([-0.00114756, -0.00218251, 0.99240388]),
            np.array([-0.0110759, 0.00774582, 0.99240388]),
            np.array([-3.13414796e-03, -1.95928179e-04, 9.92403877e-01]),
        ],
        "yp": [
            np.array([5.14595906e-02, -3.40098423e-02, -7.49515362e-18]),
            np.array([8.43425652e-03, 9.01549183e-03, -2.02145803e-17]),
            np.array([8.13756291e-03, 9.31218544e-03, -2.06680824e-18]),
            np.array([0.00271486, 0.0144169, 0.08682409]),
            np.array([0.03310785, -0.01597609, 0.08682409]),
            np.array([0.00661539, 0.01051637, 0.08682409]),
        ],
        "zp": [
            np.array([5.21341440e-02, -3.46817376e-02, 1.49650697e-17]),
            np.array([8.65321266e-03, 8.79919378e-03, -1.98567215e-17]),
            np.array([8.57785065e-03, 8.87455579e-03, -3.07704099e-17]),
            np.array([0.00286009, 0.01451263, 0.08715574]),
            np.array([0.03445391, -0.01708119, 0.08715574]),
            np.array([0.00710627, 0.01026645, 0.08715574]),
        ],
    }
)

assert_frame_equal(
    res_traj.fillna(0).reset_index(drop=True),
    test_traj.reset_index(drop=True),
    check_like=True,
)
assert_frame_equal(
    res_fit.fillna(0).reset_index(drop=True),
    test_fit.reset_index(drop=True),
    check_like=True,
)


traj = pd.DataFrame(
    {
        "trajectory_id": [0, 0, 1, 1],
        "candid": [0, 1, 2, 3],
        "ra": [0, 1, 5, 6],
        "dec": [0, 1, 5, 6],
        "jd": [0, 1, 0, 1],
        "magpsf": [0, 1, 2, 3],
        "fid": [1, 2, 2, 1],
    }
)

new_alerts = pd.DataFrame(
    {
        "trajectory_id": [-1, -1, -1, -1, -1, -1],
        "estimator_id": [0, 0, 1, 1, 1, 1],
        "roid": [4, 4, 4, 4, 4, 4],
        "candid": [40, 50, 60, 70, 40, 50],
        "jd": [2, 3, 4, 5, 2, 3],
        "ra": [8, 9, 10, 11, 8, 9],
        "dec": [8, 9, 10, 11, 8, 9],
        "magpsf": [8, 9, 10, 11, 8, 9],
        "fid": [2, 2, 1, 2, 2, 2],
    }
)


res_traj, res_fit = merge_trajectory_alerts(
    traj, pd.DataFrame(), 2, new_alerts, init_logging(), False
)


test_traj = pd.DataFrame(
    {
        "trajectory_id": [2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 2, 3, 4, 5, 6, 7],
        "candid": [0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3, 40, 50, 60, 70, 40, 50],
        "ra": [0, 1, 0, 1, 5, 6, 5, 6, 5, 6, 5, 6, 8, 9, 10, 11, 8, 9],
        "dec": [0, 1, 0, 1, 5, 6, 5, 6, 5, 6, 5, 6, 8, 9, 10, 11, 8, 9],
        "jd": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 4, 5, 2, 3],
        "magpsf": [0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3, 8, 9, 10, 11, 8, 9],
        "fid": [1, 2, 1, 2, 2, 1, 2, 1, 2, 1, 2, 1, 2, 2, 1, 2, 2, 2],
        "estimator_id": [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ],
        "roid": [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            4.0,
            4.0,
            4.0,
            4.0,
            4.0,
            4.0,
        ],
    }
)
test_fit = pd.DataFrame(
    {
        "trajectory_id": [2, 3, 4, 5, 6, 7],
        "ra_0": [1, 1, 6, 6, 6, 6],
        "dec_0": [1, 1, 6, 6, 6, 6],
        "ra_1": [8, 9, 10, 11, 8, 9],
        "dec_1": [8, 9, 10, 11, 8, 9],
        "jd_0": [1, 1, 1, 1, 1, 1],
        "jd_1": [2, 3, 4, 5, 2, 3],
        "mag_1": [8, 9, 10, 11, 8, 9],
        "fid_1": [2, 2, 1, 2, 2, 2],
        "xp": [
            np.array([-0.00937999, 0.0090754, 1.0]),
            np.array([-0.00392633, 0.00362174, 1.0]),
            np.array([-7.69771796e-04, -2.56030434e-03, 9.92403877e-01]),
            np.array([-6.08078426e-04, -2.72199771e-03, 9.92403877e-01]),
            np.array([-2.55643813e-03, -7.73638010e-04, 9.92403877e-01]),
            np.array([-0.00114756, -0.00218251, 0.99240388]),
        ],
        "yp": [
            np.array([5.14595906e-02, -3.40098423e-02, -7.49515362e-18]),
            np.array([1.70265420e-02, 4.23206329e-04, 4.68081723e-17]),
            np.array([0.00130491, 0.01582684, 0.08682409]),
            np.array([0.00074102, 0.01639074, 0.08682409]),
            np.array([0.00836554, 0.00876622, 0.08682409]),
            np.array([0.00271486, 0.0144169, 0.08682409]),
        ],
        "zp": [
            np.array([5.21341440e-02, -3.46817376e-02, 1.49650697e-17]),
            np.array([1.73462076e-02, 1.06198816e-04, 4.21139816e-17]),
            np.array([0.0014168, 0.01595592, 0.08715574]),
            np.array([0.00083948, 0.01653324, 0.08715574]),
            np.array([0.00863596, 0.00873676, 0.08715574]),
            np.array([0.00286009, 0.01451263, 0.08715574]),
        ],
    }
)

assert_frame_equal(
    res_traj.fillna(0).reset_index(drop=True),
    test_traj.reset_index(drop=True),
    check_like=True,
)
assert_frame_equal(
    res_fit.fillna(0).reset_index(drop=True),
    test_fit.reset_index(drop=True),
    check_like=True,
)


# merge traj with new night

traj = pd.DataFrame(
    {
        "trajectory_id": [2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 2, 3, 4, 5, 6, 7],
        "candid": [0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3, 40, 50, 60, 70, 40, 50],
        "ra": [0, 1, 0, 1, 5, 6, 5, 6, 5, 6, 5, 6, 8, 9, 10, 11, 8, 9],
        "dec": [0, 1, 0, 1, 5, 6, 5, 6, 5, 6, 5, 6, 8, 9, 10, 11, 8, 9],
        "jd": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 4, 5, 2, 3],
        "magpsf": [0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3, 8, 9, 10, 11, 8, 9],
        "fid": [1, 2, 1, 2, 2, 1, 2, 1, 2, 1, 2, 1, 2, 2, 1, 2, 2, 2],
    }
)
pdf_fit = pd.DataFrame(
    {
        "trajectory_id": [2, 3, 4, 5, 6, 7],
        "ra_0": [0, 0, 5, 5, 5, 5],
        "dec_0": [0, 0, 5, 5, 5, 5],
        "ra_1": [1, 1, 6, 6, 6, 6],
        "dec_1": [1, 1, 6, 6, 6, 6],
        "jd_0": [0, 0, 0, 0, 0, 0],
        "jd_1": [1, 1, 1, 1, 1, 1],
        "mag_1": [1, 1, 3, 3, 3, 3],
        "fid_1": [2, 2, 1, 1, 1, 1],
        "xp": [
            np.array([-0.00937999, 0.0090754, 1.0]),
            np.array([-0.00392633, 0.00362174, 1.0]),
            np.array([-7.69771796e-04, -2.56030434e-03, 9.92403877e-01]),
            np.array([-6.08078426e-04, -2.72199771e-03, 9.92403877e-01]),
            np.array([-2.55643813e-03, -7.73638010e-04, 9.92403877e-01]),
            np.array([-0.00114756, -0.00218251, 0.99240388]),
        ],
        "yp": [
            np.array([5.14595906e-02, -3.40098423e-02, -7.49515362e-18]),
            np.array([1.70265420e-02, 4.23206329e-04, 4.68081723e-17]),
            np.array([0.00130491, 0.01582684, 0.08682409]),
            np.array([0.00074102, 0.01639074, 0.08682409]),
            np.array([0.00836554, 0.00876622, 0.08682409]),
            np.array([0.00271486, 0.0144169, 0.08682409]),
        ],
        "zp": [
            np.array([5.21341440e-02, -3.46817376e-02, 1.49650697e-17]),
            np.array([1.73462076e-02, 1.06198816e-04, 4.21139816e-17]),
            np.array([0.0014168, 0.01595592, 0.08715574]),
            np.array([0.00083948, 0.01653324, 0.08715574]),
            np.array([0.00863596, 0.00873676, 0.08715574]),
            np.array([0.00286009, 0.01451263, 0.08715574]),
        ],
    }
)

new_alerts = pd.DataFrame(
    {
        "trajectory_id": [0, 0, -1, -1],
        "estimator_id": [[2], [2], [5], [6]],
        "ffdistnr": [[0.2], [0.25], [0.5], [0.6]],
        "roid": [4, 4, 4, 4],
        "candid": [110, 120, 130, 150],
        "jd": [2, 3, 4, 5],
        "ra": [8, 9, 10, 11],
        "dec": [8, 9, 10, 11],
        "magpsf": [8, 9, 10, 11],
        "fid": [2, 2, 1, 2],
    }
)


res_traj, res_fit = stream_association(traj, pdf_fit, new_alerts, init_logging(), False)

test_traj = pd.DataFrame(
    {
        "trajectory_id": [
            3,
            3,
            4,
            4,
            7,
            7,
            3,
            4,
            7,
            8,
            8,
            8,
            8,
            8,
            9,
            9,
            9,
            10,
            10,
            10,
            9,
            10,
        ],
        "candid": [
            0,
            1,
            2,
            3,
            2,
            3,
            50,
            60,
            50,
            0,
            1,
            40,
            110,
            120,
            2,
            3,
            70,
            2,
            3,
            40,
            130,
            150,
        ],
        "ra": [0, 1, 5, 6, 5, 6, 9, 10, 9, 0, 1, 8, 8, 9, 5, 6, 11, 5, 6, 8, 10, 11],
        "dec": [0, 1, 5, 6, 5, 6, 9, 10, 9, 0, 1, 8, 8, 9, 5, 6, 11, 5, 6, 8, 10, 11],
        "jd": [0, 1, 0, 1, 0, 1, 3, 4, 3, 0, 1, 2, 2, 3, 0, 1, 5, 0, 1, 2, 4, 5],
        "magpsf": [
            0,
            1,
            2,
            3,
            2,
            3,
            9,
            10,
            9,
            0,
            1,
            8,
            8,
            9,
            2,
            3,
            11,
            2,
            3,
            8,
            10,
            11,
        ],
        "fid": [1, 2, 2, 1, 2, 1, 2, 1, 2, 1, 2, 2, 2, 2, 2, 1, 2, 2, 1, 2, 1, 2],
        "updated": [
            "N",
            "N",
            "N",
            "N",
            "N",
            "N",
            "N",
            "N",
            "N",
            "Y",
            "Y",
            "Y",
            "Y",
            "Y",
            "Y",
            "Y",
            "Y",
            "Y",
            "Y",
            "Y",
            "Y",
            "Y",
        ],
        "ffdistnr": [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.2,
            0.25,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.5,
            0.6,
        ],
        "roid": [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            4.0,
            4.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            4.0,
            4.0,
        ],
    }
)
test_fit = pd.DataFrame(
    {
        "trajectory_id": [3, 4, 7, 8, 9, 10],
        "ra_0": [0, 5, 5, 8, 10, 8],
        "dec_0": [0, 5, 5, 8, 10, 8],
        "ra_1": [1, 6, 6, 9, 11, 11],
        "dec_1": [1, 6, 6, 9, 11, 11],
        "jd_0": [0, 0, 0, 2, 4, 2],
        "jd_1": [1, 1, 1, 3, 5, 5],
        "mag_1": [1, 3, 3, 9, 11, 11],
        "fid_1": [2, 1, 1, 2, 2, 2],
        "xp": [
            np.array([-0.00392633, 0.00362174, 1.0]),
            np.array([-7.69771796e-04, -2.56030434e-03, 9.92403877e-01]),
            np.array([-0.00114756, -0.00218251, 0.99240388]),
            np.array([-4.07840637e-04, -8.34112128e-03, 1.00211109e00]),
            np.array([-3.65538371e-04, -4.10590273e-03, 9.92831888e-01]),
            np.array([-8.06219322e-05, -5.50998204e-03, 9.93049742e-01]),
        ],
        "yp": [
            np.array([1.70265420e-02, 4.23206329e-04, 4.68081723e-17]),
            np.array([0.00130491, 0.01582684, 0.08682409]),
            np.array([0.00271486, 0.0144169, 0.08682409]),
            np.array([-0.00518833, 0.07595376, -0.01332892]),
            np.array([-0.00010482, 0.02121699, 0.08533143]),
            np.array([-0.00132307, 0.02730096, 0.08429662]),
        ],
        "zp": [
            np.array([1.73462076e-02, 1.06198816e-04, 4.21139816e-17]),
            np.array([0.0014168, 0.01595592, 0.08715574]),
            np.array([0.00286009, 0.01451263, 0.08715574]),
            np.array([-0.00509762, 0.07641522, -0.0134663]),
            np.array([-2.64878513e-05, 2.14743630e-02, 8.56275598e-02]),
            np.array([-0.00127117, 0.02768952, 0.08457128]),
        ],
    }
)


assert_frame_equal(
    res_traj.fillna(0).reset_index(drop=True),
    test_traj.reset_index(drop=True),
    check_like=True,
)
assert_frame_equal(
    res_fit.fillna(0).reset_index(drop=True),
    test_fit.reset_index(drop=True),
    check_like=True,
)
