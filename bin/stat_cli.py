import numpy as np


def test_detectable(list_diff_night, traj_time_window, orbfit_limit):
    np_array = np.array(list_diff_night["diff_night"])
    np_mask = np.ma.masked_array(np_array, np_array > traj_time_window)
    for i in range(len(np_mask) - 2):
        current = np_mask[0]
        n_next_ = np_mask[i + 2]
        if current is np.ma.masked and n_next_ is np.ma.masked:
            np_mask[i + 1] = np.ma.masked

    not_mask = np.logical_not(np_mask.mask)
    count_consecutif = np.diff(
        np.where(
            np.concatenate(([not_mask[0]], not_mask[:-1] != not_mask[1:], [True]))
        )[0]
    )[::2]

    return np.any(count_consecutif * 2 >= orbfit_limit)
