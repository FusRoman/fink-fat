import numpy as np
from collections import Counter
from terminaltables import AsciiTable
from fink_fat.others.utils import init_logging


def test_detectable(list_diff_night, traj_time_window, orbfit_limit):
    """
    Return true if a trajectory is detectable by Fink-FAT.
    A trajectory is detectable if it contains a number of consecutive point equal to the orbfit_limit
    and the number of night between the points are less than the traj_time_window.

    Parameters
    ----------
    list_diff_night : pd.Series
        A pandas series containing list with the interval of nights between the alerts
    traj_time_window : integer
        the Fink-FAT parameter that manage the number of night a trajectory can be kept after the last associated alert.
    orbfit_limit : integer
        the number of minimum point required to send a trajectory to OrbFit.

    Returns
    -------
    is_detectable : boolean
        if true, the trajectory is detectable by Fink-FAT according only to the time window
        (do not take into account of the separation / magnitude / angle filters).

    Examples
    --------
    >>> tr_night = pd.Series({"diff_night": [5, 2, 6, 3]})
    >>> test_detectable(tr_night, 8, 6)
    True

    >>> tr_night = pd.Series({"diff_night": [5, 2, 9, 3]})
    >>> test_detectable(tr_night, 8, 6)
    False

    >>> tr_night = pd.Series({"diff_night": [5, 2]})
    >>> test_detectable(tr_night, 8, 6)
    False

    >>> tr_night = pd.Series({"diff_night": [5, 2, 1]})
    >>> test_detectable(tr_night, 8, 6)
    True

    >>> tr_night = pd.Series({"diff_night": [5, 2, 12, 2, 3, 12, 15, 10, 24, 1, 2, 4]})
    >>> test_detectable(tr_night, 8, 6)
    True
    """
    np_array = np.array(list_diff_night["diff_night"])
    np_mask = np.ma.masked_array(np_array, np_array > traj_time_window)

    for i in range(len(np_mask) - 2):
        current = np_mask[i]
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


def compute_residue(df):
    """
    Compute the difference between the orbit from the MPC catalog and the ones computed by Fink_FAT.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing the orbit from the MPC catalog and from Fink-FAT, this dataframe results from a merging between the MPC catalog
        and the Fink-FAT trajectory dataframe.

    Returns
    -------
    df : pd.DataFrame
        the input dataframe with additional columns containing the residual difference (in %).

    Examples
    --------
    >>> test_residue = pd.DataFrame({
    ... "a_x": [2.3, 5.7],
    ... "e_x": [0.2, 0.7],
    ... "i_x": [0.54, 15.7],
    ... "long. node": [25.4, 29.7],
    ... "arg. peric": [23.8, 48.7],
    ... "mean anomaly": [144.7, 231.3],
    ... "a_y": [1.7, 3.9],
    ... "e_y": [0.5, 0.3],
    ... "i_y": [2.8, 17.9],
    ... "Node": [36.4, 14.7],
    ... "Peri": [56.8, 78.7],
    ... "M": [29.7, 312.3],
    ... })

    >>> residue = compute_residue(test_residue)

    >>> assert_frame_equal(residue, ts.residue_test)
    """
    df = df.reset_index(drop=True)
    computed_elem = df[
        ["a_x", "e_x", "i_x", "long. node", "arg. peric", "mean anomaly"]
    ]
    known_elem = df[["a_y", "e_y", "i_y", "Node", "Peri", "M"]]

    df[["da", "de", "di", "dNode", "dPeri", "dM"]] = (
        np.abs(known_elem.values - computed_elem.values)
    ) / known_elem.values

    return df


def assoc_metrics(x):
    """
    Compute the association information for a trajectory

    Parameters
    ----------
    x : pd.Series
        a row of a trajectory dataframe, must contains at least the 'assoc_tag' column.

    Returns
    -------
    start : integer
        Detect by which associations the trajectory has started.
            - 0 for a trajectory started by an intra nigth associations.
            - 1 for a trajectory started by a pair of alerts
            - 2 for a trajectory started by a tracklets + an old alerts
    c_A : integer
        the number of single alerts added to the trajectory.
    c_T : integer
        the number of alerts in the tracklets added to the trajectory. (the first tracklets not include)
    len_tags : the number of tags / alerts of the trajectory.

    Examples
    --------
    >>> tags = pd.Series({"assoc_tag": ['I', 'A', 'T', 'A']})
    >>> assoc_metrics(tags)
    (0, 2, 1, 4)

    >>> tags = pd.Series({"assoc_tag": ['O', 'I', 'A', 'T', 'A']})
    >>> assoc_metrics(tags)
    (2, 2, 1, 5)

    >>> tags = pd.Series({"assoc_tag": ['N', 'A', 'A', 'A']})
    >>> assoc_metrics(tags)
    (1, 3, 0, 4)

    >>> tags = pd.Series({"assoc_tag": ['I']})
    >>> assoc_metrics(tags)
    (3, 0, 0, 1)
    """
    tags = x["assoc_tag"]

    if "O" in tags:
        start = 2
    elif tags[0] == "I":
        start = 0
    elif tags[0] == "N":
        start = 1
    else:  # pragma: no cover
        logger = init_logging()
        logger.info(tags)
        raise Exception("bad trajectory starting")

    c = Counter(tags)
    k_assoc = list(c.keys())
    if len(k_assoc) == 1 and k_assoc[0] == "I":
        start = 3

    return start, c["A"], c["T"], len(tags)


def assoc_stats(traj):
    """
    Compute some statistics over the associations.

    Parameters
    ----------
    traj : pd.DataFrame
        the dataframe containing all the trajectory

    Returns
    -------
    len_track : integer
        The number of intra night trajectory
    len_pair_p : integer
        The number of pair of points
    len_b_track : integer
        the number of trajectory beginning by an intra night trajectory
    len_b_pair : integer
        the number of trajectory beginning by a pair of points
    len_b_old : integer
        the number of trajectory beginning by an intra night trajectory + an old alert
    mean_l_traj_1 : float
        the number of single alert added to a trajectory in average
    mean_l_traj_2 : float
        the number of alert contains in the intra night tracklets added to a trajectory in average

    Examples
    --------
    >>> traj_df = pd.DataFrame({
    ... "trajectory_id": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
    ... "assoc_tag": ['I', 'I', 'A', 'T', 'T', 'A', 'N', 'N', 'T', 'T', 'T', 'A', 'O', 'I', 'I', 'I', 'T', 'T'],
    ... "ra": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ... })

    >>> assoc_stats(traj_df)
    (0, 0, 1, 1, 1, 1.0, 2.3333333333333335)
    """
    gb = (
        traj[["trajectory_id", "assoc_tag", "ra"]]
        .groupby("trajectory_id")
        .agg(assoc_tag=("assoc_tag", list), count=("ra", len))
    )

    t = gb.apply(assoc_metrics, axis=1, result_type="expand")

    b_track = t[t[0] == 0]
    b_pair = t[t[0] == 1]
    b_old = t[t[0] == 2]

    track = t[t[0] == 3]
    pair_p = t[t[0] == 4]

    l_traj = t[(t[0].isin([0, 1])) | ((t[0] == 2) & (t[3] > 3))]

    return (
        len(track),
        len(pair_p),
        len(b_track),
        len(b_pair),
        len(b_old),
        np.mean(l_traj[1]),
        np.mean(l_traj[2]),
    )


def print_assoc_table(traj_df):  # pragma: no cover
    """
    Print the table that describe the associations of the trajectories.

    Parameters
    ----------
    traj_df : dataframe
        the trajectory dataframe containing the observations of the trajectories with orbital elements

    Returns
    -------
    None
    """
    (
        nb_intra,
        nb_pair_p,
        nb_b_intra,
        nb_b_pair,
        nb_b_intra_o,
        mean_a,
        mean_t,
    ) = assoc_stats(traj_df)
    nb_traj = len(np.unique(traj_df["trajectory_id"]))
    assoc_data = (
        ("descriptions", "values (absolute)", "values (percentage)"),
        (
            "Number of intra night tracklets",
            nb_intra,
            np.around((nb_intra / nb_traj) * 100, 3),
        ),
        (
            "Number of pair of points",
            nb_pair_p,
            np.around((nb_pair_p / nb_traj) * 100, 3),
        ),
        (
            "Number of trajectory beggining by an intra night tracklets",
            nb_b_intra,
            np.around((nb_b_intra / nb_traj) * 100, 3),
        ),
        (
            "```   ```   ```   ```   ```   ``` a pair of points",
            nb_b_pair,
            np.around((nb_b_pair / nb_traj) * 100, 3),
        ),
        (
            "```   ```   ```   ```   ```   ``` an intra night tracklets + an old alert",
            nb_b_intra_o,
            np.around((nb_b_intra_o / nb_traj) * 100, 3),
        ),
        (
            "Number of single alerts added to a trajectory in average",
            np.around(mean_a, 3),
            "X",
        ),
        (
            "Number of alerts in the intra night added to a trajectory in average",
            np.around(mean_t, 3),
            "X",
        ),
    )

    assoc_table = AsciiTable(
        assoc_data, "Trajectories candidates association statistics"
    )
    logger = init_logging()
    logger.info(assoc_table.table)
    logger.newline()


def describe(df, stats):
    """
    Add additional statistics to the return of pandas.describe

    Parameters
    ----------
    df : dataframe
        dataframe for compute statistics
    stats : string list
        list with name of additional stats

    Examples
    --------
    >>> test = pd.DataFrame({
    ...     "a": [0, 10, 2, 5, 6, 9, 7, 78],
    ...     "b": [5, 47, 23, 24, 98, 14, 1, 23]
    ... })

    >>> describe(test, ["median", "kurtosis", "skew"])
                      a          b
    count      8.000000   8.000000
    mean      14.625000  29.375000
    std       25.823232  31.089445
    min        0.000000   1.000000
    25%        4.250000  11.750000
    50%        6.500000  23.000000
    75%        9.250000  29.750000
    max       78.000000  98.000000
    median     6.500000  23.000000
    kurtosis   7.607253   3.664267
    skew       2.733760   1.819347
    """
    d = df.describe()
    return d.append(df.reindex(d.columns, axis=1).agg(stats))


if __name__ == "__main__":  # pragma: no cover
    import sys
    import doctest
    from pandas.testing import assert_frame_equal  # noqa: F401
    import fink_fat.test.test_sample as ts  # noqa: F401
    from unittest import TestCase  # noqa: F401
    import shutil  # noqa: F401
    import pandas as pd  # noqa: F401

    if "unittest.util" in __import__("sys").modules:
        # Show full diff in self.assertEqual.
        __import__("sys").modules["unittest.util"]._MAX_LENGTH = 999999999

    sys.exit(doctest.testmod()[0])
