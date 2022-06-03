import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates import search_around_sky
import pandas as pd
import numpy as np
from fink_fat.associations.intra_night_association import magnitude_association
from fink_fat.associations.intra_night_association import (
    get_n_last_observations_from_trajectories,
)
from fink_fat.others.utils import repeat_chunk
from fink_fat.others.utils import cast_obs_data
import sys
import doctest
from pandas.testing import assert_frame_equal  # noqa: F401
from unittest import TestCase  # noqa: F401
import fink_fat.test.test_sample as ts  # noqa: F401


def night_to_night_separation_association(
    old_observation, new_observation, separation_criterion, store_kd_tree=False
):
    """
    Perform night-night association based on the separation between the alerts.
    The separation criterion was computed by a data analysis on the MPC object.

    Parameters
    ----------
    old_observation : dataframe
        observation of night t-1
    new_observation : dataframe
        observation of night t
    separation_criterion : float
        the separation limit between the alerts to be associated, must be in arcsecond

    Returns
    -------
    left_assoc : dataframe
        Associations are a binary relation with left members and right members, return the left members (from old_observation) of the associations
    right_assoc : dataframe
        return right members (from new_observation) of the associations
    sep2d : list
        return the separation between the associated alerts

    Examples
    --------
    >>> test_night1 = pd.DataFrame({
    ... 'ra': [10, 11, 20],
    ... 'dec' : [70, 30, 50],
    ... 'candid' : [1, 2, 3],
    ... 'trajectory_id' : [10, 11, 12]
    ... })

    >>> test_night2 = pd.DataFrame({
    ... 'ra' : [11, 12, 21],
    ... 'dec' : [69, 29, 49],
    ... 'candid' : [4, 5, 6],
    ... 'trajectory_id' : [10, 11, 12]
    ... })

    >>> left, right, sep = night_to_night_separation_association(test_night1, test_night2, 2*u.degree)

    >>> assert_frame_equal(test_night1, left)
    >>> assert_frame_equal(test_night2, right)
    >>> np.any([1.05951524, 1.32569856, 1.19235978] == np.around(sep.value, 8))
    True
    """

    old_observations_coord = SkyCoord(
        old_observation["ra"], old_observation["dec"], unit=u.degree
    )
    new_observations_coord = SkyCoord(
        new_observation["ra"], new_observation["dec"], unit=u.degree
    )

    old_obs_idx, new_obs_idx, sep2d, _ = search_around_sky(
        old_observations_coord,
        new_observations_coord,
        separation_criterion,
        storekdtree=store_kd_tree,
    )

    old_obs_assoc = old_observation.iloc[old_obs_idx]
    new_obs_assoc = new_observation.iloc[new_obs_idx]
    return old_obs_assoc, new_obs_assoc, sep2d


def angle_three_point(a, b, c):
    """
    Compute the angle between three points taken as two vectors

    Parameters
    ----------
    a : numpy array
        first point
    b : numpy array
        second point
    c : numpy array
        third point

    Returns
    -------
    angle : float
        the angle formed by the three points

    Examples
    --------
    >>> a = np.array([1, 1])
    >>> b = np.array([3, 2])
    >>> c = np.array([5, 4])

    >>> np.around(angle_three_point(a, b, c), 3)
    10.305

    >>> c = np.array([-2, -2])
    >>> np.around(angle_three_point(a, b, c), 3)
    161.565
    """
    ba = b - a
    ca = c - a

    cosine_angle = np.dot(ba, ca) / (np.linalg.norm(ba) * np.linalg.norm(ca))
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)


def angle_df(x):
    """
    Taken three alerts from a dataframe rows, computes the angle between the three alerts

    Parameters
    x : dataframe rows
        a rows from a dataframe with the three consecutives alerts to compute the angle.


    Returns
    -------
    res_angle : float
        the angle between the three consecutives alerts normalized by the jd difference between the second point and the third point.

    Examples
    --------
    >>> from pandera import Check, Column, DataFrameSchema

    >>> df_schema = DataFrameSchema({
    ... "trajectory_id": Column(int),
    ... "ra_x": Column(object),
    ... "dec_x": Column(object),
    ... "jd_x": Column(object),
    ... "candid": Column(int),
    ... "index": Column(int),
    ... "ra_y": Column(float),
    ... "dec_y": Column(float),
    ... "jd_y": Column(float)
    ... })

    >>> test_dataframe = pd.DataFrame({
    ... 'trajectory_id': [0],
    ... 'ra_x': [[1, 3]],
    ... 'dec_x': [[1, 2]],
    ... 'jd_x': [[0, 1]],
    ... 'candid': [0],
    ... 'index' : [0],
    ... 'ra_y': [5.0],
    ... 'dec_y':[4.0],
    ... 'jd_y': [2.0]
    ... })

    >>> test_dataframe = df_schema.validate(test_dataframe)

    >>> res = test_dataframe.apply(angle_df, axis=1)

    >>> np.around(np.array(res)[0], 3)
    10.305
    """

    ra_x, dec_x, jd_x = x["ra_x"], x["dec_x"], x["jd_x"]

    ra_y, dec_y, jd_y = x["ra_y"], x["dec_y"], x["jd_y"]

    a = np.array([ra_x[0], dec_x[0]])
    b = np.array([ra_x[1], dec_x[1]])
    c = np.array([ra_y, dec_y])

    jd_x = jd_x[1]

    diff_jd = jd_y - jd_x

    if diff_jd > 1:
        res_angle = angle_three_point(a, b, c) / diff_jd
    else:
        res_angle = angle_three_point(a, b, c)

    return res_angle


def cone_search_association(
    two_last_observations, traj_assoc, new_obs_assoc, angle_criterion
):
    """
    Filter the association based on a cone search. The idea is to remove the alerts triplets that have an angle greater than the angle_criterion.

    Parameters
    ----------
    two_last_observations : dataframe
        a dataframe that contains the two last observations for each trajectories
    traj_assoc : dataframe
        a dataframe which contains the trajectories extremity that are associated with the new observations.
        The trajectory extremity in traj_assoc have to be in two_last_observations.
    new_obs_assoc : dataframe
        new observations that will be associated with the trajectories extremity represented by the two last observations.
    angle_criterion : float
        the angle limit between the three points formed by the two last observation and the new observations to associates.

    Returns
    -------
    traj_assoc : dataframe
        trajectories extremity associated with the new observations filtered by the angle cone search
    new_obs_assoc : dataframe
        new observations associated with the trajectories filtered by the angle cone search

    Examples
    --------
    >>> left, right = cone_search_association(ts.cone_search_two_last_observation_sample, ts.traj_assoc_sample, ts.new_obs_assoc_sample, 8.8)

    >>> assert_frame_equal(left, ts.left_cone_search_expected)
    >>> assert_frame_equal(right, ts.right_cone_search_expected)

    >>> left, right = cone_search_association(ts.false_cone_search_two_last_observation_sample, ts.false_traj_assoc_sample, ts.false_new_obs_assoc_sample, 8.8)

    >>> left_expected = pd.DataFrame(columns = ['ra', 'dec', 'jd', 'candid', 'trajectory_id'])
    >>> right_expected = pd.DataFrame(columns = ['ra', 'dec', 'jd', 'candid', 'tmp_traj', 'trajectory_id'])

    >>> assert_frame_equal(left, left_expected, check_index_type=False, check_dtype=False)
    >>> assert_frame_equal(right, right_expected, check_index_type=False, check_dtype=False)
    """

    # reset the index of the associated members in order to recovered the right rows after the angle filters.
    traj_assoc = traj_assoc.reset_index(drop=True).reset_index()
    new_obs_assoc = new_obs_assoc.reset_index(drop=True).reset_index()

    # rename the new trajectory_id column to another name in order to give the trajectory_id of the associated trajectories
    # and keep the new trajectory_id
    new_obs_assoc = new_obs_assoc.rename({"trajectory_id": "tmp_traj"}, axis=1)
    new_obs_assoc["trajectory_id"] = traj_assoc["trajectory_id"]

    # get the two last observations of the associated trajectories in order to compute the cone search angle
    two_last = two_last_observations[
        two_last_observations["trajectory_id"].isin(traj_assoc["trajectory_id"])
    ]

    # groupby the two last observations in order to prepare the merge with the new observations
    two_last = two_last.groupby(["trajectory_id"]).agg(
        {"ra": list, "dec": list, "jd": list, "candid": lambda x: len(x)}
    )

    # merge the two last observation with the new observations to be associated
    prep_angle = two_last.merge(
        new_obs_assoc[["index", "ra", "dec", "jd", "trajectory_id"]], on="trajectory_id"
    )

    # compute the cone search angle
    prep_angle["angle"] = prep_angle.apply(angle_df, axis=1)

    # filter by the physical properties angle
    remain_assoc = prep_angle[prep_angle["angle"] <= angle_criterion]

    # keep only the alerts that match with the angle filter
    traj_assoc = traj_assoc.loc[remain_assoc["index"].values]
    new_obs_assoc = new_obs_assoc.loc[remain_assoc["index"].values]

    return traj_assoc.drop(["index"], axis=1), new_obs_assoc.drop(["index"], axis=1)


def night_to_night_observation_association(
    obs_set1,
    obs_set2,
    sep_criterion,
    mag_criterion_same_fid,
    mag_criterion_diff_fid,
    store_kd_tree=False,
):
    """
    Night to night associations between the observations. Take only two set of observations and return couples of observations
    based on separation proximity and magnitude similarity.

    Parameters
    ----------
    obs_set1 : dataframe
        first set of observations (typically, the oldest observations)
    obs_set2 : dataframe
        second set of observations (typically, the newest observations)
    sep_criterion : float
        the separation criterion to associates alerts
    mag_criterion_same_fid : float
        the magnitude criterion to associates alerts if the observations have been observed with the same filter
    mag_criterion_diff_fid : float
        the magnitude criterion to associates alerts if the observations have been observed with the same filter

    Returns
    -------
    traj_assoc : dataframe
        the left members of the associations, can be an extremity of a trajectory
    new_obs_assoc : dataframe
        new observations to add to the associated trajectories

    Examples
    --------

    >>> night_1 = pd.DataFrame({
    ... "ra": [1, 1, 4, 6, 10],
    ... "dec": [1, 8, 4, 7, 2],
    ... "dcmag": [13, 15, 14, 16, 12],
    ... "fid": [1, 2, 1, 2, 2],
    ... "jd": [1, 1, 1, 1, 1],
    ... "candid": [10, 11, 12, 13, 14]
    ... })

    >>> night_2 = pd.DataFrame({
    ... "ra": [2, 6, 5, 7, 7, 8],
    ... "dec": [2, 4, 3, 10, 7, 3],
    ... "dcmag": [13.05, 17, 14.09, 18, 13, 16.4],
    ... "fid": [1, 2, 2, 2, 1, 1],
    ... "jd": [2, 2, 2, 2, 2, 2],
    ... "candid": [15, 16, 17, 18, 19, 20]
    ... })

    >>> left, right = night_to_night_observation_association(night_1, night_2, 1.5 * u.degree, 0.2, 0.5)

    >>> left_expected = pd.DataFrame({
    ... "ra": [1, 4],
    ... "dec": [1, 4],
    ... "dcmag": [13, 14],
    ... "fid": [1, 1],
    ... "jd": [1, 1],
    ... "candid": [10, 12]
    ... })

    >>> right_expected = pd.DataFrame({
    ... "ra": [2, 5],
    ... "dec": [2, 3],
    ... "dcmag": [13.05, 14.09],
    ... "fid": [1, 2],
    ... "jd": [2, 2],
    ... "candid": [15, 17]
    ... })

    >>> assert_frame_equal(left.reset_index(drop=True), left_expected)
    >>> assert_frame_equal(right.reset_index(drop=True), right_expected)
    """

    # association based separation
    traj_assoc, new_obs_assoc, sep = night_to_night_separation_association(
        obs_set1, obs_set2, sep_criterion, store_kd_tree
    )

    # filter the association based on magnitude criterion
    traj_assoc, new_obs_assoc = magnitude_association(
        traj_assoc,
        new_obs_assoc,
        mag_criterion_same_fid,
        mag_criterion_diff_fid,
        jd_normalization=True,
    )

    return traj_assoc, new_obs_assoc


def night_to_night_trajectory_associations(
    two_last_observations,
    observations_to_associates,
    sep_criterion,
    mag_criterion_same_fid,
    mag_criterion_diff_fid,
    angle_criterion,
    store_kd_tree=False,
):
    """
    Associates the extremity of the trajectories with the new observations. The new observations can be a single observations or the extremity of a trajectories.

    Parameters
    ----------
    two_last_observation : dataframe
        the two last observations of the trajectories used to compute the angle for the cone search filtering. The last observations is also used
        to perform the search around sky.
    observations_to_associates : dataframe
        The observations to associates with the extremity of the trajectories. These observations can be single or the extremity of another trajectories.
    sep_criterion : float
        the separation criterion to associates alerts
    mag_criterion_same_fid : float
        the magnitude criterion to associates alerts if the observations have been observed with the same filter
    mag_criterion_diff_fid : float
        the magnitude criterion to associates alerts if the observations have been observed with the same filter
    angle_criterion : float
        the angle criterion to associates alerts during the cone search

    Returns
    -------
    traj_assoc : dataframe
        the left members of the associations, extremity of the trajectories
    new_obs_assoc : dataframe
        new observations to add to the associated trajectories

    Examples
    --------
    >>> left, right = night_to_night_trajectory_associations(ts.night_to_night_two_last_sample, ts.night_to_night_new_observation, 2 * u.degree, 0.2, 0.5, 30)

    >>> assert_frame_equal(left.reset_index(drop=True), ts.night_to_night_traj_assoc_left_expected)
    >>> assert_frame_equal(right.reset_index(drop=True), ts.night_to_night_traj_assoc_right_expected)
    """

    # get the last observations of the trajectories to perform the associations
    last_traj_obs = (
        two_last_observations.groupby(["trajectory_id"]).last().reset_index()
    )

    (traj_assoc, new_obs_assoc) = night_to_night_observation_association(
        last_traj_obs,
        observations_to_associates,
        sep_criterion,
        mag_criterion_same_fid,
        mag_criterion_diff_fid,
        store_kd_tree,
    )

    if len(traj_assoc) != 0:

        traj_assoc, new_obs_assoc = cone_search_association(
            two_last_observations, traj_assoc, new_obs_assoc, angle_criterion
        )

        return traj_assoc, new_obs_assoc
    else:
        return traj_assoc, new_obs_assoc


def tracklets_and_trajectories_associations(
    trajectories,
    tracklets,
    next_nid,
    sep_criterion,
    mag_criterion_same_fid,
    mag_criterion_diff_fid,
    angle_criterion,
    max_traj_id,
    store_kd_tree=False,
):
    """
    Perform trajectories associations with tracklets detected in the next night.

    Parameters
    ----------
    trajectories : dataframe
        trajectories detected previously from the old night by the algorithm.
        Trajectories dataframe must have the following columns :
            ra, dec, dcmag, nid, fid, jd, candid, trajectory_id
    tracklets : dataframe
        tracklets detected in the next night
        Tracklets dataframe must have the following columns :
            ra, dec, dcmag, nid, fid, jd, candid, trajectory_id
            N.B : the id in the trajectory_id column for the tracklets are temporary.
    next_nid : The next night id which is the night id of the tracklets.
    sep_criterion : float
        the separation criterion for the alert based position associations
    mag_criterion_same_fid : float
        the magnitude criterion to associates alerts if the observations have been observed with the same filter
    mag_criterion_diff_fid : float
        the magnitude criterion to associates alerts if the observations have been observed with the same filter
    angle_criterion : float
        the angle criterion to associates alerts during the cone search

    Return
    ------
    trajectories : dataframe
        trajectories associated with new tracklets
    tracklets : dataframe
        remaining tracklets

    Examples
    --------
    >>> trajectories = ts.trajectory_df_sample
    >>> tracklets = ts.traj_next_night_sample

    >>> trajectories["not_updated"] = np.ones(len(trajectories), dtype=np.bool_)

    >>> tr, tk, max_tr_id = tracklets_and_trajectories_associations(trajectories, tracklets, 4, 2 * u.degree, 0.2, 0.5, 30, 6)

    >>> assert_frame_equal(tr, ts.trajectories_expected_1, check_dtype=False)
    >>> assert_frame_equal(tk, ts.tracklets_expected_1, check_dtype=False)
    >>> max_tr_id
    6

    >>> trajectories = ts.trajectories_sample_1
    >>> tracklets = ts.tracklets_sample_1

    >>> trajectories["not_updated"] = np.ones(len(trajectories), dtype=np.bool_)

    >>> tr, tk, max_tr_id = tracklets_and_trajectories_associations(trajectories, tracklets, 3, 1.5 * u.degree, 0.2, 0.5, 30, 5)

    >>> assert_frame_equal(tr, ts.trajectories_expected_2, check_dtype=False)
    >>> assert_frame_equal(tk, ts.tracklets_expected_2, check_dtype=False)
    >>> max_tr_id
    5

    >>> tracklets = ts.tracklets_sample_2
    >>> trajectories = ts.trajectories_sample_2

    >>> trajectories["not_updated"] = np.ones(len(trajectories), dtype=np.bool_)

    >>> tr, tk, max_tr_id = tracklets_and_trajectories_associations(trajectories, tracklets, 3, 1 * u.degree, 0.2, 0.5, 30, 5)

    >>> assert_frame_equal(tr, ts.trajectories_expected_3, check_dtype=False)
    >>> assert_frame_equal(tk, pd.DataFrame(columns=["ra", "dec", "dcmag", "fid", "nid", "jd", "candid", "trajectory_id", "assoc_tag"]), check_index_type=False, check_dtype=False)
    >>> max_tr_id
    8

    >>> tr, tk, max_tr_id = tracklets_and_trajectories_associations(pd.DataFrame(), tracklets, 3, 1 * u.degree, 0.2, 0.5, 30, 0)

    >>> assert_frame_equal(tr, pd.DataFrame(), check_dtype=False)
    >>> assert_frame_equal(tk, tracklets)
    >>> max_tr_id
    0
    """

    if len(trajectories) == 0 or len(tracklets) == 0:
        return trajectories, tracklets, max_traj_id
    else:

        trajectories_not_updated = trajectories[trajectories["not_updated"]]

        # get the last two observations for each trajectories
        two_last_observation_trajectory = get_n_last_observations_from_trajectories(
            trajectories_not_updated, 2
        )

        # get the last observations for each trajectories
        last_observation_trajectory = get_n_last_observations_from_trajectories(
            trajectories_not_updated, 1
        )

        # get the recently extremity of the new tracklets to perform associations with the latest observations in the trajectories
        tracklets_extremity = get_n_last_observations_from_trajectories(
            tracklets, 1, False
        )

        # perform association with all previous nid within the time window
        # Warning : sort by descending order to do the association with the recently previous night in first.
        trajectories_nid = np.sort(np.unique(last_observation_trajectory["nid"]))[::-1]

        # for each trajectory nid from the last observations
        for tr_nid in trajectories_nid:

            # get the two last observations with the tracklets extremity that have the current nid
            two_last_current_nid = two_last_observation_trajectory[
                two_last_observation_trajectory["trajectory_id"].isin(
                    last_observation_trajectory[
                        last_observation_trajectory["nid"] == tr_nid
                    ]["trajectory_id"]
                )
            ]

            diff_night = next_nid - tr_nid
            norm_sep_crit = sep_criterion * diff_night
            norm_same_fid = mag_criterion_same_fid * diff_night
            norm_diff_fid = mag_criterion_diff_fid * diff_night

            # trajectory association with the new tracklets
            (
                traj_left,
                traj_extremity_associated,
            ) = night_to_night_trajectory_associations(
                two_last_current_nid,
                tracklets_extremity,
                norm_sep_crit,
                norm_same_fid,
                norm_diff_fid,
                angle_criterion,
                store_kd_tree,
            )

            if len(traj_extremity_associated) > 0:

                # creates a dataframe for each duplicated trajectory associated with the tracklets
                duplicates = traj_extremity_associated["trajectory_id"].duplicated()

                # get the duplicated tracklets
                tracklets_duplicated = traj_extremity_associated[duplicates]

                if len(tracklets_duplicated) > 0:

                    # get the trajectories involved in the duplicates
                    duplicate_traj = (
                        trajectories[
                            trajectories["trajectory_id"].isin(
                                tracklets_duplicated["trajectory_id"]
                            )
                        ]
                        .sort_values(["trajectory_id", "jd"])
                        .reset_index(drop=True)
                    )

                    # get the tracklets involved in the duplicates
                    duplicate_track = (
                        tracklets[
                            tracklets["trajectory_id"].isin(
                                tracklets_duplicated["tmp_traj"]
                            )
                        ]
                        .sort_values(["trajectory_id", "jd"])
                        .reset_index(drop=True)
                    )

                    # compute the size of each trajectories
                    traj_size = (
                        duplicate_traj.groupby(["trajectory_id"]).count()["ra"].values
                    )

                    # compute the size of each tracklets
                    track_size = (
                        duplicate_track.groupby(["trajectory_id"]).count()["ra"].values
                    )

                    # compute the number of time each trajectories need to be duplicates
                    _, traj_counts_duplicates = np.unique(
                        tracklets_duplicated["trajectory_id"], return_counts=True
                    )

                    # compute the occurence of each duplicated tracklets
                    _, track_counts_duplicates = np.unique(
                        tracklets_duplicated["tmp_traj"], return_counts=True
                    )

                    tr_index = duplicate_traj.index.values
                    tk_index = duplicate_track.index.values

                    # duplicates each trajectory according to the numbers of duplicates for each of them
                    # solution for the next instruction taken from :
                    # https://stackoverflow.com/questions/63510977/repeat-but-in-variable-sized-chunks-in-numpy

                    tr_index = repeat_chunk(tr_index, traj_size, traj_counts_duplicates)
                    tr_df = duplicate_traj.loc[tr_index]

                    tk_index = repeat_chunk(
                        tk_index, track_size, track_counts_duplicates
                    )

                    tk_df = duplicate_track.loc[tk_index]

                    # compute the new trajectory id
                    new_obs_id = np.arange(
                        max_traj_id, max_traj_id + len(tracklets_duplicated)
                    )

                    max_traj_id += len(tracklets_duplicated)

                    # repeat all the trajectory_id according to the size of each trajectory and
                    # the number of time they need to be repeated
                    tr_id_repeat = np.repeat(traj_size, traj_counts_duplicates)
                    new_tr_id = np.repeat(new_obs_id, tr_id_repeat)

                    track_dp_size = np.repeat(track_size, track_counts_duplicates)
                    new_obs_id = np.repeat(new_obs_id, track_dp_size)

                    # assign the new trajectory_id
                    # silence the copy warning
                    with pd.option_context("mode.chained_assignment", None):
                        tr_df["trajectory_id"] = new_tr_id
                        tr_df["not_updated"] = False

                        tk_df["trajectory_id"] = new_obs_id
                        tk_df["not_updated"] = False

                    tk_df["assoc_tag"] = "T"
                    # add the duplicated new trajectories to the set of trajectories
                    all_duplicate_traj = cast_obs_data(pd.concat([tr_df, tk_df]))

                    trajectories = cast_obs_data(
                        pd.concat([trajectories, all_duplicate_traj])
                    )

                    # remove the duplicates tracklets associated with a trajectory from the
                    # set of tracklets detected this night.
                    tracklets = tracklets[
                        ~tracklets["trajectory_id"].isin(
                            tracklets_duplicated["tmp_traj"]
                        )
                    ]

                    # Becareful to remove also the duplicated tracklets extremity
                    # that have been associated during this loop
                    tracklets_extremity = tracklets_extremity[
                        ~tracklets_extremity["trajectory_id"].isin(
                            tracklets_duplicated["tmp_traj"]
                        )
                    ]

                # remove duplicates associations
                traj_extremity_associated = traj_extremity_associated[~duplicates]

                associated_tracklets = []

                # If too slow, need to be optimized/vectorized
                for _, rows in traj_extremity_associated.iterrows():

                    # get all rows of the associated tracklets of the next night
                    next_night_tracklets = tracklets[
                        tracklets["trajectory_id"] == rows["tmp_traj"]
                    ]

                    # assign the trajectory id to the tracklets that will be added to this trajectory with this id
                    # the tracklets contains already the alerts within traj_extremity_associated.
                    with pd.option_context("mode.chained_assignment", None):
                        next_night_tracklets["trajectory_id"] = rows["trajectory_id"]

                    associated_tracklets.append(next_night_tracklets)

                # create a dataframe with all tracklets that will be added to a trajectory
                associated_tracklets = cast_obs_data(pd.concat(associated_tracklets))
                associated_tracklets["assoc_tag"] = "T"

                # remove the tracklets that will be added to a trajectory from the dataframe of all tracklets
                tracklets = tracklets[
                    ~tracklets["trajectory_id"].isin(
                        traj_extremity_associated["tmp_traj"]
                    )
                ]

                # concatenation of trajectories with new tracklets
                trajectories = cast_obs_data(
                    pd.concat([trajectories, associated_tracklets])
                )

                # keep trace of the updated trajectories
                # get the trajectory_id of the updated trajectories
                associated_tr_id = np.unique(associated_tracklets["trajectory_id"])
                trajectories = trajectories.reset_index(drop=True)

                # get all observations of the updated trajectories
                tr_updated_index = trajectories[
                    trajectories["trajectory_id"].isin(associated_tr_id)
                ].index
                # update the updated status
                trajectories.loc[tr_updated_index, "not_updated"] = False

                # remove the two last trajectory observation that have been associated during this loop.
                two_last_current_nid = two_last_current_nid[
                    ~two_last_current_nid["trajectory_id"].isin(
                        traj_left["trajectory_id"]
                    )
                ]

                # remove also the tracklets extremity that have been associated during this loop
                tracklets_extremity = tracklets_extremity[
                    ~tracklets_extremity["trajectory_id"].isin(
                        traj_extremity_associated["tmp_traj"]
                    )
                ]

        return (
            cast_obs_data(trajectories).reset_index(drop=True),
            cast_obs_data(tracklets).reset_index(drop=True),
            max_traj_id,
        )


def trajectories_with_new_observations_associations(
    trajectories,
    new_observations,
    next_nid,
    sep_criterion,
    mag_criterion_same_fid,
    mag_criterion_diff_fid,
    angle_criterion,
    max_traj_id,
    store_kd_tree=False,
):
    """
    Perform trajectories associations with the remaining new observations return by the intra night associations

    Parameters
    ----------
    trajectories : dataframe
        trajectories detected previously from the old night by the algorithm.
        Trajectories dataframe must have the following columns :
            ra, dec, dcmag, nid, fid, jd, candid, trajectory_id
    new_observations : dataframe
        new observations from the next night and not associated during the intra night process and the
        new observations dataframe must have the following columns :
            ra, dec, dcmag, nid, fid, jd, candid
    next_nid : The next night id which is the night id of the tracklets.
    sep_criterion : float
        the separation criterion for the alert based position associations
    mag_criterion_same_fid : float
        the magnitude criterion to associates alerts if the observations have been observed with the same filter
    mag_criterion_diff_fid : float
        the magnitude criterion to associates alerts if the observations have been observed with the same filter
    angle_criterion : float
        the angle criterion to associates alerts during the cone search

    Return
    ------
    trajectories : dataframe
        trajectories associated with new observations
    new_observations : dataframe
        remaining new observations

    Examples
    --------
    >>> trajectories = ts.trajectories_sample_3
    >>> new_observations = ts.new_observations_sample_1

    >>> trajectories["not_updated"] = np.ones(len(trajectories), dtype=np.bool_)

    >>> tr, obs, max_tr_id = trajectories_with_new_observations_associations(
    ... trajectories, new_observations, 3, 1.5 * u.degree, 0.2, 0.5, 30, 6
    ... )

    >>> assert_frame_equal(tr, ts.trajectories_expected_4, check_dtype=False)
    >>> assert_frame_equal(obs, ts.new_observations_expected_1, check_dtype=False)
    >>> max_tr_id
    6

    >>> trajectories = ts.trajectories_sample_4
    >>> new_observations = ts.new_observations_sample_2

    >>> trajectories["not_updated"] = np.ones(len(trajectories), dtype=np.bool_)

    >>> tr, obs, max_tr_id = trajectories_with_new_observations_associations(
    ... trajectories, new_observations, 3, 1.5 * u.degree, 0.2, 0.5, 30, 4
    ... )

    >>> assert_frame_equal(tr, ts.trajectories_expected_5, check_dtype=False)
    >>> assert_frame_equal(obs, ts.new_observations_expected_2, check_dtype=False)
    >>> max_tr_id
    8

    >>> trajectories = ts.trajectories_sample_5
    >>> new_observations = ts.new_observations_sample_3

    >>> trajectories["not_updated"] = np.ones(len(trajectories), dtype=np.bool_)

    >>> tr, obs, max_tr_id = trajectories_with_new_observations_associations(
    ... trajectories, new_observations, 3, 1.5 * u.degree, 0.2, 0.5, 30, 7, True
    ... )

    >>> assert_frame_equal(tr, ts.trajectories_expected_6, check_dtype=False)
    >>> assert_frame_equal(obs, ts.new_observations_expected_3, check_dtype=False)
    >>> max_tr_id
    7

    >>> tr, obs, max_tr_id = trajectories_with_new_observations_associations(
    ... pd.DataFrame(), new_observations, 3, 1.5 * u.degree, 0.2, 0.5, 30, 0
    ... )

    >>> assert_frame_equal(tr, pd.DataFrame(), check_dtype=False)
    >>> assert_frame_equal(obs, new_observations, check_dtype=False)
    >>> max_tr_id
    0
    """

    if len(trajectories) == 0 or len(new_observations) == 0:
        return trajectories, new_observations, max_traj_id
    else:

        # get only the trajectories not updated by a previous step
        trajectories_not_updated = trajectories[trajectories["not_updated"]]

        # get the last two observations for each trajectories
        two_last_observation_trajectory = get_n_last_observations_from_trajectories(
            trajectories_not_updated, 2
        )

        # get the last observations for each trajectories
        last_observation_trajectory = get_n_last_observations_from_trajectories(
            trajectories_not_updated, 1
        )

        # perform association with all previous nid within the time window
        # Warning : sort by descending order to do the association with the recently previous night in first.
        trajectories_nid = np.sort(np.unique(last_observation_trajectory["nid"]))[::-1]

        # for each trajectory nid from the last observations
        for tr_nid in trajectories_nid:

            # get the two last observations with the tracklets extremity that have the current nid
            two_last_current_nid = two_last_observation_trajectory[
                two_last_observation_trajectory["trajectory_id"].isin(
                    last_observation_trajectory[
                        last_observation_trajectory["nid"] == tr_nid
                    ]["trajectory_id"]
                )
            ]

            diff_night = next_nid - tr_nid
            norm_sep_crit = sep_criterion * diff_night
            norm_same_fid = mag_criterion_same_fid * diff_night
            norm_diff_fid = mag_criterion_diff_fid * diff_night

            # trajectory associations with the new observations
            (traj_left, obs_assoc) = night_to_night_trajectory_associations(
                two_last_current_nid,
                new_observations,
                norm_sep_crit,
                norm_same_fid,
                norm_diff_fid,
                angle_criterion,
                store_kd_tree,
            )

            if "tmp_traj" in obs_assoc:  # pragma: no cover
                obs_assoc = obs_assoc.drop(["tmp_traj"], axis=1)

            # remove the associateds observations from the set of new observations
            new_observations = new_observations[
                ~new_observations["candid"].isin(obs_assoc["candid"])
            ]

            if len(obs_assoc) > 0:

                # creates a dataframe for each duplicated trajectory associated with the tracklets
                duplicates = obs_assoc["trajectory_id"].duplicated()

                # get the duplicated tracklets
                duplicate_obs = obs_assoc[duplicates]

                if len(duplicate_obs) > 0:

                    # get the trajectories involved in the duplicates
                    duplicate_traj = (
                        trajectories[
                            trajectories["trajectory_id"].isin(
                                duplicate_obs["trajectory_id"]
                            )
                        ]
                        .sort_values(["trajectory_id", "jd"])
                        .reset_index(drop=True)
                    )

                    # compute the size of each trajectories
                    traj_size = (
                        duplicate_traj.groupby(["trajectory_id"]).count()["ra"].values
                    )

                    # compute the number of time each trajectories need to be duplicates
                    _, traj_counts_duplicates = np.unique(
                        duplicate_obs["trajectory_id"], return_counts=True
                    )

                    tr_index = duplicate_traj.index.values

                    # # duplicates each trajectory according to the numbers of duplicates for each of them
                    # # solution for the next instruction taken from :
                    # # https://stackoverflow.com/questions/63510977/repeat-but-in-variable-sized-chunks-in-numpy

                    dp_index = repeat_chunk(tr_index, traj_size, traj_counts_duplicates)
                    df = duplicate_traj.loc[dp_index]

                    # compute the new trajectory id
                    new_obs_id = np.arange(
                        max_traj_id, max_traj_id + len(duplicate_obs)
                    )

                    max_traj_id += len(duplicate_obs)

                    # repeat all the trajectory_id according to the size of each trajectory and
                    # the number of time they need to be repeated
                    tr_id_repeat = np.repeat(traj_size, traj_counts_duplicates)
                    new_tr_id = np.repeat(new_obs_id, tr_id_repeat)

                    # assign the new trajectory_id
                    # silence the copy warning
                    with pd.option_context("mode.chained_assignment", None):
                        df["trajectory_id"] = new_tr_id
                        df["not_updated"] = False

                        duplicate_obs["trajectory_id"] = new_obs_id
                        duplicate_obs["not_updated"] = False

                        duplicate_obs["assoc_tag"] = "A"

                    # add the duplicated new trajectories to the set of trajectories
                    all_duplicate_traj = cast_obs_data(pd.concat([df, duplicate_obs]))

                    trajectories = cast_obs_data(
                        pd.concat([trajectories, all_duplicate_traj])
                    )

                # remove duplicates associations
                obs_assoc = obs_assoc[~duplicates]
                obs_assoc["assoc_tag"] = "A"

                # add the new associated observations in the recorded trajectory dataframe
                trajectories = cast_obs_data(pd.concat([trajectories, obs_assoc]))

                # keep trace of the updated trajectories
                # get the trajectory_id of the updated trajectories
                associated_tr_id = np.unique(obs_assoc["trajectory_id"])
                trajectories = trajectories.reset_index(drop=True)
                # get all observations of the updated trajectories
                tr_updated_index = trajectories[
                    trajectories["trajectory_id"].isin(associated_tr_id)
                ].index
                # update the updated status
                trajectories.loc[tr_updated_index, "not_updated"] = False

        return (
            cast_obs_data(trajectories).reset_index(drop=True),
            cast_obs_data(new_observations).reset_index(drop=True),
            max_traj_id,
        )


def old_observations_with_tracklets_associations(
    tracklets,
    old_observations,
    next_nid,
    sep_criterion,
    mag_criterion_same_fid,
    mag_criterion_diff_fid,
    angle_criterion,
    max_traj_id,
    store_kd_tree=False,
):
    """
    Perform associations between the tracklets and the old observations from the previous nights.

    Parameters
    ----------
    tracklets : dataframe
        tracklets detected previously from the intra night associations and not associated with a recorded trajectories.
        Tracklets dataframe must have the following columns :
            ra, dec, dcmag, nid, fid, jd, candid, trajectory_id
    old_observations : dataframe
        old observations from the previous nights.
        old observations dataframe must have the following columns :
            ra, dec, dcmag, nid, fid, jd, candid
    next_nid : The next night id which is the night id of the tracklets.
    sep_criterion : float
        the separation criterion for the alert based position associations
    mag_criterion_same_fid : float
        the magnitude criterion to associates alerts if the observations have been observed with the same filter
    mag_criterion_diff_fid : float
        the magnitude criterion to associates alerts if the observations have been observed with the same filter
    angle_criterion : float
        the angle criterion to associates alerts during the cone search

    Return
    ------
    tracklets : dataframe
        tracklets associated with old observations
    old observations : dataframe
        remaining old observations

    Examples
    --------
    >>> tracklets = ts.tracklets_sample_3
    >>> old_observations = ts.old_observations_sample_1

    >>> tracklets["not_updated"] = np.ones(len(tracklets), dtype=np.bool_)

    >>> tk, old, max_tr_id = old_observations_with_tracklets_associations(tracklets, old_observations, 3, 1.5 * u.degree, 0.1, 0.3, 30, 6)


    >>> assert_frame_equal(tk, ts.tracklets_obs_expected_1, check_dtype=False)
    >>> assert_frame_equal(old.reset_index(drop=True), ts.old_obs_expected_1, check_dtype=False)
    >>> max_tr_id
    6

    >>> tracklets = ts.tracklets_sample_4
    >>> old_observations = ts.old_observations_sample_2

    >>> tracklets["not_updated"] = np.ones(len(tracklets), dtype=np.bool_)

    >>> tk, old, max_tr_id = old_observations_with_tracklets_associations(tracklets, old_observations, 3, 1.5 * u.degree, 0.1, 0.3, 30, 5)

    >>> assert_frame_equal(tk, ts.tracklets_obs_expected_2, check_dtype=False)
    >>> assert_frame_equal(old.reset_index(drop=True), ts.old_obs_expected_2, check_dtype=False)
    >>> max_tr_id
    6

    >>> tracklets = ts.tracklets_sample_5
    >>> old_observations = ts.old_observations_sample_3

    >>> tracklets["not_updated"] = np.ones(len(tracklets), dtype=np.bool_)

    >>> tk, old, max_tr_id = old_observations_with_tracklets_associations(tracklets, old_observations, 3, 1.5 * u.degree, 0.1, 0.3, 30, 3)

    >>> assert_frame_equal(tk, ts.tracklets_obs_expected_3, check_dtype=False)
    >>> assert_frame_equal(old.reset_index(drop=True), ts.old_obs_expected_3, check_dtype=False)
    >>> max_tr_id
    5

    >>> tk, old, max_tr_id = old_observations_with_tracklets_associations(tracklets, pd.DataFrame(), 3, 1.5 * u.degree, 0.1, 0.3, 30, 3)

    >>> assert_frame_equal(tk, tracklets, check_dtype=False)
    >>> assert_frame_equal(old, pd.DataFrame(), check_dtype=False)
    >>> max_tr_id
    3
    """

    if len(tracklets) == 0 or len(old_observations) == 0:
        return tracklets, old_observations, max_traj_id
    else:

        # get all the old night id sort by descending order to begin the associations
        # with the recently ones
        old_obs_nid = np.sort(np.unique(old_observations["nid"]))[::-1]

        two_first_obs_tracklets = get_n_last_observations_from_trajectories(
            tracklets, 2, False
        )

        for obs_nid in old_obs_nid:

            current_old_obs = old_observations[old_observations["nid"] == obs_nid]

            diff_night = next_nid - obs_nid
            norm_sep_crit = sep_criterion * diff_night
            norm_same_fid = mag_criterion_same_fid * diff_night
            norm_diff_fid = mag_criterion_diff_fid * diff_night

            # association between the new tracklets and the old observations
            (
                track_left_assoc,
                old_obs_right_assoc,
            ) = night_to_night_trajectory_associations(
                two_first_obs_tracklets,
                current_old_obs,
                norm_sep_crit,
                norm_same_fid,
                norm_diff_fid,
                angle_criterion,
                store_kd_tree,
            )

            # remove tmp_traj column added by the cone_search_associations function in  night_to_night_trajectory_associations
            if "tmp_traj" in old_obs_right_assoc:  # pragma: no cover
                old_obs_right_assoc = old_obs_right_assoc.drop("tmp_traj", axis=1)

            # remove the associateds observations from the set of new observations
            old_observations = old_observations[
                ~old_observations["candid"].isin(old_obs_right_assoc["candid"])
            ]

            if len(track_left_assoc) > 0:

                # creates a dataframe for each duplicated trajectory associated with the tracklets
                duplicates = old_obs_right_assoc["trajectory_id"].duplicated()

                # get the duplicated tracklets
                duplicate_obs = old_obs_right_assoc[duplicates]

                if len(duplicate_obs) > 0:

                    # get the tracklets involved with duplicates
                    duplicate_traj = (
                        tracklets[
                            tracklets["trajectory_id"].isin(
                                duplicate_obs["trajectory_id"]
                            )
                        ]
                        .sort_values(["trajectory_id", "jd"])
                        .reset_index(drop=True)
                    )

                    # compute size of each tracklets
                    traj_size = (
                        duplicate_traj.groupby(["trajectory_id"]).count()["ra"].values
                    )

                    # compute the number of duplicates for each tracklets
                    _, traj_counts_duplicates = np.unique(
                        duplicate_obs["trajectory_id"], return_counts=True
                    )

                    tr_index = duplicate_traj.index

                    # duplicates each trajectory according to the number of duplicates for each of them
                    # solution for the next instruction taken from :
                    # https://stackoverflow.com/questions/63510977/repeat-but-in-variable-sized-chunks-in-numpy
                    dp_index = repeat_chunk(tr_index, traj_size, traj_counts_duplicates)

                    # apply the duplicated index to the tracklets dataframe
                    df = duplicate_traj.loc[dp_index]

                    # compute the trajectory_id for each new tracklets
                    new_obs_id = np.arange(
                        max_traj_id, max_traj_id + len(duplicate_obs)
                    )

                    max_traj_id += len(duplicate_obs)

                    # duplicates the trajectory_id for each duplicates tracklets
                    tr_id_repeat = np.repeat(traj_size, traj_counts_duplicates)
                    new_tr_id = np.repeat(new_obs_id, tr_id_repeat)

                    # assign the new list of trajectory_id
                    # silence the copy warning
                    with pd.option_context("mode.chained_assignment", None):
                        df["trajectory_id"] = new_tr_id
                        df["not_updated"] = False

                        duplicate_obs["trajectory_id"] = new_obs_id
                        duplicate_obs["not_updated"] = False

                        duplicate_obs["assoc_tag"] = "O"

                    all_duplicate_track = cast_obs_data(pd.concat([df, duplicate_obs]))
                    tracklets = cast_obs_data(
                        pd.concat([tracklets, all_duplicate_track])
                    )

                # remove duplicates associations
                old_obs_right_assoc = old_obs_right_assoc[~duplicates]
                old_obs_right_assoc["assoc_tag"] = "O"

                # add the associated old observations to the tracklets
                tracklets = cast_obs_data(pd.concat([tracklets, old_obs_right_assoc]))

                # remove the associated tracklets for the next loop turn
                two_first_obs_tracklets = two_first_obs_tracklets[
                    ~two_first_obs_tracklets["trajectory_id"].isin(
                        track_left_assoc["trajectory_id"]
                    )
                ]

                # keep trace of the updated trajectories
                # get the trajectory_id of the updated trajectories
                associated_tr_id = np.unique(old_obs_right_assoc["trajectory_id"])
                tracklets = tracklets.reset_index(drop=True)
                # get all observations of the updated trajectories
                tr_updated_index = tracklets[
                    tracklets["trajectory_id"].isin(associated_tr_id)
                ].index
                # update the updated status
                tracklets.loc[tr_updated_index, "not_updated"] = False

                # remove the associated old observations
                old_observations = old_observations[
                    ~old_observations["candid"].isin(old_obs_right_assoc["candid"])
                ]

                # not usefull due to modification of current_old_obs each loop turn
                # current_old_obs = current_old_obs[
                #     ~current_old_obs["candid"].isin(old_obs_right_assoc["candid"])
                # ]

        return (
            cast_obs_data(tracklets).reset_index(drop=True),
            cast_obs_data(old_observations).reset_index(drop=True),
            max_traj_id,
        )


def old_with_new_observations_associations(
    old_observations,
    new_observations,
    next_nid,
    last_trajectory_id,
    sep_criterion,
    mag_criterion_same_fid,
    mag_criterion_diff_fid,
    store_kd_tree=False,
):
    """
    Perform inter night association between the remaining old observation return by the other associations with the remaining new observations.

    Parameters
    ----------
    old_observations : dataframe
        remaining old observations return by old_observations_associations
        old_observations dataframe must have the following columns :
            ra, dec, dcmag, nid, fid, jd, candid
    new_observations : dataframe
        remaining new observations return by trajectories_associations
        new_observations dataframe must have the following columns :
            ra, dec, dcmag, nid, fid, jd, candid
    next_nid : The next night id which is the night id of the tracklets.
    last_trajectory_id : integer
        the last trajectory identifier assign to a trajectory in the trajectories dataframe (currently, it is just the maximum).
    sep_criterion : float
        the separation criterion for the alert based position associations
    mag_criterion_same_fid : float
        the magnitude criterion to associates alerts if the observations have been observed with the same filter
    mag_criterion_diff_fid : float
        the magnitude criterion to associates alerts if the observations have been observed with the same filter
    angle_criterion : float
        the angle criterion to associates alerts during the cone search

    Return
    ------
    trajectory_df : dataframe
        the new trajectory detected by observations associations,
        They are only two points trajeectory so we don't compute orbital elements with this trajectory_df
    old_observations : dataframe
        remaining old observations after associations
    new_observations : dataframe
        remaining new observations after associations

    Examples
    --------
    >>> new_tr, remain_old, remain_new = old_with_new_observations_associations(ts.old_observations_sample_4, ts.new_observation_sample_4, 3, 0, 1.5 * u.degree, 0.1, 0.3)

    >>> assert_frame_equal(new_tr.reset_index(drop=True), ts.new_trajectory, check_dtype=False)
    >>> assert_frame_equal(remain_old.reset_index(drop=True), ts.remaining_old_obs, check_dtype=False)
    >>> assert_frame_equal(remain_new.reset_index(drop=True), ts.remaining_new_obs, check_dtype=False)

    >>> new_tr, remain_old, remain_new = old_with_new_observations_associations(pd.DataFrame(), ts.new_observation_sample_4, 3, 0, 1.5 * u.degree, 0.1, 0.3)
    >>> assert_frame_equal(new_tr, pd.DataFrame(), check_dtype=False)
    >>> assert_frame_equal(remain_old, pd.DataFrame(), check_dtype=False)
    >>> assert_frame_equal(remain_new.reset_index(drop=True), ts.new_observation_sample_4, check_dtype=False)
    """
    if len(old_observations) == 0 or len(new_observations) == 0:
        return pd.DataFrame(), old_observations, new_observations
    else:

        old_obs_nid = np.sort(np.unique(old_observations["nid"]))[::-1]
        trajectory_df = pd.DataFrame()

        for obs_nid in old_obs_nid:

            current_old_obs = old_observations[old_observations["nid"] == obs_nid]

            diff_night = next_nid - obs_nid
            norm_sep_crit = sep_criterion * diff_night
            norm_same_fid = mag_criterion_same_fid * diff_night
            norm_diff_fid = mag_criterion_diff_fid * diff_night

            (left_assoc, right_assoc) = night_to_night_observation_association(
                current_old_obs,
                new_observations,
                norm_sep_crit,
                norm_same_fid,
                norm_diff_fid,
                store_kd_tree,
            )

            if len(left_assoc) > 0:

                new_trajectory_id = np.arange(
                    last_trajectory_id, last_trajectory_id + len(left_assoc)
                )

                last_trajectory_id = last_trajectory_id + len(left_assoc)

                # remove the associated old observation
                old_observations = old_observations[
                    ~old_observations["candid"].isin(left_assoc["candid"])
                ]

                # remove the associated new observation
                new_observations = new_observations[
                    ~new_observations["candid"].isin(right_assoc["candid"])
                ]

                # assign a new trajectory id to the new association
                left_assoc["trajectory_id"] = right_assoc[
                    "trajectory_id"
                ] = new_trajectory_id

                left_assoc["assoc_tag"] = right_assoc["assoc_tag"] = "N"

                trajectory_df = cast_obs_data(
                    pd.concat([trajectory_df, left_assoc, right_assoc])
                )

        return (trajectory_df, old_observations, new_observations)


def time_window_management(
    trajectory_df,
    old_observation,
    last_nid,
    nid_next_night,
    traj_time_window,
    obs_time_window,
    traj_2_points_time_windows,
    keep_last=False,
):
    """
    Management of the old observation and trajectories.

    Remove the old observation when the nid difference between the nid of the next night
    and the nid of the old observation are greater than the time window.

    Perform the same process for trajectories but take the most recent trajectory extremity.

    Remove also the trajectories with no computed orbit element.

    If a number of night without observation exceeds the time window parameters, keep the observations and trajectories
    from the night before the non observation gap.

    Parameters
    ----------
    trajectory_df : dataframe
        all recorded trajectories
    old_observation : dataframe
        old observation from previous night
    last_nid : integer
        nid of the previous observation night
    nid_next_night : integer
        nid of the incoming night
    traj_time_window : integer
        limit to keep the recorded trajectories
    obs_time_window : integer
        limit to keep the old observations
    traj_2_points_time_windows : integer
        limit to keep the trajectories of two points.
        These are observations detected during the observations association step and are not accurate.
        To limit the combinatorial, keep them less time than the other trajectories with more points.
    orbfit_limit : integer
        The number of points required to send trajectories to the orbit fitting program.
        Remove the trajectories with more point than "orbfit limit" points and without orbital elements.
        Keep the trajectories with less than "orbfit limit" points
    keep_last : boolean
        Is set to true, keep the trajectories and the old observation of the last observation night
        irrespective of the traj_time_window and old_obs_time_window.

    Return
    ------
    oldest_traj : dataframe
        the trajectories older than the time window
    most_recent_traj : dataframe
        the trajectories with an nid from an extremity observation smaller than the time window
    old_observation : dataframe
        all the old observation with a difference of nid with the next incoming nid smaller than
        the time window

    Examples
    --------
    >>> test_traj = pd.DataFrame({
    ... "candid" : [10, 11, 12, 13, 14, 15],
    ... "nid" : [1, 2, 3, 10, 11, 12],
    ... "jd" : [1, 2, 3, 10, 11, 12],
    ... "trajectory_id" : [1, 1, 1, 2, 2, 2]
    ... })

    >>> test_obs = pd.DataFrame({
    ... "nid" : [1, 4, 11, 12],
    ... "candid" : [16, 17, 18, 19]
    ... })

    >>> (test_old_traj, test_most_recent_traj), test_old_obs = time_window_management(test_traj, test_obs, 12, 17, 3, 3, 3, True)

    >>> expected_old_traj = pd.DataFrame({
    ... "candid" : [10, 11, 12],
    ... "nid" : [1, 2, 3],
    ... "jd" : [1, 2, 3],
    ... "trajectory_id" : [1, 1, 1]
    ... })

    >>> expected_most_recent_traj = pd.DataFrame({
    ... "candid" : [13, 14, 15],
    ... "nid" : [10, 11, 12],
    ... "jd" : [10, 11, 12],
    ... "trajectory_id" : [2, 2, 2]
    ... })

    >>> assert_frame_equal(expected_old_traj.reset_index(drop=True), test_old_traj.reset_index(drop=True))
    >>> assert_frame_equal(expected_most_recent_traj.reset_index(drop=True), test_most_recent_traj.reset_index(drop=True))
    >>> assert_frame_equal(pd.DataFrame(columns=["nid"]), test_old_obs)

    >>> test_traj = pd.DataFrame({
    ... "candid" : [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
    ... "nid" : [1, 2, 3, 10, 11, 12, 13, 14, 15, 16, 17, 18, 17, 18, 13, 14, 3, 8, 12, 16],
    ... "jd" : [1, 2, 3, 10, 11, 12, 13, 14, 15, 16, 17, 18, 17, 18, 13, 14, 3, 8, 12, 16],
    ... "trajectory_id" : [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 6, 6, 7, 7, 7, 7]
    ... })

    >>> test_obs = pd.DataFrame({
    ... "nid" : [1, 4, 11, 12, 14, 15, 17, 18, 13],
    ... "candid" : [22, 23, 24, 25, 26, 27, 28, 29, 30]
    ... })

    >>> (test_old_traj, test_most_recent_traj), test_old_obs = time_window_management(test_traj, test_obs, 15, 18, 5, 4, 3)

    >>> expected_old_traj = pd.DataFrame({'candid': [10, 11, 12, 13, 14, 15], 'nid': [1, 2, 3, 10, 11, 12], 'jd': [1, 2, 3, 10, 11, 12], 'trajectory_id': [1, 1, 1, 2, 2, 2]})

    >>> expected_most_recent_traj = pd.DataFrame({'candid': [16, 17, 18, 19, 20, 21, 22, 23, 26, 27, 28, 29], 'nid': [13, 14, 15, 16, 17, 18, 17, 18, 3, 8, 12, 16], 'jd': [13, 14, 15, 16, 17, 18, 17, 18, 3, 8, 12, 16], 'trajectory_id': [3, 3, 3, 4, 4, 4, 5, 5, 7, 7, 7, 7]})

    >>> expected_old_obs = pd.DataFrame({'nid': [14, 15, 17, 18], 'candid': [26, 27, 28, 29]})

    >>> assert_frame_equal(expected_old_traj.reset_index(drop=True), test_old_traj.reset_index(drop=True))
    >>> assert_frame_equal(expected_most_recent_traj.reset_index(drop=True), test_most_recent_traj.reset_index(drop=True))
    >>> assert_frame_equal(expected_old_obs.reset_index(drop=True), test_old_obs.reset_index(drop=True))


    >>> (old_traj, keep_traj), old_obs = time_window_management(
    ... ts.test_time_window,
    ... pd.DataFrame(columns=["nid"]),
    ... 8,
    ... 10,
    ... 5,
    ... 2,
    ... 5
    ... )

    >>> assert_frame_equal(old_traj.reset_index(drop=True), ts.old_traj_expected.reset_index(drop=True))
    >>> assert_frame_equal(keep_traj.reset_index(drop=True), ts.keep_traj_expected.reset_index(drop=True))
    >>> assert_frame_equal(old_obs, pd.DataFrame(columns=["nid"]))
    """

    most_recent_traj = pd.DataFrame(columns=trajectory_df.columns)
    oldest_traj = pd.DataFrame(columns=trajectory_df.columns)

    if len(trajectory_df) > 0:

        # get the lost observation of all trajectories
        last_obs_of_all_traj = get_n_last_observations_from_trajectories(
            trajectory_df, 1
        )

        # if the night difference exceed the time window
        if (nid_next_night - last_nid > traj_time_window) and keep_last:

            # get last observation of trajectories from the last nid
            last_obs_of_all_traj = last_obs_of_all_traj[
                last_obs_of_all_traj["nid"] == last_nid
            ]

            mask_traj = trajectory_df["trajectory_id"].isin(
                last_obs_of_all_traj["trajectory_id"]
            )
            most_recent_traj = trajectory_df[mask_traj]

            traj_size = (
                most_recent_traj.groupby(["trajectory_id"]).count().reset_index()
            )

            # keep only the trajectories with more than 2 points
            # if the difference of night is greater than the time window
            test = most_recent_traj["trajectory_id"].isin(
                traj_size[traj_size["nid"] > 2]["trajectory_id"]
            )

            most_recent_traj = most_recent_traj[test]

            oldest_traj = trajectory_df[~mask_traj]

            # return the trajectories from the last night id
            return (oldest_traj, most_recent_traj), pd.DataFrame(columns=["nid"])

        last_obs_of_all_traj["diff_nid"] = nid_next_night - last_obs_of_all_traj["nid"]

        most_recent_last_obs = last_obs_of_all_traj[
            last_obs_of_all_traj["diff_nid"] <= traj_time_window
        ]

        # mask used to split the trajectories between those inside the traj time window and the oldest one
        mask_traj = trajectory_df["trajectory_id"].isin(
            most_recent_last_obs["trajectory_id"]
        )

        most_recent_traj = trajectory_df[mask_traj]

        traj_size = most_recent_traj.groupby(["trajectory_id"]).count().reset_index()

        most_recent_traj = most_recent_traj.merge(
            most_recent_last_obs[["trajectory_id", "diff_nid"]], on="trajectory_id"
        )

        # Keep trajectories with more than 2 points
        test_1 = most_recent_traj["trajectory_id"].isin(
            traj_size[traj_size["nid"] > 2]["trajectory_id"]
        )

        # get the trajectories with only two points
        test_2_1 = most_recent_traj["trajectory_id"].isin(
            traj_size[traj_size["nid"] == 2]["trajectory_id"]
        )

        # keep the trajectories with two points and that not exceed the time windows for the trajectories with 2 points
        test_2_2 = most_recent_traj["diff_nid"] <= traj_2_points_time_windows
        test_2 = test_2_1 & test_2_2

        most_recent_traj = most_recent_traj[(test_1 | test_2)]

        most_recent_traj = most_recent_traj.drop(["diff_nid"], axis=1)

        oldest_traj = trajectory_df[~mask_traj]

    diff_nid_old_observation = nid_next_night - old_observation["nid"]
    old_observation = old_observation[diff_nid_old_observation <= obs_time_window]

    return (oldest_traj, most_recent_traj), old_observation


if __name__ == "__main__":  # pragma: no cover

    if "unittest.util" in __import__("sys").modules:
        # Show full diff in self.assertEqual.
        __import__("sys").modules["unittest.util"]._MAX_LENGTH = 999999999

    sys.exit(doctest.testmod()[0])
