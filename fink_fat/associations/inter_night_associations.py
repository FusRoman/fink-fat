import numpy as np
import pandas as pd
import astropy.units as u
import time as t
from fink_fat.associations.intra_night_association import intra_night_association
from fink_fat.associations.intra_night_association import new_trajectory_id_assignation
from fink_fat.associations.associations import (
    tracklets_and_trajectories_associations,
    trajectories_with_new_observations_associations,
    old_observations_with_tracklets_associations,
    old_with_new_observations_associations,
    time_window_management,
)


def separate_trajectories(trajectory_df, orbfit_limit):
    """
    Return the trajectories with less than orbfit_limit points and the others with orbfit_limit points and more.
    The trajectories with 3 points and more will be used for the computation of orbital elements.

    Parameters
    ----------
    trajectory_df : dataframe
        dataframe containing trajectories observations
        the following columns are required : trajectory_id, ra
    orbfit_limit : integer
        trajectories with a number of points greater or equal to orbfit_limit can go to the orbit fitting step.

    Return
    ------
    other_track : dataframe
        trajectories with less than 3 points
    track_to_orb : dataframe
        trajectories with 3 points and more

    Examples
    --------
    >>> trajectories = pd.DataFrame({"trajectory_id": [1, 1, 1, 2, 2], "ra": [0, 0, 0, 0, 0]})

    >>> other_track, track_to_orb = separate_trajectories(trajectories, 3)

    >>> other_track
       trajectory_id  ra
    3              2   0
    4              2   0
    >>> track_to_orb
       trajectory_id  ra
    0              1   0
    1              1   0
    2              1   0
    """

    with pd.option_context("mode.chained_assignment", None):
        trajectory_df["trajectory_id"] = trajectory_df["trajectory_id"].astype(int)

    traj_length = (
        trajectory_df.groupby(["trajectory_id"]).agg({"ra": len}).reset_index()
    )

    mask = traj_length["ra"] >= orbfit_limit

    traj_id_sup = traj_length[mask]["trajectory_id"]
    traj_id_inf = traj_length[~mask]["trajectory_id"]

    track_to_orb = trajectory_df[trajectory_df["trajectory_id"].isin(traj_id_sup)]
    other_track = trajectory_df[trajectory_df["trajectory_id"].isin(traj_id_inf)]

    return other_track.copy(), track_to_orb.copy()


def intra_night_step(
    new_observation,
    last_trajectory_id,
    intra_night_sep_criterion,
    intra_night_mag_criterion_same_fid,
    intra_night_mag_criterion_diff_fid,
):
    """
    Perform the intra nigth associations step at the beginning of the inter night association function.

    Parameters
    ----------
    new_observation : dataframe
        The new observation from the new observations night.
        The dataframe must have the following columns : ra, dec, jd, fid, nid, dcmag, candid, ssnamenr
    last_trajectory_id : integer
        The last trajectory identifier assign to a trajectory
    intra_night_sep_criterion : float
        separation criterion between the alerts to be associated, must be in arcsecond
    intra_night_mag_criterion_same_fid : float
        magnitude criterion between the alerts with the same filter id
    intra_night_mag_criterion_diff_fid : float
        magnitude criterion between the alerts with a different filter id

    Returns
    -------
    tracklets : dataframe
        The tracklets detected inside the new night.
    new_observation_not_associated : dataframe
        All the observation that not occurs in a tracklets

    Examples
    --------
    >>> track, remaining_new_obs = intra_night_step(
    ... ts.input_observation,
    ... 0,
    ... intra_night_sep_criterion = 145 * u.arcsecond,
    ... intra_night_mag_criterion_same_fid = 2.21,
    ... intra_night_mag_criterion_diff_fid = 1.75
    ... )

    >>> assert_frame_equal(track.reset_index(drop=True), ts.tracklets_output, check_dtype = False)
    >>> assert_frame_equal(remaining_new_obs, ts.remaining_new_obs_output, check_dtype = False)

    >>> track, remaining_new_obs = intra_night_step(
    ... ts.input_observation_2,
    ... 0,
    ... intra_night_sep_criterion = 145 * u.arcsecond,
    ... intra_night_mag_criterion_same_fid = 2.21,
    ... intra_night_mag_criterion_diff_fid = 1.75,
    ... )

    >>> assert_frame_equal(track.reset_index(drop=True), ts.tracklets_output_2, check_dtype = False)
    >>> assert_frame_equal(remaining_new_obs, ts.remaining_new_obs_output_2, check_dtype = False)
    """

    # intra-night association of the new observations
    new_left, new_right = intra_night_association(
        new_observation,
        sep_criterion=intra_night_sep_criterion,
        mag_criterion_same_fid=intra_night_mag_criterion_same_fid,
        mag_criterion_diff_fid=intra_night_mag_criterion_diff_fid,
    )

    new_left, new_right = (
        new_left.reset_index(drop=True),
        new_right.reset_index(drop=True),
    )

    tracklets = new_trajectory_id_assignation(new_left, new_right, last_trajectory_id)

    if len(tracklets) > 0:
        # remove all the alerts that appears in the tracklets
        new_observation_not_associated = new_observation[
            ~new_observation["candid"].isin(tracklets["candid"])
        ]
    else:
        new_observation_not_associated = new_observation

    return (tracklets, new_observation_not_associated)


def night_to_night_association(
    trajectory_df,
    old_observation,
    new_observation,
    last_trajectory_id,
    last_nid,
    next_nid,
    traj_time_window,
    obs_time_window,
    traj_2_points_time_window,
    intra_night_sep_criterion=145 * u.arcsecond,
    intra_night_mag_criterion_same_fid=2.21,
    intra_night_mag_criterion_diff_fid=1.75,
    sep_criterion=0.24 * u.degree,
    mag_criterion_same_fid=0.18,
    mag_criterion_diff_fid=0.7,
    angle_criterion=8.8,
    store_kd_tree=False,
    orbfit_limit=3,
    do_track_and_traj_assoc=True,
    do_traj_and_new_obs_assoc=True,
    do_track_and_old_obs_assoc=True,
    do_new_obs_and_old_obs_assoc=True,
    verbose=False,
):
    """
    Perform night to night associations.

    Firstly, detect the intra night tracklets and then do the inter night associations in four steps :

    1. associates the recorded trajectories with the new tracklets detected in the new night.
    Associations based on the extremity alerts of the trajectories and the tracklets.

    2. associates the new observations with the extremity of the recorded trajectories.

    3. associates the old observations with the extremity of the new tracklets.

    4. associates the remaining old observations with the remaining new observations.

    Parameters
    ----------
    trajectory_df : dataframe
        all the recorded trajectory
    old_observation : dataframe
        all the observations from the previous night
    new_observation : dataframe
        all observations from the next night
    last_trajectory_id : integer
        the maximum trajectory_id from the trajectories candidates database and the orbital candidates database
    last_nid : integer
        nid of the previous observation night
    next_night : integer
        nid of the incoming night
    traj_time_window : integer
        limit to keep old trajectories in memory
    obs_time_window : integer
        limit to keep old observations in memory
    traj_2_points_time_windows : integer
        limit to keep the trajectories of two points.
        These are observations detected during the observations association step and are not accurate.
        To limit the combinatorial, keep them less time than the other trajectories with more points.
    intra_night_sep_criterion : float
        separation criterion between the intra night alerts to be associated, must be in arcsecond
    intra_night_mag_criterion_same_fid : float
        magnitude criterion between the intra night alerts with the same filter id
    intra_night_mag_criterion_diff_fid : float
        magnitude criterion between the intra night alerts with a different filter id
    sep_criterion : float
        the separation criterion to associates inter night alerts
    mag_criterion_same_fid : float
        the magnitude criterion to associates inter night alerts if the observations have been observed with the same filter
    mag_criterion_diff_fid : float
        the magnitude criterion to associates inter night alerts if the observations have been observed with the same filter
    angle_criterion : float
        the angle criterion to associates alerts during the cone search
    store_kd_tree : boolean
        if set to true, store the kd tree return by the search_around_sky method of astropy.
    orbfit_limit : integer
        The number of points required to send trajectories to the orbit fitting program.
        Remove the trajectories with more point than "orbfit limit" points and without orbital elements.
        Keep the trajectories with less than "orbfit limit" points
    ram_dir : string
        Path where files needed for the OrbFit computation are located
    do_track_and_traj_assoc : boolean
        if set to false, deactivate the association between the trajectories and the tracklets
    do_traj_and_new_obs_assoc : boolean
        if set to false, deactivate the association between the trajectories and the new observations
    do_track_and_old_obs_assoc : boolean
        if set to false, deactivate the association between the old observations and the tracklets
    do_new_obs_and_old_obs_assoc : boolean
        if set to false, deactivate the association between the old observations and the new observations
    verbose : boolean
        if set to true, print information of the assocation process

    Returns
    -------
    trajectory_df : dataframe
        the updated trajectories with the new observations
    old_observation : dataframe
        the new set of old observations updated with the remaining non-associated new observations.

    Examples
    --------
    >>> traj_expected, old_expected = night_to_night_association(
    ... ts.trajectory_df_1,
    ... ts.old_observation_1,
    ... ts.new_observation_1,
    ... 0,
    ... 1619,
    ... 1623,
    ... traj_time_window=30,
    ... obs_time_window=30,
    ... traj_2_points_time_window=30,
    ... sep_criterion=24 * u.arcminute,
    ... mag_criterion_same_fid=0.3,
    ... mag_criterion_diff_fid=0.7,
    ... orbfit_limit=5,
    ... angle_criterion=1.5,
    ... )

    >>> assert_frame_equal(traj_expected, ts.trajectory_df_expected_1, check_dtype=False)
    >>> assert_frame_equal(old_expected, ts.old_observation_expected_1, check_dtype=False)


    >>> traj_expected, old_expected = night_to_night_association(
    ... ts.trajectory_df_2,
    ... ts.old_observation_2,
    ... ts.new_observation_2,
    ... 0,
    ... 1522,
    ... 1526,
    ... traj_time_window=30,
    ... obs_time_window=30,
    ... traj_2_points_time_window=30,
    ... intra_night_sep_criterion=500 * u.arcsecond,
    ... sep_criterion=2 * u.degree,
    ... mag_criterion_same_fid=0.3,
    ... mag_criterion_diff_fid=0.7,
    ... orbfit_limit=5,
    ... angle_criterion=5,
    ... )

    >>> assert_frame_equal(traj_expected, ts.trajectory_df_expected_2, check_dtype=False)
    >>> assert_frame_equal(old_expected, ts.old_observation_expected_2, check_dtype=False)


    >>> traj_expected, old_expected = night_to_night_association(
    ... ts.trajectory_df_3,
    ... ts.old_observation_3,
    ... ts.new_observation_3,
    ... 0,
    ... 1539,
    ... 1550,
    ... traj_time_window=30,
    ... obs_time_window=30,
    ... traj_2_points_time_window=30,
    ... intra_night_sep_criterion=500 * u.arcsecond,
    ... sep_criterion=2 * u.degree,
    ... mag_criterion_same_fid=0.3,
    ... mag_criterion_diff_fid=0.7,
    ... orbfit_limit=5,
    ... angle_criterion=5,
    ... )

    >>> assert_frame_equal(traj_expected.reset_index(drop=True), ts.trajectory_df_expected_3, check_dtype=False)
    >>> assert_frame_equal(old_expected, ts.old_observation_expected_3, check_dtype=False)


    >>> traj_expected, old_expected = night_to_night_association(
    ... ts.trajectory_df_4,
    ... ts.old_observation_4,
    ... ts.new_observation_4,
    ... 0,
    ... 1520,
    ... 1521,
    ... traj_time_window=30,
    ... obs_time_window=30,
    ... traj_2_points_time_window=30,
    ... intra_night_sep_criterion=500 * u.arcsecond,
    ... sep_criterion=2 * u.degree,
    ... mag_criterion_same_fid=0.3,
    ... mag_criterion_diff_fid=5,
    ... orbfit_limit=5,
    ... angle_criterion=5,
    ... )

    >>> assert_frame_equal(traj_expected.reset_index(drop=True), ts.trajectory_df_expected_4, check_dtype=False)
    >>> assert_frame_equal(old_expected, ts.old_observation_expected_4, check_dtype=False)


    >>> traj_expected, old_expected = night_to_night_association(
    ... ts.trajectory_df_5,
    ... ts.old_observation_5,
    ... ts.new_observation_5,
    ... 0,
    ... 1520,
    ... 1521,
    ... traj_time_window=30,
    ... obs_time_window=30,
    ... traj_2_points_time_window=30,
    ... intra_night_sep_criterion=500 * u.arcsecond,
    ... sep_criterion=2 * u.degree,
    ... mag_criterion_same_fid=0.3,
    ... mag_criterion_diff_fid=5,
    ... orbfit_limit=5,
    ... angle_criterion=5,
    ... )

    >>> assert_frame_equal(traj_expected.reset_index(drop=True), ts.trajectory_df_expected_5, check_dtype=False)
    >>> assert_frame_equal(old_expected, ts.old_observation_expected_5, check_dtype=False)


    >>> traj_expected, old_expected = night_to_night_association(
    ... ts.trajectory_df_5,
    ... ts.old_observation_5,
    ... ts.new_observation_6,
    ... 0,
    ... 1520,
    ... 1521,
    ... traj_time_window=30,
    ... obs_time_window=30,
    ... traj_2_points_time_window=30,
    ... intra_night_sep_criterion=500 * u.arcsecond,
    ... sep_criterion=2 * u.degree,
    ... mag_criterion_same_fid=0.3,
    ... mag_criterion_diff_fid=5,
    ... orbfit_limit=5,
    ... angle_criterion=5,
    ... )

    >>> assert_frame_equal(traj_expected.reset_index(drop=True), ts.trajectory_df_5, check_dtype=False)
    >>> assert_frame_equal(old_expected, ts.old_observation_expected_6, check_dtype=False)


    >>> traj_expected, old_expected = night_to_night_association(
    ... ts.trajectory_df_5,
    ... ts.old_observation_6,
    ... ts.new_observation_7,
    ... 0,
    ... 1520,
    ... 1521,
    ... traj_time_window=30,
    ... obs_time_window=30,
    ... traj_2_points_time_window=30,
    ... intra_night_sep_criterion=500 * u.arcsecond,
    ... sep_criterion=2 * u.degree,
    ... mag_criterion_same_fid=0.3,
    ... mag_criterion_diff_fid=5,
    ... orbfit_limit=5,
    ... angle_criterion=5,
    ... )

    >>> assert_frame_equal(traj_expected.reset_index(drop=True), ts.trajectory_df_expected_6, check_dtype=False)
    >>> assert_frame_equal(old_expected, ts.old_observation_expected_5, check_dtype=False)
    """

    (old_traj, most_recent_traj), old_observation = time_window_management(
        trajectory_df,
        old_observation,
        last_nid,
        next_nid,
        traj_time_window,
        obs_time_window,
        traj_2_points_time_window,
    )

    if verbose:  # pragma: no cover
        t_before = t.time()

    # intra night associations steps with the new observations
    (tracklets, remaining_new_observations,) = intra_night_step(
        new_observation,
        last_trajectory_id,
        intra_night_sep_criterion,
        intra_night_mag_criterion_same_fid,
        intra_night_mag_criterion_diff_fid,
    )

    tracklets["assoc_tag"] = "I"

    if verbose:  # pragma: no cover
        print("elapsed time to find tracklets : {}".format(t.time() - t_before))

    if len(most_recent_traj) == 0 and len(old_observation) == 0:

        return (pd.concat([old_traj, tracklets]), remaining_new_observations)

    if len(tracklets) > 0:
        # separate the tracklets with enough points to be sent to orbfit and the other small one.
        # don't take the risk to add more points to a tracklets as the associations are not accurates.
        small_track, track_orb = separate_trajectories(tracklets, orbfit_limit)
        last_trajectory_id = np.max(tracklets["trajectory_id"]) + 1
    else:
        small_track = pd.DataFrame(columns=new_observation.columns)
        track_orb = pd.DataFrame(columns=new_observation.columns)

    # call tracklets_and_trajectories_steps if they have most_recent_traj and tracklets
    if len(most_recent_traj) > 0 and len(tracklets) > 0 and do_track_and_traj_assoc:

        if verbose:  # pragma: no cover
            t_before = t.time()

        (
            traj_with_track,
            not_associated_tracklets,
            max_traj_id,
        ) = tracklets_and_trajectories_associations(
            most_recent_traj,
            small_track,
            next_nid,
            sep_criterion,
            mag_criterion_same_fid,
            mag_criterion_diff_fid,
            angle_criterion,
            last_trajectory_id,
            store_kd_tree,
        )

        if verbose:  # pragma: no cover
            print(
                "elapsed time to associates tracklets with trajectories : {}".format(
                    t.time() - t_before
                )
            )

    else:
        traj_with_track = most_recent_traj
        not_associated_tracklets = small_track
        max_traj_id = last_trajectory_id

    # fmt: off
    assoc_test = len(traj_with_track) > 0 and len(remaining_new_observations) > 0 and do_traj_and_new_obs_assoc
    # fmt: on
    if assoc_test:

        if verbose:  # pragma: no cover
            t_before = t.time()

        # perform associations with the recorded trajectories
        (
            traj_with_new_obs,
            remaining_new_observations,
            max_traj_id,
        ) = trajectories_with_new_observations_associations(
            traj_with_track,
            remaining_new_observations,
            next_nid,
            sep_criterion,
            mag_criterion_same_fid,
            mag_criterion_diff_fid,
            angle_criterion,
            max_traj_id,
            store_kd_tree,
        )

        if verbose:  # pragma: no cover
            print(
                "elapsed time to associates new points to a trajectories : {}".format(
                    t.time() - t_before
                )
            )

    else:
        traj_with_new_obs = traj_with_track

    # fmt: off
    test = len(not_associated_tracklets) > 0 and len(old_observation) > 0 and do_track_and_old_obs_assoc
    # fmt: on
    if test:

        if verbose:  # pragma: no cover
            t_before = t.time()

        # perform associations with the tracklets and the old observations
        (
            track_with_old_obs,
            remain_old_obs,
            max_traj_id,
        ) = old_observations_with_tracklets_associations(
            not_associated_tracklets,
            old_observation,
            next_nid,
            sep_criterion,
            mag_criterion_same_fid,
            mag_criterion_diff_fid,
            angle_criterion,
            max_traj_id,
            store_kd_tree,
        )

        if verbose:  # pragma: no cover
            print(
                "elapsed time to associates the old points to the tracklets  : {}".format(
                    t.time() - t_before
                )
            )

    else:
        track_with_old_obs = not_associated_tracklets
        remain_old_obs = old_observation

    # fmt: off
    test = len(remain_old_obs) > 0 and len(remaining_new_observations) > 0 and do_new_obs_and_old_obs_assoc
    # fmt: on
    if test:

        if verbose:  # pragma: no cover
            t_before = t.time()
        (
            new_trajectory,
            remain_old_obs,
            remaining_new_observations,
        ) = old_with_new_observations_associations(
            remain_old_obs,
            remaining_new_observations,
            next_nid,
            max_traj_id,
            sep_criterion,
            mag_criterion_same_fid,
            mag_criterion_diff_fid,
            store_kd_tree,
        )

        if verbose:  # pragma: no cover
            print(
                "elapsed time to associates couples of observations : {}".format(
                    t.time() - t_before
                )
            )
    else:
        new_trajectory = pd.DataFrame(columns=remain_old_obs.columns)

    # concatenate all the trajectories with computed orbital elements and the other trajectories/tracklets.
    most_recent_traj = pd.concat(
        [track_orb, traj_with_new_obs, track_with_old_obs, new_trajectory]
    )

    old_observation = pd.concat([remain_old_obs, remaining_new_observations])

    trajectory_df = pd.concat([old_traj, most_recent_traj])
    trajectory_df["not_updated"] = np.ones(len(trajectory_df), dtype=np.bool_)

    return trajectory_df, old_observation


if __name__ == "__main__":  # pragma: no cover
    import sys
    import doctest
    from pandas.testing import assert_frame_equal  # noqa: F401
    import fink_fat.test.test_sample as ts  # noqa: F401
    from unittest import TestCase  # noqa: F401

    if "unittest.util" in __import__("sys").modules:
        # Show full diff in self.assertEqual.
        __import__("sys").modules["unittest.util"]._MAX_LENGTH = 999999999

    sys.exit(doctest.testmod()[0])
