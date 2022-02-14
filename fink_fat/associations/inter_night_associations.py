import numpy as np
import pandas as pd
import multiprocessing as mp
import os
import astropy.units as u
from fink_fat.associations.intra_night_association import intra_night_association
from fink_fat.associations.intra_night_association import new_trajectory_id_assignation
from fink_fat.orbit_fitting.orbfit_management import compute_df_orbit_param
from fink_fat.associations.associations import (
    tracklets_and_trajectories_associations,
    trajectories_with_new_observations_associations,
    old_observations_with_tracklets_associations,
    old_with_new_observations_associations,
    time_window_management,
)


def prep_orbit_computation(trajectory_df, orbfit_limit):
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

    >>> other_track, track_to_orb = prep_orbit_computation(trajectories, 3)

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


from collections import Counter
import time as t

def compute_orbit_elem(trajectory_df, q, ram_dir=""):
    """
    Compute the orbital elements of a set of trajectories.
    The computation are done in parallel process.

    Parameters
    ----------
    trajectory_df : dataframe
        A dataframe containing the observations of each trajectories
    q : Queue from Multiprocessing package
        A queue used to return the results of the process
    ram_dir : string
        ram_dir : string
        Path where files needed for the OrbFit computation are located

    Returns
    -------
    code : integer
        a return code indicating that the process has ended correctly

    Examples
    --------
    >>> q = mp.Queue()
    >>> compute_orbit_elem(pd.DataFrame(), q)
    0
    >>> d = q.get()
    >>> assert_frame_equal(pd.DataFrame(), d)

    >>> df_test = pd.DataFrame({
    ... "a" : [2.3]
    ... })

    >>> compute_orbit_elem(df_test, q)
    0
    >>> res_q = q.get()
    >>> assert_frame_equal(df_test, res_q)

    >>> orbit_column = [
    ... "provisional designation",
    ... "ref_epoch",
    ... "a",
    ... "e",
    ... "i",
    ... "long. node",
    ... "arg. peric",
    ... "mean anomaly",
    ... "rms_a",
    ... "rms_e",
    ... "rms_i",
    ... "rms_long. node",
    ... "rms_arg. peric",
    ... "rms_mean anomaly",
    ... ]

    >>> test_orbit = ts.orbfit_samples
    >>> test_orbit[orbit_column] = -1.0

    >>> compute_orbit_elem(ts.orbfit_samples, q)
    0

    >>> res_orb = q.get()

    >>> assert_frame_equal(res_orb, ts.compute_orbit_elem_output)
    """

    _pid = os.getpid()
    current_ram_path = os.path.join(ram_dir, str(_pid), "")

    if len(trajectory_df) == 0:
        q.put(trajectory_df)
        return 0

    traj_to_compute = trajectory_df[trajectory_df["a"] == -1.0]
    traj_with_orbelem = trajectory_df[trajectory_df["a"] != -1.0]

    if len(traj_to_compute) == 0:
        q.put(trajectory_df)
        return 0

    os.mkdir(current_ram_path)

    orbit_column = [
        "provisional designation",
        "ref_epoch",
        "a",
        "e",
        "i",
        "long. node",
        "arg. peric",
        "mean anomaly",
        "rms_a",
        "rms_e",
        "rms_i",
        "rms_long. node",
        "rms_arg. peric",
        "rms_mean anomaly",
    ]

    traj_to_compute = traj_to_compute.drop(orbit_column, axis=1)

    tr_gb = traj_to_compute.groupby(["trajectory_id"]).count()

    print("____")
    print("Orbfit traj size distribution")
    print()
    print(Counter(tr_gb["ra"].values))
    print()
    print("____")

    t_before = t.time()
    orbit_elem = compute_df_orbit_param(
        traj_to_compute, int(mp.cpu_count() / 2), current_ram_path
    )
    print("ORBFIT elapsed time: {}".format(t.time() - t_before))

    traj_to_return = traj_to_compute.merge(orbit_elem, on="trajectory_id")

    os.rmdir(current_ram_path)
    q.put(pd.concat([traj_with_orbelem, traj_to_return]))

    return 0


def intra_night_step(
    new_observation,
    last_trajectory_id,
    intra_night_sep_criterion,
    intra_night_mag_criterion_same_fid,
    intra_night_mag_criterion_diff_fid,
    run_metrics,
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
    run_metrics : boolean
        execute and return the performance metrics of the intra night association

    Returns
    -------
    tracklets : dataframe
        The tracklets detected inside the new night.
    new_observation_not_associated : dataframe
        All the observation that not occurs in a tracklets
    intra_night_report : dictionary
        Statistics about the intra night association, contains the following entries :

                    number of separation association

                    number of association filtered by magnitude

                    association metrics if compute_metrics is set to True

    Examples
    --------
    >>> track, remaining_new_obs, metrics = intra_night_step(
    ... ts.input_observation,
    ... 0,
    ... intra_night_sep_criterion = 145 * u.arcsecond,
    ... intra_night_mag_criterion_same_fid = 2.21,
    ... intra_night_mag_criterion_diff_fid = 1.75,
    ... run_metrics = True,
    ... )

    >>> assert_frame_equal(track.reset_index(drop=True), ts.tracklets_output, check_dtype = False)
    >>> assert_frame_equal(remaining_new_obs, ts.remaining_new_obs_output, check_dtype = False)
    >>> TestCase().assertDictEqual(metrics, ts.tracklets_expected_metrics)


    >>> track, remaining_new_obs, metrics = intra_night_step(
    ... ts.input_observation_2,
    ... 0,
    ... intra_night_sep_criterion = 145 * u.arcsecond,
    ... intra_night_mag_criterion_same_fid = 2.21,
    ... intra_night_mag_criterion_diff_fid = 1.75,
    ... run_metrics = True,
    ... )

    >>> assert_frame_equal(track.reset_index(drop=True), ts.tracklets_output_2, check_dtype = False)
    >>> assert_frame_equal(remaining_new_obs, ts.remaining_new_obs_output_2, check_dtype = False)
    >>> TestCase().assertDictEqual(metrics, {'number of separation association': 0, 'number of association filtered by magnitude': 0, 'association metrics': {}, 'number of intra night tracklets': 0})
    """

    # intra-night association of the new observations
    new_left, new_right, intra_night_report = intra_night_association(
        new_observation,
        sep_criterion=intra_night_sep_criterion,
        mag_criterion_same_fid=intra_night_mag_criterion_same_fid,
        mag_criterion_diff_fid=intra_night_mag_criterion_diff_fid,
        compute_metrics=run_metrics,
    )

    new_left, new_right = (
        new_left.reset_index(drop=True),
        new_right.reset_index(drop=True),
    )

    tracklets = new_trajectory_id_assignation(new_left, new_right, last_trajectory_id)

    intra_night_report["number of intra night tracklets"] = len(
        np.unique(tracklets["trajectory_id"])
    )

    if len(tracklets) > 0:
        # remove all the alerts that appears in the tracklets
        new_observation_not_associated = new_observation[
            ~new_observation["candid"].isin(tracklets["candid"])
        ]
    else:
        new_observation_not_associated = new_observation

    return (
        tracklets,
        new_observation_not_associated,
        intra_night_report,
    )


def tracklets_and_trajectories_steps(
    most_recent_traj,
    tracklets,
    next_nid,
    return_trajectories_queue,
    sep_criterion,
    mag_criterion_same_fid,
    mag_criterion_diff_fid,
    angle_criterion,
    orbfit_limit,
    max_traj_id,
    ram_dir,
    store_kd_tree,
    run_metrics,
):
    """
    Perform associations with the recorded trajectories and the tracklets detected inside the new night.
    The trajectories send to OrbFit can be recovered with the get method from Multiprocessing.Queue.

    Parameters
    ----------
    most_recent_traj : dataframe
        The trajectories to be combined with the tracklets
    tracklets : dataframe
        The new detected tracklets
    next_nid : integer
        the nid of the new observation night
    return_trajectories_queue : Queue from Multiprocessing package
        The trajectories send to OrbFit for orbital elements computation are return in this queue.
    sep_criterion : float
        separation criterion between the alerts to be associated, must be in arcsecond
    mag_criterion_same_fid : float
        magnitude criterion between the alerts with the same filter id
    mag_criterion_diff_fid : float
        magnitude criterion between the alerts with a different filter id
    angle_criterion : float
        the angle criterion to associates alerts during the cone search
    orbfit_limit : integer
        the minimum number of point for a trajectory to be send to OrbFit
    max_traj_id : integer
        The next trajectory identifier to assign to a trajectory.
    run_metrics : boolean
        run the inter night association metrics : trajectory_df, traj_next_night, old_observations and new_observations
        parameters should have the ssnamenr column.

    Returns
    -------
    traj_not_updated : dataframe
        The set of trajectories that have not been associated with the tracklets
    small_traj : dataframe
        The set of updated trajectories with not enough observations to be send to OrbFit.
    small_track : dataframe
        The set of not associated tracklets with not enough observations to be send to OrbFit.
    traj_and_track_assoc_report : dictionary
        statistics about the trajectory and tracklets association process, contains the following entries :

            list of updated trajectories

            all nid report with the following entries for each reports :

                        current nid

                        trajectories to tracklets report

                        metrics, no sense if run_metrics is set to False

    max_traj_id : integer
        The next trajectory identifier to a assign to a trajectory.
    tracklets_orbfit_process : processus
        The processus that run OrbFit over all the updated trajectories and tracklets in parallel of the main processus.

    Examples
    --------
    >>> q = mp.Queue()

    >>> traj_not_updated, small_traj, small_track, traj_and_track_assoc_report, max_traj_id, _ = tracklets_and_trajectories_steps(
    ... ts.traj_sample,
    ... ts.track_sample,
    ... 1538,
    ... q,
    ... sep_criterion=24 * u.arcminute,
    ... mag_criterion_same_fid=0.5,
    ... mag_criterion_diff_fid=1.5,
    ... angle_criterion=8.8,
    ... orbfit_limit=5,
    ... max_traj_id=3,
    ... ram_dir="",
    ... store_kd_tree=True,
    ... run_metrics=True,
    ... )

    >>> track_orb = q.get()

    >>> assert_frame_equal(traj_not_updated.reset_index(drop=True), ts.traj_not_updated_expected, check_dtype = False)
    >>> assert_frame_equal(small_track.reset_index(drop=True), ts.other_track_expected, check_dtype = False)
    >>> assert_frame_equal(track_orb.reset_index(drop=True), ts.track_orb_expected, check_dtype = False, atol=1e-1, rtol=1e-1)
    >>> assert_frame_equal(small_traj.reset_index(drop=True), ts.small_traj_expected, check_dtype = False)
    >>> TestCase().assertDictEqual(traj_and_track_assoc_report, ts.traj_track_metrics_expected)
    >>> max_traj_id
    3
    """

    (
        traj_with_track,
        not_associated_tracklets,
        max_traj_id,
        traj_and_track_assoc_report,
    ) = tracklets_and_trajectories_associations(
        most_recent_traj,
        tracklets,
        next_nid,
        sep_criterion,
        mag_criterion_same_fid,
        mag_criterion_diff_fid,
        angle_criterion,
        max_traj_id,
        store_kd_tree,
        run_metrics,
    )

    # get the trajectories updated with new tracklets and the trajectory not updated for the next step
    traj_updated = traj_with_track[~traj_with_track["not_updated"]]
    traj_not_updated = traj_with_track[traj_with_track["not_updated"]]

    # separate trajectories with more than orbfit_limit points for the orbit computation and the smallest trajectories
    small_traj, traj_to_orb = prep_orbit_computation(traj_updated, orbfit_limit)

    # separate traklets with more than orbfit_limit points for the orbit computation and the smallest tracklets
    small_track, track_to_orb = prep_orbit_computation(
        not_associated_tracklets, orbfit_limit
    )

    # concatenate the updated trajectories and the remaining tracklets
    all_traj_to_orb = pd.concat([traj_to_orb, track_to_orb])

    tracklets_orbfit_process = mp.Process(
        target=compute_orbit_elem,
        args=(all_traj_to_orb, return_trajectories_queue, ram_dir,),
    )

    tracklets_orbfit_process.start()

    return (
        traj_not_updated,
        small_traj,
        small_track,
        traj_and_track_assoc_report,
        max_traj_id,
        tracklets_orbfit_process,
    )


def trajectories_and_new_observations_steps(
    traj_not_updated,
    remaining_new_observations,
    next_nid,
    return_trajectories_queue,
    sep_criterion,
    mag_criterion_same_fid,
    mag_criterion_diff_fid,
    angle_criterion,
    orbfit_limit,
    max_traj_id,
    ram_dir,
    store_kd_tree,
    run_metrics,
):
    """
    Do the associations between recorded trajectories and the remaining associations from the new observations night that have not been associated before.

    Parameters
    ----------
    traj_not_updated : dataframe
        The trajectories that have not been associated with a tracklets in the last steps.
    remaining_new_observations : dataframe
        The observations from the new observation night that not occurs in a tracklets.
    next_nid : integer
        the nid of the new observation night
    return_trajectories_queue : Queue from Multiprocessing package
        The trajectories send to OrbFit for orbital elements computation are return in this queue.
    sep_criterion : float
        separation criterion between the alerts to be associated, must be in arcsecond
    mag_criterion_same_fid : float
        magnitude criterion between the alerts with the same filter id
    mag_criterion_diff_fid : float
        magnitude criterion between the alerts with a different filter id
    angle_criterion : float
        the angle criterion to associates alerts during the cone search
    orbfit_limit : integer
        the minimum number of point for a trajectory to be send to OrbFit
    max_traj_id : integer
        The next trajectory identifier to assign to a trajectory.
    run_metrics : boolean
        run the inter night association metrics

    Returns
    -------
    small_traj : dataframe
        The trajectories with not enough observations to be send to OrbFit
    remaining_new_observations : dataframe
        The remaining observations from the new night after associations.
    trajectories_associations_report : dictionary
        statistics about the trajectory and tracklets association process, contains the following entries :

            list of updated trajectories

            all nid report with the following entries for each reports :

                        current nid

                        trajectories to new observations report

                        metrics, no sense if run_metrics is set to False

    max_traj_id : integer
        The next trajectory identifier to a assign to a trajectory.
    trajectories_orbfit_process : processus
        The processus that run OrbFit over all the updated trajectories and tracklets in parallel of the main processus.


    Examples
    --------

    >>> q = mp.Queue()
    >>> other_traj, remaining_new_observations, trajectories_associations_report, max_traj_id, trajectories_orbfit_process = trajectories_and_new_observations_steps(
    ... ts.traj_sample_2,
    ... ts.new_obs_sample,
    ... 1524,
    ... q,
    ... sep_criterion=24 * u.arcminute,
    ... mag_criterion_same_fid=1.5,
    ... mag_criterion_diff_fid=0.7,
    ... angle_criterion=8.8,
    ... orbfit_limit=4,
    ... max_traj_id=9,
    ... ram_dir="",
    ... store_kd_tree=True,
    ... run_metrics=True,
    ... )

    >>> max_traj_id
    9

    >>> traj_orb = q.get()

    >>> assert_frame_equal(other_traj.reset_index(drop=True), ts.other_traj_expected, check_dtype=False)
    >>> assert_frame_equal(remaining_new_observations.reset_index(drop=True), ts.remaining_new_observations_expected, check_dtype=False)
    >>> assert_frame_equal(traj_orb.reset_index(drop=True), ts.traj_orb_expected, check_dtype=False)
    """

    # perform associations with the recorded trajectories
    (
        traj_with_new_obs,
        remaining_new_observations,
        max_traj_id,
        trajectories_associations_report,
    ) = trajectories_with_new_observations_associations(
        traj_not_updated,
        remaining_new_observations,
        next_nid,
        sep_criterion,
        mag_criterion_same_fid,
        mag_criterion_diff_fid,
        angle_criterion,
        max_traj_id,
        store_kd_tree,
        run_metrics,
    )

    # separate trajectories with more than orbfit_limit points for the orbit computation and the other trajectories
    small_traj, traj_to_orb = prep_orbit_computation(traj_with_new_obs, orbfit_limit)

    trajectories_orbfit_process = mp.Process(
        target=compute_orbit_elem,
        args=(traj_to_orb, return_trajectories_queue, ram_dir,),
    )
    trajectories_orbfit_process.start()

    return (
        small_traj,
        remaining_new_observations,
        trajectories_associations_report,
        max_traj_id,
        trajectories_orbfit_process,
    )


def tracklets_and_old_observations_steps(
    other_track,
    old_observation,
    next_nid,
    return_trajectories_queue,
    sep_criterion,
    mag_criterion_same_fid,
    mag_criterion_diff_fid,
    angle_criterion,
    orbfit_limit,
    max_traj_id,
    ram_dir,
    store_kd_tree,
    run_metrics,
):
    """
    Do the associations between the old observations keep in memory and the tracklets detected with the observations of the new observation night
    The tracklets that have a number of observations more or equal to orbfit_limit can be recover with the return_trajectories_queue.

    Parameters
    ----------
    other_track : dataframe
        all the remaining tracklets after the associations steps with the trajectories.
    old_observation : dataframe
        The old observations from previous observations night keep in memory
    next_nid : integer
        the nid of the new observation night
    return_trajectories_queue : Queue from Multiprocessing package
        The trajectories send to OrbFit for orbital elements computation are return in this queue.
    sep_criterion : float
        separation criterion between the alerts to be associated, must be in arcsecond
    mag_criterion_same_fid : float
        magnitude criterion between the alerts with the same filter id
    mag_criterion_diff_fid : float
        magnitude criterion between the alerts with a different filter id
    angle_criterion : float
        the angle criterion to associates alerts during the cone search
    orbfit_limit : integer
        the minimum number of point for a trajectory to be send to OrbFit
    max_traj_id : integer
        The next trajectory identifier to assign to a trajectory.
    run_metrics : boolean
        run the inter night association metrics

    Returns
    -------
    remaining_tracklets : dataframe
        The remaining tracklets after the associations process
    remain_old_obs : dataframe
        The remaining old observations after the associations process
    track_and_obs_report : dictionary
        statistics about the old observations and tracklets association process, contains the following entries :

            list of updated trajectories

            all nid report with the following entries for each reports :

                        current nid

                        trajectories to new observations report

                        metrics, no sense if run_metrics is set to False

    max_traj_id : integer
        The next trajectory identifier to a assign to a trajectory.
    tracklets_with_new_obs_orbfit_process : processus
        The processus that run OrbFit over all the updated trajectories and tracklets in parallel of the main processus.

    Examples
    --------
    >>> q = mp.Queue()

    >>> remaining_track, remaining_old_observations, track_associations_report, max_traj_id, track_orbfit_process = tracklets_and_old_observations_steps(
    ... ts.track_sample_2,
    ... ts.old_obs_sample,
    ... 1524,
    ... q,
    ... sep_criterion=24 * u.arcminute,
    ... mag_criterion_same_fid=1.5,
    ... mag_criterion_diff_fid=0.7,
    ... angle_criterion=8.8,
    ... orbfit_limit=3,
    ... max_traj_id=9,
    ... ram_dir="",
    ... store_kd_tree = True,
    ... run_metrics=True,
    ... )

    >>> max_traj_id
    9

    >>> traj_orb = q.get()

    >>> assert_frame_equal(remaining_track.reset_index(drop=True), ts.remaining_track_expected, check_dtype=False)
    >>> assert_frame_equal(remaining_old_observations.reset_index(drop=True), ts.remaining_old_observations_expected, check_dtype=False)
    >>> assert_frame_equal(traj_orb.reset_index(drop=True), ts.traj_orb_expected_2, check_dtype=False)
    """

    # perform associations with the tracklets and the old observations
    (
        track_with_old_obs,
        remain_old_obs,
        max_traj_id,
        track_and_obs_report,
    ) = old_observations_with_tracklets_associations(
        other_track,
        old_observation,
        next_nid,
        sep_criterion,
        mag_criterion_same_fid,
        mag_criterion_diff_fid,
        angle_criterion,
        max_traj_id,
        store_kd_tree,
        run_metrics,
    )

    updated_tracklets = track_with_old_obs[~track_with_old_obs["not_updated"]]
    not_updated_tracklets = track_with_old_obs[track_with_old_obs["not_updated"]]

    # separate trajectories with more than 3 points for the orbit computation and the other trajectories
    track_not_orb, track_to_orb = prep_orbit_computation(
        updated_tracklets, orbfit_limit
    )

    tracklets_with_new_obs_orbfit_process = mp.Process(
        target=compute_orbit_elem,
        args=(track_to_orb, return_trajectories_queue, ram_dir,),
    )
    tracklets_with_new_obs_orbfit_process.start()

    remaining_tracklets = pd.concat([not_updated_tracklets, track_not_orb])

    return (
        remaining_tracklets,
        remain_old_obs,
        track_and_obs_report,
        max_traj_id,
        tracklets_with_new_obs_orbfit_process,
    )


def night_to_night_association(
    trajectory_df,
    old_observation,
    new_observation,
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
    ram_dir="",
    run_metrics=False,
    do_track_and_traj_assoc=True,
    do_traj_and_new_obs_assoc=True,
    do_track_and_old_obs_assoc=True,
    do_new_obs_and_old_obs_assoc=True,
):
    """
    Perform night to night associations.

    Firstly, detect the intra night tracklets and then do the inter night associations in four steps :

    1. associates the recorded trajectories with the new tracklets detected in the new night.
    Associations based on the extremity alerts of the trajectories and the tracklets.

    2. associates the old observations with the extremity of the new tracklets.

    3. associates the new observations with the extremity of the recorded trajectories.

    4. associates the remaining old observations with the remaining new observations.

    Parameters
    ----------
    trajectory_df : dataframe
        all the recorded trajectory
    old_observation : dataframe
        all the observations from the previous night
    new_observation : dataframe
        all observations from the next night
    last_nid : integer
        nid of the previous observation night
    next_night : integer
        nid of the incoming night
    traj_time_window : integer
        limit to keep old trajectories in memory
    obs_time_window : integer
        limit to keep old observations in memory
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
    run_metrics : boolean
        launch and return the performance metrics of the intra night association and inter night association

    Returns
    -------
    trajectory_df : dataframe
        the updated trajectories with the new observations
    old_observation : dataframe
        the new set of old observations updated with the remaining non-associated new observations.
    inter_night_report : dictionary
        Statistics about the night_to_night_association, contains the following entries :

                    intra night report

                    trajectory association report

                    tracklets and observation report

    Examples
    --------
    >>> traj_expected, old_expected, report_expected = night_to_night_association(
    ... ts.trajectory_df_1,
    ... ts.old_observation_1,
    ... ts.new_observation_1,
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


    >>> traj_expected, old_expected, report_expected = night_to_night_association(
    ... ts.trajectory_df_2,
    ... ts.old_observation_2,
    ... ts.new_observation_2,
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


    >>> traj_expected, old_expected, report_expected = night_to_night_association(
    ... ts.trajectory_df_3,
    ... ts.old_observation_3,
    ... ts.new_observation_3,
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


    >>> traj_expected, old_expected, report_expected = night_to_night_association(
    ... ts.trajectory_df_4,
    ... ts.old_observation_4,
    ... ts.new_observation_4,
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


    >>> traj_expected, old_expected, report_expected = night_to_night_association(
    ... ts.trajectory_df_5,
    ... ts.old_observation_5,
    ... ts.new_observation_5,
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


    >>> traj_expected, old_expected, report_expected = night_to_night_association(
    ... ts.trajectory_df_5,
    ... ts.old_observation_5,
    ... ts.new_observation_6,
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


    >>> traj_expected, old_expected, report_expected = night_to_night_association(
    ... ts.trajectory_df_5,
    ... ts.old_observation_6,
    ... ts.new_observation_7,
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

    if len(trajectory_df) > 0:
        last_trajectory_id = np.max(trajectory_df["trajectory_id"]) + 1
    else:
        last_trajectory_id = 0

    (old_traj, most_recent_traj), old_observation = time_window_management(
        trajectory_df,
        old_observation,
        last_nid,
        next_nid,
        traj_time_window,
        obs_time_window,
        traj_2_points_time_window,
        orbfit_limit,
    )

    inter_night_report = dict()
    inter_night_report["nid of the next night"] = int(next_nid)

    t_before = t.time()
    # intra night associations steps with the new observations
    (tracklets, remaining_new_observations, intra_night_report,) = intra_night_step(
        new_observation,
        last_trajectory_id,
        intra_night_sep_criterion,
        intra_night_mag_criterion_same_fid,
        intra_night_mag_criterion_diff_fid,
        run_metrics,
    )
    print("elapsed time to find tracklets : {}".format(t.time() - t_before))

    if len(tracklets) > 0:
        last_trajectory_id = np.max(tracklets["trajectory_id"]) + 1

    if len(most_recent_traj) == 0 and len(old_observation) == 0:

        if len(tracklets) > 0:

            other_track, track_to_orb = prep_orbit_computation(tracklets, orbfit_limit)

            q = mp.Queue()
            process = mp.Process(
                target=compute_orbit_elem, args=(track_to_orb, q, ram_dir,)
            )
            process.start()
            track_with_orb_elem = q.get()

            process.terminate()

            inter_night_report["intra night report"] = intra_night_report
            inter_night_report["tracklets associations report"] = {}
            inter_night_report["trajectories associations report"] = {}
            inter_night_report["track and old obs associations report"] = {}
            inter_night_report["old observation and new observation report"] = {}

            return (
                pd.concat([other_track, track_with_orb_elem, old_traj]),
                remaining_new_observations,
                inter_night_report,
            )

        else:

            inter_night_report["intra night report"] = intra_night_report
            inter_night_report["tracklets associations report"] = {}
            inter_night_report["trajectories associations report"] = {}
            inter_night_report["track and old obs associations report"] = {}
            inter_night_report["old observation and new observation report"] = {}

            return (
                trajectory_df,
                new_observation,
                inter_night_report,
            )

    return_trajectories_queue = mp.Queue()
    orbfit_process = []

    # call tracklets_and_trajectories_steps if they have most_recent_traj and tracklets
    if len(most_recent_traj) > 0 and len(tracklets) > 0 and do_track_and_traj_assoc:

        t_before = t.time()
        (
            traj_not_updated,
            small_traj,
            small_track,
            traj_and_track_report,
            max_traj_id,
            tracklets_orbfit_process,
        ) = tracklets_and_trajectories_steps(
            most_recent_traj,
            tracklets,
            next_nid,
            return_trajectories_queue,
            sep_criterion,
            mag_criterion_same_fid,
            mag_criterion_diff_fid,
            angle_criterion,
            orbfit_limit,
            last_trajectory_id,
            ram_dir,
            store_kd_tree,
            run_metrics,
        )

        print("elapsed time to associates tracklets with trajectories : {}".format(t.time() - t_before))

        orbfit_process.append(tracklets_orbfit_process)

    else:
        # if they have only tracklets, we compute orbital elements for those with more than orbfit_limit observations
        if len(tracklets) > 0:
            small_track, track_to_orb = prep_orbit_computation(tracklets, orbfit_limit)

            tracklets_process = mp.Process(
                target=compute_orbit_elem,
                args=(track_to_orb, return_trajectories_queue,),
            )
            tracklets_process.start()
            orbfit_process.append(tracklets_process)

            traj_not_updated = most_recent_traj
            small_traj = pd.DataFrame(columns=most_recent_traj.columns)
            max_traj_id = last_trajectory_id
            traj_and_track_report = {}
        # else they have no most recent traj and no tracklets, we keep the most recent traj and the last_trajectory_id
        else:
            traj_not_updated = most_recent_traj
            small_traj = pd.DataFrame(columns=most_recent_traj.columns)
            small_track = pd.DataFrame(columns=tracklets.columns)
            max_traj_id = last_trajectory_id
            traj_and_track_report = {}

    # fmt: off
    assoc_test = len(traj_not_updated) > 0 and len(remaining_new_observations) > 0 and do_traj_and_new_obs_assoc
    # fmt: on
    if assoc_test:

        t_before = t.time()
        (
            not_associated_traj,
            remaining_new_observations,
            traj_and_new_obs_report,
            max_traj_id,
            traj_with_new_obs_orbfit_process,
        ) = trajectories_and_new_observations_steps(
            traj_not_updated,
            remaining_new_observations,
            next_nid,
            return_trajectories_queue,
            sep_criterion,
            mag_criterion_same_fid,
            mag_criterion_diff_fid,
            angle_criterion,
            orbfit_limit,
            max_traj_id,
            ram_dir,
            store_kd_tree,
            run_metrics,
        )

        print("elapsed time to associates new points to a trajectories : {}".format(len(t.time() - t_before)))

        orbfit_process.append(traj_with_new_obs_orbfit_process)
    else:
        not_associated_traj = traj_not_updated
        traj_and_new_obs_report = {}

    if len(small_track) > 0 and len(old_observation) > 0 and do_track_and_old_obs_assoc:

        t_before = t.time()
        (
            not_updated_tracklets,
            remain_old_obs,
            track_and_old_obs_report,
            max_traj_id,
            track_with_old_obs_orbfit_process,
        ) = tracklets_and_old_observations_steps(
            small_track,
            old_observation,
            next_nid,
            return_trajectories_queue,
            sep_criterion,
            mag_criterion_same_fid,
            mag_criterion_diff_fid,
            angle_criterion,
            orbfit_limit,
            max_traj_id,
            ram_dir,
            store_kd_tree,
            run_metrics,
        )

        print("elapsed time to associates the old points to the tracklets  : {}".format(len(t.time() - t_before)))

        orbfit_process.append(track_with_old_obs_orbfit_process)
    else:
        not_updated_tracklets = small_track
        remain_old_obs = old_observation
        track_and_old_obs_report = {}

    if do_new_obs_and_old_obs_assoc:

        t_before = t.time()
        (
            new_trajectory,
            remain_old_obs,
            remaining_new_observations,
            observation_report,
        ) = old_with_new_observations_associations(
            remain_old_obs,
            remaining_new_observations,
            next_nid,
            max_traj_id,
            sep_criterion,
            mag_criterion_same_fid,
            mag_criterion_diff_fid,
            store_kd_tree,
            run_metrics,
        )

        print("elapsed time to associates couples of observations : {}".format(len(t.time() - t_before)))
    else:
        new_trajectory = pd.DataFrame(columns=remain_old_obs.columns)
        observation_report = {}

    tmp_traj_orb_elem = []
    for process in orbfit_process:
        tmp_traj_orb_elem.append(return_trajectories_queue.get())

    if len(tmp_traj_orb_elem) > 0:
        traj_with_orb_elem = pd.concat(tmp_traj_orb_elem)
    else:
        traj_with_orb_elem = pd.DataFrame(columns=most_recent_traj.columns)

    # concatenate all the trajectories with computed orbital elements and the other trajectories/tracklets.
    most_recent_traj = pd.concat(
        [
            traj_with_orb_elem,
            not_associated_traj,
            not_updated_tracklets,
            new_trajectory,
            small_traj,
        ]
    )

    old_observation = pd.concat([remain_old_obs, remaining_new_observations])

    inter_night_report["intra night report"] = intra_night_report
    inter_night_report["tracklets associations report"] = traj_and_track_report
    inter_night_report["trajectories associations report"] = traj_and_new_obs_report
    inter_night_report[
        "track and old obs associations report"
    ] = track_and_old_obs_report
    inter_night_report[
        "old observation and new observation report"
    ] = observation_report

    trajectory_df = pd.concat([old_traj, most_recent_traj])
    trajectory_df["not_updated"] = np.ones(len(trajectory_df), dtype=np.bool_)

    # terminate properly all the orbfit process
    # for process in orbfit_process:
    #     process.terminate()

    return trajectory_df, old_observation, inter_night_report


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
