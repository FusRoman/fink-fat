from decimal import DivisionByZero
import numpy as np
import pandas as pd
import multiprocessing as mp
import os
import astropy.units as u
from astropy.coordinates import SkyCoord
from alert_association.intra_night_association import intra_night_association
from alert_association.intra_night_association import new_trajectory_id_assignation
from alert_association.orbit_fitting.orbfit_management import compute_df_orbit_param
from alert_association.associations import (
    tracklets_and_trajectories_associations,
    trajectories_with_new_observations_associations,
    old_observations_with_tracklets_associations,
    old_with_new_observations_associations,
    time_window_management,
)


# constant to locate the ram file system
ram_dir = "/media/virtuelram/"


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


def compute_orbit_elem(trajectory_df, q):

    # print()
    # print("#####")
    # acceleration_filter(trajectory_df, 0.4)
    # print("#####")
    # print()

    _pid = os.getpid()
    current_ram_path = os.path.join(ram_dir, str(_pid), "")
    os.mkdir(current_ram_path)

    if len(trajectory_df) == 0:
        print("no orbit computation")
        q.put(trajectory_df)
        return 0

    traj_to_compute = trajectory_df[trajectory_df["a"] == -1.0]
    traj_with_orbelem = trajectory_df[trajectory_df["a"] != -1.0]

    if len(traj_to_compute) == 0:
        print("no orbit computation")
        q.put(trajectory_df)
        return 0

    print(
        "nb traj to compute orb elem: {}".format(
            len(np.unique(traj_to_compute["trajectory_id"]))
        )
    )

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

    orbit_elem = compute_df_orbit_param(
        traj_to_compute, int(mp.cpu_count() / 3), current_ram_path
    )

    traj_to_compute = traj_to_compute.merge(orbit_elem, on="trajectory_id")

    os.rmdir(current_ram_path)
    q.put(pd.concat([traj_with_orbelem, traj_to_compute]))

    return 0


def intra_night_step(
    new_observation,
    last_trajectory_id,
    intra_night_sep_criterion,
    intra_night_mag_criterion_same_fid,
    intra_night_mag_criterion_diff_fid,
    run_metrics,
):

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

    # remove all the alerts that appears in the tracklets
    new_observation_not_associated = new_observation[
        ~new_observation["candid"].isin(tracklets["candid"])
    ]

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
    run_metrics,
):
    # perform associations with the recorded trajectories :
    #   - trajectories with tracklets

    print("tracklets associations")

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
        run_metrics,
    )    

    # get the trajectories updated with new tracklets and the trajectory not updated for the next step
    traj_to_orb = traj_with_track[~traj_with_track["not_updated"]]
    traj_not_updated = traj_with_track[traj_with_track["not_updated"]]

    # concatenate the updated trajectories and the remaining tracklets
    all_traj_to_orb = pd.concat([traj_to_orb, not_associated_tracklets])

    # separate traklets with more than orbfit_limit points for the orbit computation and the other tracklets
    other_track, track_to_orb = prep_orbit_computation(all_traj_to_orb, orbfit_limit)

    tracklets_orbfit_process = mp.Process(
        target=compute_orbit_elem, args=(track_to_orb, return_trajectories_queue,)
    )
    tracklets_orbfit_process.start()

    return (
        traj_not_updated,
        other_track,
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
    run_metrics,
):
    print("trajectories associations")

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
        run_metrics,
    )

    # separate trajectories with more than 3 points for the orbit computation and the other trajectories
    other_traj, traj_to_orb = prep_orbit_computation(traj_with_new_obs, orbfit_limit)

    trajectories_orbfit_process = mp.Process(
        target=compute_orbit_elem, args=(traj_to_orb, return_trajectories_queue,)
    )
    trajectories_orbfit_process.start()

    return (
        other_traj,
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
    run_metrics,
):
    print("tracklets and old observations associations")

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
        run_metrics,
    )

    updated_tracklets = track_with_old_obs[~track_with_old_obs["not_updated"]]
    not_updated_tracklets = track_with_old_obs[track_with_old_obs["not_updated"]]

    # separate trajectories with more than 3 points for the orbit computation and the other trajectories
    track_not_orb, track_to_orb = prep_orbit_computation(updated_tracklets, orbfit_limit)

    tracklets_with_new_obs_orbfit_process = mp.Process(
        target=compute_orbit_elem, args=(track_to_orb, return_trajectories_queue,)
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
    intra_night_sep_criterion=145 * u.arcsecond,
    intra_night_mag_criterion_same_fid=2.21,
    intra_night_mag_criterion_diff_fid=1.75,
    sep_criterion=0.24 * u.degree,
    mag_criterion_same_fid=0.18,
    mag_criterion_diff_fid=0.7,
    angle_criterion=8.8,
    orbfit_limit = 3,
    run_metrics=False,
):
    """
    Perform night to night associations in four steps.

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
    nid_next_night : integer
        nid of the incoming night
    time_window : integer
        limit to keep old observation and trajectories
    sep_criterion : float
        the separation criterion to associates alerts
    mag_criterion_same_fid : float
        the magnitude criterion to associates alerts if the observations have been observed with the same filter
    mag_criterion_diff_fid : float
        the magnitude criterion to associates alerts if the observations have been observed with the same filter
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
    """

    if len(trajectory_df) > 0:
        last_trajectory_id = np.max(trajectory_df["trajectory_id"]) + 1
    else:
        last_trajectory_id = 0

    (old_traj, most_recent_traj), old_observation = time_window_management(
        trajectory_df, old_observation, last_nid, next_nid, traj_time_window, obs_time_window, orbfit_limit
    )

    inter_night_report = dict()
    inter_night_report["nid of the next night"] = int(next_nid)

    # intra night associations steps with the new observations
    (tracklets, remaining_new_observations, intra_night_report,) = intra_night_step(
        new_observation,
        last_trajectory_id,
        intra_night_sep_criterion,
        intra_night_mag_criterion_same_fid,
        intra_night_mag_criterion_diff_fid,
        run_metrics,
    )

    last_trajectory_id = np.max(tracklets["trajectory_id"]) + 1

    if len(most_recent_traj) == 0 and len(old_observation) == 0:

        other_track, track_to_orb = prep_orbit_computation(tracklets, orbfit_limit)

        q = mp.Queue()
        process = mp.Process(target=compute_orbit_elem, args=(track_to_orb, q,))
        process.start()
        track_with_orb_elem = q.get()

        process.terminate()

        inter_night_report["intra night report"] = intra_night_report
        inter_night_report["tracklets associations report"] = {}
        inter_night_report["trajectories associations report"] = {}
        inter_night_report["track and old obs associations report"] = {}
        inter_night_report["old observation and new observation report"] = {}

        return (
            pd.concat([other_track, track_with_orb_elem]),
            remaining_new_observations,
            inter_night_report,
        )

    return_trajectories_queue = mp.Queue()

    (
        traj_not_updated,
        other_track,
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
        run_metrics,
    )

    (
        not_associates_traj,
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
        run_metrics,
    )

    (
        not_updated_tracklets,
        remain_old_obs,
        track_and_old_obs_report,
        max_traj_id,
        track_with_old_obs_orbfit_process,
    ) = tracklets_and_old_observations_steps(
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
        run_metrics,
    )

    print("old observations and new observations associations")

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
        run_metrics,
    )

    tmp_traj_orb_elem = []
    for _ in range(3):
        tmp_traj_orb_elem.append(return_trajectories_queue.get())

    traj_with_orb_elem = pd.concat(tmp_traj_orb_elem)

    tracklets_orbfit_process.terminate()
    traj_with_new_obs_orbfit_process.terminate()
    track_with_old_obs_orbfit_process.terminate()

    print("----------------")
    acceleration_filter(not_associates_traj, 0.3)
    print("----------------")

    # concatenate all the trajectories with computed orbital elements with the other trajectories/tracklets.
    most_recent_traj = pd.concat(
        [traj_with_orb_elem, not_associates_traj, not_updated_tracklets, new_trajectory]
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

    return trajectory_df, old_observation, inter_night_report


def acceleration_filter(trajectory_df, acc_criteria):

    # import warnings
    # warnings.filterwarnings("error")

    if len(trajectory_df) > 0:
        def acceleration_df(x):
                ra, dec, jd = x["ra"], x["dec"], x["jd"]

                c1 = SkyCoord(ra, dec, unit = u.degree)

                diff_jd = np.diff(jd)

                sep = c1[0:-1].separation(c1[1:]).degree
                
                try:
                    velocity = sep / diff_jd

                    velocity = velocity[~np.isnan(velocity)]

                    return np.mean(np.abs(np.diff(velocity)))
                except RuntimeWarning:
                    print()
                    print(x["objectId"])
                    print()
                    print(x["candid"])
                    print()
                    print(x["ssnamenr"])
                    print()
                    print(x["nid"])
                    print()
                    print(jd)


        print(
            "*** nb trajectories: {}".format(len(np.unique(trajectory_df["trajectory_id"])))
        )
        t_before = t.time()
        tt = trajectory_df.groupby(['trajectory_id'])\
            .agg(
                ra=('ra',list), 
                dec=('dec',list), 
                dcmag=('dcmag',list), 
                candid=('candid', list),
                objectId=('objectId', list),
                ssnamenr=('ssnamenr', list),
                fid=('fid',list),
                nid=('nid',list),
                jd=('jd',list),
                trajectory_size=('candid',lambda x: len(list(x)))
                )

        tt = tt[tt["trajectory_size"] >= 3]

        if len(tt) > 0:
            tt["acc"] = tt.apply(acceleration_df, axis=1)

            print(t.time() - t_before)
            print()
            print("before acc filter: {}".format(len(tt)))
            print("after acc filter: {}".format(len(tt[tt["acc"] <= acc_criteria])))
            print()
        else:
            print("0 trajectories")
    
    else:
        print("0 trajectories")


if __name__ == "__main__":  # pragma: no cover
    import sys
    import doctest
    from pandas.testing import assert_frame_equal  # noqa: F401
    import test_sample as ts  # noqa: F401
    from unittest import TestCase  # noqa: F401

    import time as t

    if "unittest.util" in __import__("sys").modules:
        # Show full diff in self.assertEqual.
        __import__("sys").modules["unittest.util"]._MAX_LENGTH = 999999999

    def print_df_to_dict(df):
        print("{")
        for col in df.columns:
            print('"{}": {},'.format(col, list(df[col])))
        print("}")

    # sys.exit(doctest.testmod()[0])
    from alert_association.continuous_integration import load_data

    df_sso = load_data("Solar System MPC", 0)

    tr_orb_columns = [
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
        "not_updated",
        "trajectory_id",
    ]

    trajectory_df = pd.DataFrame(columns=tr_orb_columns)
    old_observation = pd.DataFrame(columns=["nid"])

    last_nid = np.min(df_sso["nid"])

    for tr_nid in np.unique(df_sso["nid"]):
        print("---New Night---")
        print(tr_nid)
        print()

        if tr_nid == 1540:
            break

        new_observation = df_sso[df_sso["nid"] == tr_nid]
        with pd.option_context("mode.chained_assignment", None):
            new_observation[tr_orb_columns] = -1.0
            new_observation["not_updated"] = np.ones(
                len(new_observation), dtype=np.bool_
            )

        next_nid = new_observation["nid"].values[0]

        print(
            "nb trajectories: {}".format(len(np.unique(trajectory_df["trajectory_id"])))
        )
        print("nb old obs: {}".format(len(old_observation)))
        print("nb new obs: {}".format(len(new_observation)))
        print()

        t_before = t.time()
        trajectory_df, old_observation, report = night_to_night_association(
            trajectory_df,
            old_observation,
            new_observation,
            last_nid,
            next_nid,
            traj_time_window=7,
            obs_time_window=3,
            sep_criterion=24 * u.arcminute,
            mag_criterion_same_fid=0.2,
            mag_criterion_diff_fid=0.5,
            orbfit_limit=5,
            angle_criterion=2,
        )

        elapsed_time = t.time() - t_before
        if elapsed_time <= 60:
            print()
            print("associations elapsed time: {} sec".format(round(elapsed_time, 3)))
        else:
            time_min = int(elapsed_time / 60)
            time_sec = round(elapsed_time % 60, 3)
            print()
            print(
                "associations elapsed time: {} min: {} sec".format(time_min, time_sec)
            )

        print()
        print()

        # print()
        # orb_elem = trajectory_df[trajectory_df["a"] != -1.0]
        # print(
        #     "number of trajectories with orbital elements: {}".format(
        #         len(np.unique(orb_elem["trajectory_id"]))
        #     )
        # )
        # print()

        print()
        print()
        print("/////////TEST JD DUPLICATES///////////")
        test = trajectory_df.groupby(['trajectory_id']).agg(jd=('jd',list)).reset_index()
        diff_jd = test.apply(lambda x: np.any(np.diff(x["jd"]) == 0), axis=1)
        keep_traj = test[diff_jd]["trajectory_id"]
        tttt = trajectory_df[trajectory_df["trajectory_id"].isin(keep_traj)]
        print(tttt[["ra", "dec", "objectId", "jd", "nid", "ssnamenr", "trajectory_id"]])
        print()
        print("/////////FIN TEST JD DUPLICATES///////////")

        print()
        print("---End Night---")
        print()
        print()
        print()
        print()

        last_nid = next_nid

    
    trajectory_df = trajectory_df.infer_objects()
    trajectory_df["ssnamenr"] = trajectory_df["ssnamenr"].astype(str)
    trajectory_df["fink_class"] = trajectory_df["fink_class"].astype(str)
    trajectory_df["objectId"] = trajectory_df["objectId"].astype(str)
    trajectory_df["provisional designation"] = trajectory_df[
        "provisional designation"
    ].astype(str)

    trajectory_df.to_parquet("trajectory_df.parquet")

    sys.exit(doctest.testmod()[0])
