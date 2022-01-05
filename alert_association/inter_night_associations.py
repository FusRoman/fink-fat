import numpy as np
import pandas as pd
from alert_association.night_to_night_association import (
    night_to_night_trajectory_associations,
)
from alert_association.intra_night_association import (
    get_n_last_observations_from_trajectories,
)
from alert_association.intra_night_association import compute_inter_night_metric
import astropy.units as u
from alert_association.intra_night_association import intra_night_association
from alert_association.intra_night_association import new_trajectory_id_assignation
from alert_association.orbit_fitting.orbfit_management import compute_df_orbit_param
from alert_association.night_to_night_association import time_window_management

# constant to locate the ram file system
ram_dir = "/media/virtuelram/"


# TODO : tester le retour des rapports pour tracklets_associations
def tracklets_associations(
    trajectories,
    tracklets,
    next_nid,
    sep_criterion,
    mag_criterion_same_fid,
    mag_criterion_diff_fid,
    angle_criterion,
    run_metrics=False,
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
    run_metrics : boolean
        run the inter night association metrics : trajectory_df, traj_next_night, old_observations and new_observations
        parameters should have the ssnamenr column.

    Return
    ------
    trajectories : dataframe
        trajectories associated with new tracklets
    tracklets : dataframe
        remaining tracklets
    traj_to_track_report : dict list
        statistics about the trajectory and tracklets association process, contains the following entries :

            list of updated trajectories

            all nid report with the following entries for each reports :

                        current nid

                        trajectories to tracklets report

                        metrics, no sense if run_metrics is set to False

    Examples
    --------
    >>> trajectories = ts.trajectory_df_sample
    >>> tracklets = ts.traj_next_night_sample

    >>> tr_orb_columns = [
    ... "provisional designation",
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
    ... "rms_mean anomaly"
    ... ]

    >>> trajectories[tr_orb_columns] = -1.0
    >>> tracklets[tr_orb_columns] = -1.0
    >>> trajectories["not_updated"] = np.ones(len(trajectories), dtype=np.bool_)

    >>> tr, tk, report = tracklets_associations(trajectories, tracklets, 4, 2 * u.degree, 0.2, 0.5, 30)

    >>> assert_frame_equal(tr, ts.trajectories_expected_1, check_dtype=False)
    >>> assert_frame_equal(tk, ts.tracklets_expected_1, check_dtype=False)

    >>> trajectories = ts.trajectories_sample_1
    >>> tracklets = ts.tracklets_sample_1

    >>> tr_orb_columns = [
    ... "provisional designation",
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
    ... "rms_mean anomaly"
    ... ]

    >>> trajectories[tr_orb_columns] = -1.0
    >>> tracklets[tr_orb_columns] = -1.0
    >>> trajectories["not_updated"] = np.ones(len(trajectories), dtype=np.bool_)

    >>> tr, tk, report = tracklets_associations(trajectories, tracklets, 3, 1.5 * u.degree, 0.2, 0.5, 30, True)

    >>> assert_frame_equal(tr, ts.trajectories_expected_2, check_dtype=False)
    >>> assert_frame_equal(tk, ts.tracklets_expected_2, check_dtype=False)
    >>> TestCase().assertDictEqual(ts.traj_and_track_assoc_report_expected, report)

    >>> tracklets = ts.tracklets_sample_2
    >>> trajectories = ts.trajectories_sample_2

    >>> tr_orb_columns = [
    ... "provisional designation",
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
    ... "rms_mean anomaly"
    ... ]

    >>> trajectories[tr_orb_columns] = -1.0
    >>> tracklets[tr_orb_columns] = -1.0
    >>> trajectories["not_updated"] = np.ones(len(trajectories), dtype=np.bool_)

    >>> tr, tk, report = tracklets_associations(trajectories, tracklets, 3, 1 * u.degree, 0.2, 0.5, 30)

    >>> assert_frame_equal(tr, ts.trajectories_expected_3, check_dtype=False)
    >>> assert_frame_equal(tk, pd.DataFrame(columns=["ra", "dec", "dcmag", "fid", "nid", "jd", "candid", "trajectory_id", "provisional designation", "a", "e", "i", "long. node", "arg. peric", "mean anomaly", "rms_a", "rms_e", "rms_i", "rms_long. node", "rms_arg. peric", "rms_mean anomaly",]), check_index_type=False, check_dtype=False)

    >>> tr, tk, report = tracklets_associations(pd.DataFrame(), tracklets, 3, 1 * u.degree, 0.2, 0.5, 30)

    >>> assert_frame_equal(tr, pd.DataFrame(), check_dtype=False)
    >>> assert_frame_equal(tk, tracklets)
    >>> TestCase().assertDictEqual({}, report)
    """

    if len(trajectories) == 0 or len(tracklets) == 0:
        return trajectories, tracklets, {}
    else:

        trajectories_not_updated = trajectories[
            trajectories["not_updated"] & (trajectories["a"] == -1.0)
        ]
        trajectories_and_tracklets_associations_report = dict()

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

        traj_to_track_report = []
        updated_trajectories = []

        # for each trajectory nid from the last observations
        for tr_nid in trajectories_nid:

            current_nid_assoc_report = dict()
            current_nid_assoc_report["old nid"] = int(tr_nid)

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
                night_to_night_traj_to_tracklets_report,
            ) = night_to_night_trajectory_associations(
                two_last_current_nid,
                tracklets_extremity,
                norm_sep_crit,
                norm_same_fid,
                norm_diff_fid,
                angle_criterion,
            )

            if len(traj_extremity_associated) > 0:

                # duplicates management is not really optimized, optimisation would be to remove the for loop
                # and vectorized the operation

                # creates a dataframe for each duplicated trajectory associated with the tracklets
                duplicates = traj_extremity_associated["trajectory_id"].duplicated()
                all_duplicate_traj = []

                # get the duplicated tracklets
                tracklets_duplicated = traj_extremity_associated[duplicates]

                if len(tracklets_duplicated) > 0:

                    # get the max trajectory id
                    max_traj_id = np.max(np.unique(trajectories["trajectory_id"]))

                    for _, rows in tracklets_duplicated.iterrows():
                        max_traj_id += 1

                        # silence the copy warning
                        with pd.option_context("mode.chained_assignment", None):

                            # get the trajectory associated with the tracklets and update the trajectory id
                            duplicate_traj = trajectories[
                                trajectories["trajectory_id"] == rows["trajectory_id"]
                            ]
                            duplicate_traj["trajectory_id"] = max_traj_id
                            duplicate_traj["not_updated"] = False

                            # get the tracklets and update the trajectory id
                            other_track = tracklets[
                                tracklets["trajectory_id"] == rows["tmp_traj"]
                            ]
                            other_track["trajectory_id"] = max_traj_id
                            other_track["not_updated"] = False

                        # append the trajectory and the tracklets into the list
                        all_duplicate_traj.append(duplicate_traj)
                        all_duplicate_traj.append(other_track)
                        updated_trajectories.append(max_traj_id)

                    # creates a dataframe with all duplicates and adds it to the trajectories dataframe
                    all_duplicate_traj = pd.concat(all_duplicate_traj)
                    trajectories = pd.concat([trajectories, all_duplicate_traj])

                    tracklets = tracklets[
                        ~tracklets["trajectory_id"].isin(
                            tracklets_duplicated["tmp_traj"]
                        )
                    ]

                updated_trajectories = np.union1d(
                    updated_trajectories, np.unique(traj_left["trajectory_id"])
                ).tolist()

                nb_assoc_with_duplicates = len(traj_extremity_associated)
                night_to_night_traj_to_tracklets_report[
                    "number of duplicated association"
                ] = 0
                night_to_night_traj_to_tracklets_report["metrics"] = {}

                # remove duplicates associations
                traj_extremity_associated = traj_extremity_associated[~duplicates]

                night_to_night_traj_to_tracklets_report[
                    "number of duplicated association"
                ] = nb_assoc_with_duplicates - len(traj_extremity_associated)

                associated_tracklets = []

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
                associated_tracklets = pd.concat(associated_tracklets)

                # remove the tracklets that will be added to a trajectory from the dataframe of all tracklets
                tracklets = tracklets[
                    ~tracklets["trajectory_id"].isin(
                        traj_extremity_associated["tmp_traj"]
                    )
                ]

                # concatenation of trajectories with new tracklets
                trajectories = pd.concat([trajectories, associated_tracklets])

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

            if run_metrics:
                last_traj_obs = (
                    two_last_current_nid.groupby(["trajectory_id"]).last().reset_index()
                )

                inter_night_metric = compute_inter_night_metric(
                    last_traj_obs,
                    tracklets_extremity,
                    traj_left,
                    traj_extremity_associated,
                )
                night_to_night_traj_to_tracklets_report["metrics"] = inter_night_metric

            current_nid_assoc_report[
                "trajectories_to_tracklets_report"
            ] = night_to_night_traj_to_tracklets_report
            traj_to_track_report.append(current_nid_assoc_report)

        if len(updated_trajectories) > 0:
            trajectories_and_tracklets_associations_report[
                "updated trajectories"
            ] = updated_trajectories
        else:
            trajectories_and_tracklets_associations_report[
                "updated trajectories"
            ] = updated_trajectories

        trajectories_and_tracklets_associations_report[
            "all nid_report"
        ] = traj_to_track_report

        return (
            trajectories.reset_index(drop=True).infer_objects(),
            tracklets.reset_index(drop=True).infer_objects(),
            trajectories_and_tracklets_associations_report,
        )


def trajectories_associations(
    trajectories,
    new_observations,
    next_nid,
    sep_criterion,
    mag_criterion_same_fid,
    mag_criterion_diff_fid,
    angle_criterion,
    run_metrics=False,
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
    run_metrics : boolean
        run the inter night association metrics : trajectory_df, traj_next_night, old_observations and new_observations
        parameters should have the ssnamenr column.

    Return
    ------
    trajectories : dataframe
        trajectories associated with new observations
    new_observations : dataframe
        remaining new observations
    traj_to_obs_report : dict list
        statistics about the trajectory and observations association process, contains the following entries :

            list of updated trajectories

            all nid report with the following entries for each reports :

                        current nid

                        trajectories to observations report

                        metrics, no sense if run_metrics is set to False

    Examples
    --------
    >>> trajectories = ts.trajectories_sample_3
    >>> new_observations = ts.new_observations_sample_1

    >>> tr_orb_columns = [
    ... "provisional designation",
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
    ... "rms_mean anomaly"
    ... ]

    >>> trajectories[tr_orb_columns] = -1.0
    >>> new_observations[tr_orb_columns] = -1.0
    >>> trajectories["not_updated"] = np.ones(len(trajectories), dtype=np.bool_)

    >>> tr, obs, report = trajectories_associations(
    ... trajectories, new_observations, 3, 1.5 * u.degree, 0.2, 0.5, 30
    ... )

    >>> assert_frame_equal(tr, ts.trajectories_expected_4, check_dtype=False)
    >>> assert_frame_equal(obs, ts.new_observations_expected_1, check_dtype=False)
    >>> TestCase().assertDictEqual(ts.expected_traj_obs_report_1, report)


    >>> trajectories = ts.trajectories_sample_4
    >>> new_observations = ts.new_observations_sample_2

    >>> tr_orb_columns = [
    ... "provisional designation",
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
    ... "rms_mean anomaly"
    ... ]

    >>> trajectories[tr_orb_columns] = -1.0
    >>> new_observations[tr_orb_columns] = -1.0
    >>> trajectories["not_updated"] = np.ones(len(trajectories), dtype=np.bool_)

    >>> tr, obs, report = trajectories_associations(
    ... trajectories, new_observations, 3, 1.5 * u.degree, 0.2, 0.5, 30
    ... )

    >>> assert_frame_equal(tr, ts.trajectories_expected_5, check_dtype=False)
    >>> assert_frame_equal(obs, ts.new_observations_expected_2, check_dtype=False)
    >>> TestCase().assertDictEqual(ts.expected_traj_obs_report_2, report)


    >>> trajectories = ts.trajectories_sample_5
    >>> new_observations = ts.new_observations_sample_3

    >>> tr_orb_columns = [
    ... "provisional designation",
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
    ... "rms_mean anomaly"
    ... ]

    >>> trajectories[tr_orb_columns] = -1.0
    >>> new_observations[tr_orb_columns] = -1.0
    >>> trajectories["not_updated"] = np.ones(len(trajectories), dtype=np.bool_)

    >>> tr, obs, report = trajectories_associations(
    ... trajectories, new_observations, 3, 1.5 * u.degree, 0.2, 0.5, 30
    ... )

    >>> assert_frame_equal(tr, ts.trajectories_expected_6, check_dtype=False)
    >>> assert_frame_equal(obs, ts.new_observations_expected_3, check_dtype=False)
    >>> TestCase().assertDictEqual(ts.expected_traj_obs_report_3, report)
    """

    if len(trajectories) == 0 or len(new_observations) == 0:
        return trajectories, new_observations, {}
    else:

        trajectories_not_updated = trajectories[
            trajectories["not_updated"] & (trajectories["a"] == -1.0)
        ]

        trajectories_and_observations_associations_report = dict()

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

        traj_to_obs_report = []
        updated_trajectories = []

        # for each trajectory nid from the last observations
        for tr_nid in trajectories_nid:

            current_nid_assoc_report = dict()
            current_nid_assoc_report["old nid"] = int(tr_nid)

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
            (
                traj_left,
                obs_assoc,
                night_to_night_traj_to_obs_report,
            ) = night_to_night_trajectory_associations(
                two_last_current_nid,
                new_observations,
                norm_sep_crit,
                norm_same_fid,
                norm_diff_fid,
                angle_criterion,
            )

            # remove the associateds observations from the set of new observations
            new_observations = new_observations[
                ~new_observations["candid"].isin(obs_assoc["candid"])
            ]

            nb_traj_to_obs_assoc_with_duplicates = len(obs_assoc)
            night_to_night_traj_to_obs_report["number of duplicated association"] = 0
            night_to_night_traj_to_obs_report["metrics"] = {}

            if len(obs_assoc) > 0:

                # duplicates management is not really optimized, optimisation would be to remove the for loop
                # and vectorized the operation

                # creates a dataframe for each duplicated trajectory associated with the tracklets
                duplicates = obs_assoc["trajectory_id"].duplicated()
                all_duplicate_traj = []

                # get the duplicated tracklets
                duplicate_obs = obs_assoc[duplicates]

                orbit_column = [
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

                if len(duplicate_obs) > 0:

                    # get the max trajectory id
                    max_traj_id = np.max(np.unique(trajectories["trajectory_id"]))

                    for _, rows in duplicate_obs.iterrows():
                        max_traj_id += 1

                        # silence the copy warning
                        with pd.option_context("mode.chained_assignment", None):

                            # get the trajectory associated with the tracklets and update the trajectory id
                            duplicate_traj = trajectories[
                                trajectories["trajectory_id"] == rows["trajectory_id"]
                            ]
                            duplicate_traj[orbit_column] = -1.0
                            duplicate_traj["trajectory_id"] = max_traj_id
                            duplicate_traj["not_updated"] = False

                            # get the observations and update the trajectory id
                            obs_duplicate = pd.DataFrame(rows.copy()).T

                            obs_duplicate[orbit_column] = -1.0
                            obs_duplicate["trajectory_id"] = max_traj_id
                            obs_duplicate["not_updated"] = False

                        # append the trajectory and the tracklets into the list
                        all_duplicate_traj.append(duplicate_traj)
                        all_duplicate_traj.append(obs_duplicate)
                        updated_trajectories.append(max_traj_id)

                    # creates a dataframe with all duplicates and adds it to the trajectories dataframe
                    all_duplicate_traj = pd.concat(all_duplicate_traj)

                    trajectories = pd.concat([trajectories, all_duplicate_traj])

                updated_trajectories = np.union1d(
                    updated_trajectories, np.unique(obs_assoc["trajectory_id"])
                ).tolist()

                # remove duplicates associations
                obs_assoc = obs_assoc[~duplicates]

                # traj_left = traj_left.drop_duplicates(["trajectory_id"])

                night_to_night_traj_to_obs_report[
                    "number of duplicated association"
                ] = nb_traj_to_obs_assoc_with_duplicates - len(obs_assoc)

                # add the new associated observations in the recorded trajectory dataframe
                trajectories = pd.concat([trajectories, obs_assoc])

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

            if run_metrics:
                last_traj_obs = (
                    two_last_current_nid.groupby(["trajectory_id"]).last().reset_index()
                )
                inter_night_metric = compute_inter_night_metric(
                    last_traj_obs, new_observations, traj_left, obs_assoc
                )
                night_to_night_traj_to_obs_report["metrics"] = inter_night_metric

            current_nid_assoc_report[
                "trajectories_to_new_observation_report"
            ] = night_to_night_traj_to_obs_report

            traj_to_obs_report.append(current_nid_assoc_report)

        trajectories_and_observations_associations_report[
            "traj to obs report"
        ] = traj_to_obs_report
        trajectories_and_observations_associations_report[
            "updated trajectories"
        ] = updated_trajectories

        return (
            trajectories.reset_index(drop=True).infer_objects(),
            new_observations.reset_index(drop=True).infer_objects(),
            trajectories_and_observations_associations_report,
        )


def prep_orbit_computation(trajectory_df):
    trajectory_df["jd"] = pd.to_numeric(trajectory_df["jd"])
    trajectory_df["trajectory_id"] = trajectory_df["trajectory_id"].astype(int)

    traj_length = (
        trajectory_df.groupby(["trajectory_id"]).agg({"ra": len}).reset_index()
    )

    traj_id_sup_to_3 = traj_length[traj_length["ra"] >= 3]["trajectory_id"]
    traj_id_inf_to_3 = traj_length[traj_length["ra"] < 3]["trajectory_id"]
    track_to_orb = trajectory_df[trajectory_df["trajectory_id"].isin(traj_id_sup_to_3)]
    other_track = trajectory_df[trajectory_df["trajectory_id"].isin(traj_id_inf_to_3)]

    return other_track, track_to_orb


def compute_orbit_elem(trajectory_df):

    if len(trajectory_df) == 0:
        return trajectory_df

    traj_to_compute = trajectory_df[trajectory_df["a"] == -1.0]
    traj_with_orbelem = trajectory_df[trajectory_df["a"] != -1.0]

    print(
        "nb traj to compute orb elem: {}".format(
            len(np.unique(traj_to_compute["trajectory_id"]))
        )
    )

    orbit_column = [
        "provisional designation",
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

    orbit_elem = compute_df_orbit_param(traj_to_compute, 10, ram_dir)
    traj_to_compute = traj_to_compute.merge(orbit_elem, on="trajectory_id")

    return pd.concat([traj_with_orbelem, traj_to_compute])


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

    traj_next_night = new_trajectory_id_assignation(
        new_left, new_right, last_trajectory_id
    )

    intra_night_report["number of intra night tracklets"] = len(
        np.unique(traj_next_night["trajectory_id"])
    )

    if len(traj_next_night) > 0:
        last_trajectory_id = np.max(traj_next_night["trajectory_id"]) + 1
    else:
        traj_next_night = pd.DataFrame(columns=["trajectory_id", "candid"])

    # remove all the alerts that appears in the tracklets
    new_observation_not_associated = new_observation[
        ~new_observation["candid"].isin(traj_next_night["candid"])
    ]

    return (
        traj_next_night,
        last_trajectory_id,
        new_observation_not_associated,
        intra_night_report,
    )


def night_to_night_association(
    trajectory_df,
    old_observation,
    new_observation,
    last_nid,
    next_nid,
    time_window,
    intra_night_sep_criterion=145 * u.arcsecond,
    intra_night_mag_criterion_same_fid=2.21,
    intra_night_mag_criterion_diff_fid=1.75,
    sep_criterion=0.24 * u.degree,
    mag_criterion_same_fid=0.18,
    mag_criterion_diff_fid=0.7,
    angle_criterion=8.8,
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

    last_trajectory_id = np.nan_to_num(np.max(trajectory_df["trajectory_id"])) + 1
    (old_traj, most_recent_traj), old_observation = time_window_management(
        trajectory_df, old_observation, last_nid, next_nid, time_window
    )

    inter_night_report = dict()
    inter_night_report["nid of the next night"] = int(next_nid)

    # intra night associations steps with the new observations
    (
        tracklets,
        last_trajectory_id,
        remaining_new_observations,
        intra_night_report,
    ) = intra_night_step(
        new_observation,
        last_trajectory_id,
        intra_night_sep_criterion,
        intra_night_mag_criterion_same_fid,
        intra_night_mag_criterion_diff_fid,
        run_metrics,
    )

    # temporary import
    import time as t

    if len(trajectory_df) == 0:

        t_before = t.time()

        other_track, track_to_orb = prep_orbit_computation(tracklets)
        track_with_orb_elem = compute_orbit_elem(track_to_orb)

        print("orb elem computation time: {}".format(t.time() - t_before))

        return (
            pd.concat([other_track, track_with_orb_elem]),
            remaining_new_observations,
            intra_night_report,
        )

    # perform associations with the recorded trajectories :
    #   - trajectories with tracklets

    print("tracklets associations")

    (
        traj_with_track,
        not_associated_tracklets,
        traj_and_track_assoc_report,
    ) = tracklets_associations(
        most_recent_traj,
        tracklets,
        next_nid,
        sep_criterion,
        mag_criterion_same_fid,
        mag_criterion_diff_fid,
        angle_criterion,
        run_metrics,
    )

    # restore the type of each columns with the infer_objects columns
    # already done in tracklets_associations
    # most_recent_traj = most_recent_traj.reset_index(drop=True).infer_objects()

    # get the trajectories updated with new tracklets and the trajectory not updated for the next step
    traj_to_orb = traj_with_track[~traj_with_track["not_updated"]]
    traj_not_updated = traj_with_track[traj_with_track["not_updated"]]

    # separate traklets with more than 3 points for the orbit computation and the other tracklets
    other_track, track_to_orb = prep_orbit_computation(not_associated_tracklets)

    # concatenate the updated trajectories and the tracklets with more than 3 points
    all_traj_to_orb = pd.concat([traj_to_orb, track_to_orb])

    if len(all_traj_to_orb) > 0:
        print(
            "updated trajectories: {}".format(
                len(np.unique(all_traj_to_orb["trajectory_id"]))
            )
        )

    t_before = t.time()

    track_traj_with_orb = compute_orbit_elem(all_traj_to_orb)
    print("orb elem computation time: {}".format(t.time() - t_before))

    print("trajectories associations")

    # perform associations with the recorded trajectories :
    #   - trajectories with new observations
    (
        traj_with_new_obs,
        remaining_new_observations,
        trajectories_associations_report,
    ) = trajectories_associations(
        traj_not_updated,
        remaining_new_observations,
        next_nid,
        sep_criterion,
        mag_criterion_same_fid,
        mag_criterion_diff_fid,
        angle_criterion,
        run_metrics,
    )

    # separate trajectories with more than 3 points for the orbit computation and the other tracklets
    other_traj, traj_to_orb = prep_orbit_computation(traj_with_new_obs)

    if len(traj_to_orb) > 0:
        print(
            "updated trajectories: {}".format(
                len(np.unique(traj_to_orb["trajectory_id"]))
            )
        )

    t_before = t.time()
    new_traj_with_orb = compute_orbit_elem(traj_to_orb)
    print("orb elem computation time: {}".format(t.time() - t_before))

    # concatenate all the trajectories with computed orbital elements with the other trajectories/tracklets.
    most_recent_traj = pd.concat(
        [track_traj_with_orb, new_traj_with_orb, other_traj, other_track]
    )

    old_observation = pd.concat([old_observation, remaining_new_observations])

    inter_night_report["intra night report"] = intra_night_report
    inter_night_report["trajectory association report"] = traj_and_track_assoc_report

    # inter_night_report[
    #     "tracklets and observation association report"
    # ] = tracklets_and_observation_report

    trajectory_df = pd.concat([old_traj, most_recent_traj])

    return trajectory_df, old_observation, inter_night_report


if __name__ == "__main__":  # pragma: no cover
    import sys
    import doctest
    from pandas.testing import assert_frame_equal  # noqa: F401
    import test_sample as ts  # noqa: F401
    from unittest import TestCase  # noqa: F401

    if "unittest.util" in __import__("sys").modules:
        # Show full diff in self.assertEqual.
        __import__("sys").modules["unittest.util"]._MAX_LENGTH = 999999999

    sys.exit(doctest.testmod()[0])

    def print_df_to_dict(df):
        print("{")
        for col in df.columns:
            print('"{}": {},'.format(col, list(df[col])))
        print("}")

    from alert_association.continuous_integration import load_data

    df_sso = load_data("Solar System MPC", 0)

    tr_orb_columns = [
        "provisional designation",
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

        print(tr_nid)
        print()

        if tr_nid > 1540:
            break

        new_observation = df_sso[df_sso["nid"] == tr_nid]
        with pd.option_context("mode.chained_assignment", None):
            new_observation[tr_orb_columns] = -1.0

        next_nid = new_observation["nid"].values[0]

        print("nb new obs: {}".format(len(new_observation)))

        trajectory_df, old_obs, report = night_to_night_association(
            trajectory_df,
            old_observation,
            new_observation,
            last_nid,
            next_nid,
            time_window=5,
            sep_criterion=16.45 * u.arcminute,
            mag_criterion_same_fid=0.1,
            mag_criterion_diff_fid=0.34,
            angle_criterion=1.19,
        )

        trajectory_df["not_updated"] = np.ones(len(trajectory_df), dtype=np.bool_)

        print()
        print()
        print(trajectory_df[trajectory_df["a"] != -1.0])
        print()
        print("--------------------")
        print()

        last_nid = next_nid

    sys.exit(doctest.testmod()[0])

    # >>> trajectory_df, old_observation, inter_night_report = night_to_night_association(
    # ... ts.night_night_trajectory_sample,
    # ... ts.night_to_night_old_obs,
    # ... ts.night_to_night_new_obs,
    # ... 4,
    # ... intra_night_sep_criterion=1.5 * u.degree,
    # ... intra_night_mag_criterion_same_fid=0.2,
    # ... intra_night_mag_criterion_diff_fid=0.5,
    # ... sep_criterion = 1.5 * u.degree,
    # ... mag_criterion_same_fid = 0.2,
    # ... mag_criterion_diff_fid = 0.5,
    # ... angle_criterion = 30,
    # ... run_metrics = True
    # ... )

    # >>> TestCase().assertDictEqual(ts.inter_night_report1, inter_night_report)

    # >>> assert_frame_equal(trajectory_df.reset_index(drop=True), ts.night_to_night_trajectory_df_expected, check_dtype=False)

    # >>> assert_frame_equal(old_observation.reset_index(drop=True), ts.night_to_night_old_observation_expected)

    # >>> trajectory_df, old_observation, inter_night_report = night_to_night_association(
    # ... pd.DataFrame(),
    # ... ts.night_to_night_old_obs,
    # ... ts.night_to_night_new_obs,
    # ... 4,
    # ... intra_night_sep_criterion=1.5 * u.degree,
    # ... intra_night_mag_criterion_same_fid=0.2,
    # ... intra_night_mag_criterion_diff_fid=0.5,
    # ... sep_criterion = 1.5 * u.degree,
    # ... mag_criterion_same_fid = 0.2,
    # ... mag_criterion_diff_fid = 0.5,
    # ... angle_criterion = 30
    # ... )

    # >>> TestCase().assertDictEqual(ts.inter_night_report2, inter_night_report)

    # >>> assert_frame_equal(trajectory_df.reset_index(drop=True), ts.night_to_night_trajectory_df_expected2, check_dtype=False)

    # >>> assert_frame_equal(old_observation.reset_index(drop=True), ts.night_to_night_old_observation_expected2)

    # >>> trajectory_df, old_observation, inter_night_report = night_to_night_association(
    # ... ts.night_night_trajectory_sample,
    # ... ts.night_to_night_old_obs,
    # ... ts.night_to_night_new_obs2,
    # ... 4,
    # ... intra_night_sep_criterion=1.5 * u.degree,
    # ... intra_night_mag_criterion_same_fid=0.2,
    # ... intra_night_mag_criterion_diff_fid=0.5,
    # ... sep_criterion = 1.5 * u.degree,
    # ... mag_criterion_same_fid = 0.2,
    # ... mag_criterion_diff_fid = 0.5,
    # ... angle_criterion = 30
    # ... )

    # >>> TestCase().assertDictEqual(ts.inter_night_report3, inter_night_report)

    # >>> assert_frame_equal(trajectory_df.reset_index(drop=True), ts.night_to_night_trajectory_df_expected3, check_dtype=False)

    # >>> assert_frame_equal(old_observation.reset_index(drop=True), ts.night_to_night_old_observation_expected3)
