import numpy as np
import pandas as pd
from alert_association.night_to_night_association import night_to_night_trajectory_associations
from alert_association.intra_night_association import get_n_last_observations_from_trajectories
from alert_association.intra_night_association import compute_inter_night_metric
import astropy.units as u

def tracklets_associations(
    trajectories,
    tracklets,
    next_nid,
    sep_criterion,
    mag_criterion_same_fid,
    mag_criterion_diff_fid,
    angle_criterion,
    run_metrics=False
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

                        number of trajectories to tracklets duplicated associations

                        trajectories to new observations report

                        number of trajectories to new observations duplicated associations

                        metrics, no sense if run_metrics is set to False

    Examples
    --------
    >>> trajectories = ts.trajectory_df_sample
    >>> trajectories["not_updated"] = np.ones(len(trajectories), dtype=np.bool_)
    >>> tracklets = ts.traj_next_night_sample
    >>> tr, tk, report = tracklets_associations(trajectories, tracklets, 4, 2 * u.degree, 0.2, 0.5, 30)

    >>> assert_frame_equal(tr, ts.trajectories_expected_1, check_dtype=False)
    >>> assert_frame_equal(tk, ts.tracklets_expected_1, check_dtype=False)

    >>> trajectories = ts.trajectories_sample_1
    >>> trajectories["not_updated"] = np.ones(len(trajectories), dtype=np.bool_)
    >>> tracklets = ts.tracklets_sample_1
    >>> tr2, tk2, report = tracklets_associations(trajectories, tracklets, 3, 1.5 * u.degree, 0.2, 0.5, 30, True)

    >>> assert_frame_equal(tr2, ts.trajectories_expected_2, check_dtype=False)
    >>> assert_frame_equal(tk2, ts.tracklets_expected_2, check_dtype=False)
    >>> TestCase().assertDictEqual(ts.traj_and_track_assoc_report_expected, report)
    """

    trajectories_not_updated = trajectories[trajectories["not_updated"]]
    trajectories_and_tracklets_associations_report = dict()

    if len(trajectories_not_updated) == 0 or len(tracklets) == 0:
        return trajectories, tracklets
    else:

        # get the last two observations for each trajectories
        two_last_observation_trajectory = get_n_last_observations_from_trajectories(
            trajectories_not_updated, 2
        )

        # get the last observations for each trajectories
        last_observation_trajectory = get_n_last_observations_from_trajectories(
            trajectories_not_updated, 1
        )

        # get the oldest extremity of the new tracklets to perform associations with the latest observations in the trajectories
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

            updated_trajectories = np.union1d(updated_trajectories, np.unique(traj_left["trajectory_id"]))
            nb_assoc_with_duplicates = len(traj_extremity_associated)
            night_to_night_traj_to_tracklets_report["number of duplicated association"] = 0
            night_to_night_traj_to_tracklets_report["metrics"] = {}

            if len(traj_extremity_associated) > 0:

                # remove duplicates associations
                # do somethings with the duplicates later in the project
                traj_extremity_associated = traj_extremity_associated.drop_duplicates(
                    ["trajectory_id"]
                )
                traj_left = traj_left.drop_duplicates(["trajectory_id"])

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

                # concatenation of trajectory_df with new tracklets doesn't work if we decide to manage the multiples associations.
                # need to keep track of multiple association with a list of trajectory_id.
                trajectories = pd.concat([trajectories, associated_tracklets])
                
                associated_tr_id = np.unique(associated_tracklets["trajectory_id"])
                
                trajectories = trajectories.reset_index(drop=True)
                tr_updated_index = trajectories[trajectories["trajectory_id"].isin(associated_tr_id)].index
                trajectories.loc[tr_updated_index, "not_updated"] = False


                # remove the two last trajectory observation that have been associated during this loop.
                two_last_current_nid = two_last_current_nid[
                    ~two_last_current_nid["trajectory_id"].isin(traj_left["trajectory_id"])
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

        trajectories_and_tracklets_associations_report['updated trajectories'] = updated_trajectories.tolist()
        trajectories_and_tracklets_associations_report['all nid_report'] = traj_to_track_report
        return trajectories, tracklets.reset_index(drop=True), trajectories_and_tracklets_associations_report

def print_df(df):
    for c in df.columns:
        print("\"{}\" : {},".format(c, df[c].to_list()))

if __name__ == "__main__": # pragma: no cover
    import sys
    import doctest
    from pandas.testing import assert_frame_equal  # noqa: F401
    import test_sample as ts  # noqa: F401
    from unittest import TestCase  # noqa: F401

    if "unittest.util" in __import__("sys").modules:
        # Show full diff in self.assertEqual.
        __import__("sys").modules["unittest.util"]._MAX_LENGTH = 999999999

    sys.exit(doctest.testmod()[0])