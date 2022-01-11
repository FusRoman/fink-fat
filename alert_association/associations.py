import astropy.units as u
from astropy.coordinates import SkyCoord
import pandas as pd
import numpy as np

from alert_association.intra_night_association import magnitude_association
from alert_association.intra_night_association import (
    get_n_last_observations_from_trajectories,
)
from alert_association.intra_night_association import compute_inter_night_metric


def night_to_night_separation_association(
    old_observation, new_observation, separation_criterion
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

    new_obs_idx, old_obs_idx, sep2d, _ = old_observations_coord.search_around_sky(
        new_observations_coord, separation_criterion
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

    ra_x, dec_x, jd_x = x[1], x[2], x[3]

    ra_y, dec_y, jd_y = x[6], x[7], x[8]

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
    obs_set1, obs_set2, sep_criterion, mag_criterion_same_fid, mag_criterion_diff_fid
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
    inter_night_obs_assoc_report : dictionary
        statistics about the observation association process, contains the following entries :

                    number of inter night separation based association

                    number of inter night magnitude filtered association

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

    >>> left, right, report = night_to_night_observation_association(night_1, night_2, 1.5 * u.degree, 0.2, 0.5)

    >>> expected_report = {'number of inter night separation based association': 3, 'number of inter night magnitude filtered association': 1}

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
    >>> TestCase().assertDictEqual(expected_report, report)
    """

    inter_night_obs_assoc_report = dict()

    # association based separation
    traj_assoc, new_obs_assoc, sep = night_to_night_separation_association(
        obs_set1, obs_set2, sep_criterion
    )

    inter_night_obs_assoc_report[
        "number of inter night separation based association"
    ] = len(new_obs_assoc)

    # filter the association based on magnitude criterion
    traj_assoc, new_obs_assoc = magnitude_association(
        traj_assoc,
        new_obs_assoc,
        mag_criterion_same_fid,
        mag_criterion_diff_fid,
        jd_normalization=True,
    )

    inter_night_obs_assoc_report[
        "number of inter night magnitude filtered association"
    ] = inter_night_obs_assoc_report[
        "number of inter night separation based association"
    ] - len(
        new_obs_assoc
    )

    return traj_assoc, new_obs_assoc, inter_night_obs_assoc_report


def night_to_night_trajectory_associations(
    two_last_observations,
    observations_to_associates,
    sep_criterion,
    mag_criterion_same_fid,
    mag_criterion_diff_fid,
    angle_criterion,
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
    inter_night_obs_report : dictionary
        statistics about the trajectory association process, contains the following entries :

                    number of inter night separation based association

                    number of inter night magnitude filtered association

                    number of inter night angle filtered association

    Examples
    --------
    >>> left, right, report = night_to_night_trajectory_associations(ts.night_to_night_two_last_sample, ts.night_to_night_new_observation, 2 * u.degree, 0.2, 0.5, 30)

    >>> expected_report = {'number of inter night separation based association': 6, 'number of inter night magnitude filtered association': 1, 'number of inter night angle filtered association': 1}

    >>> assert_frame_equal(left.reset_index(drop=True), ts.night_to_night_traj_assoc_left_expected)
    >>> assert_frame_equal(right.reset_index(drop=True), ts.night_to_night_traj_assoc_right_expected)
    >>> TestCase().assertDictEqual(expected_report, report)
    """
    # get the last observations of the trajectories to perform the associations
    last_traj_obs = (
        two_last_observations.groupby(["trajectory_id"]).last().reset_index()
    )

    (
        traj_assoc,
        new_obs_assoc,
        inter_night_obs_report,
    ) = night_to_night_observation_association(
        last_traj_obs,
        observations_to_associates,
        sep_criterion,
        mag_criterion_same_fid,
        mag_criterion_diff_fid,
    )

    nb_assoc_before_angle_filtering = len(new_obs_assoc)
    inter_night_obs_report["number of inter night angle filtered association"] = 0
    if len(traj_assoc) != 0:

        traj_assoc, new_obs_assoc = cone_search_association(
            two_last_observations, traj_assoc, new_obs_assoc, angle_criterion
        )

        inter_night_obs_report[
            "number of inter night angle filtered association"
        ] = nb_assoc_before_angle_filtering - len(new_obs_assoc)

        return traj_assoc, new_obs_assoc, inter_night_obs_report
    else:
        return traj_assoc, new_obs_assoc, inter_night_obs_report


# TODO : tester le retour des rapports pour tracklets_associations
def tracklets_and_trajectories_associations(
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

    >>> tr, tk, report = tracklets_and_trajectories_associations(trajectories, tracklets, 4, 2 * u.degree, 0.2, 0.5, 30)

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

    >>> tr, tk, report = tracklets_and_trajectories_associations(trajectories, tracklets, 3, 1.5 * u.degree, 0.2, 0.5, 30, True)

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

    >>> tr, tk, report = tracklets_and_trajectories_associations(trajectories, tracklets, 3, 1 * u.degree, 0.2, 0.5, 30)

    >>> assert_frame_equal(tr, ts.trajectories_expected_3, check_dtype=False)
    >>> assert_frame_equal(tk, pd.DataFrame(columns=["ra", "dec", "dcmag", "fid", "nid", "jd", "candid", "trajectory_id", "provisional designation", "a", "e", "i", "long. node", "arg. peric", "mean anomaly", "rms_a", "rms_e", "rms_i", "rms_long. node", "rms_arg. peric", "rms_mean anomaly",]), check_index_type=False, check_dtype=False)

    >>> tr, tk, report = tracklets_and_trajectories_associations(pd.DataFrame(), tracklets, 3, 1 * u.degree, 0.2, 0.5, 30)

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


def trajectories_with_new_observations_associations(
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

    >>> tr, obs, report = trajectories_with_new_observations_associations(
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

    >>> tr, obs, report = trajectories_with_new_observations_associations(
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

    >>> tr, obs, report = trajectories_with_new_observations_associations(
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

        # remove tmp_traj column added by the cone_search_associations function in  night_to_night_trajectory_associations
        if "tmp_traj" in trajectories:
            trajectories = trajectories.drop("tmp_traj", axis=1)

        return (
            trajectories.reset_index(drop=True).infer_objects(),
            new_observations.reset_index(drop=True).infer_objects(),
            trajectories_and_observations_associations_report,
        )


def old_observations_with_tracklets_associations(
    tracklets,
    old_observations,
    next_nid,
    sep_criterion,
    mag_criterion_same_fid,
    mag_criterion_diff_fid,
    angle_criterion,
    run_metrics=False,
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
    run_metrics : boolean
        run the inter night association metrics : trajectory_df, traj_next_night, old_observations and new_observations
        parameters should have the ssnamenr column.

    Return
    ------
    tracklets : dataframe
        tracklets associated with old observations
    old observations : dataframe
        remaining old observations
    track_and_obs_report : dict list
        statistics about the trajectory and observations association process, contains the following entries :

            list of updated tracklets

            all nid report with the following entries for each reports :

                        current nid

                        trajectories to observations report

                        metrics, no sense if run_metrics is set to False


    Examples
    --------
    >>> tracklets = ts.tracklets_sample_3
    >>> old_observations = ts.old_observations_sample_1

    >>> tr_orb_columns = [
    ...    "provisional designation",
    ...    "a",
    ...    "e",
    ...    "i",
    ...    "long. node",
    ...    "arg. peric",
    ...    "mean anomaly",
    ...    "rms_a",
    ...    "rms_e",
    ...    "rms_i",
    ...    "rms_long. node",
    ...    "rms_arg. peric",
    ...    "rms_mean anomaly",
    ... ]

    >>> tracklets[tr_orb_columns] = -1.0
    >>> old_observations[tr_orb_columns] = -1.0
    >>> tracklets["not_updated"] = np.ones(len(tracklets), dtype=np.bool_)

    >>> tk, old, report = old_observations_with_tracklets_associations(tracklets, old_observations, 3, 1.5 * u.degree, 0.1, 0.3, 30)


    >>> assert_frame_equal(tk, ts.tracklets_obs_expected_1, check_dtype=False)
    >>> assert_frame_equal(old.reset_index(drop=True), ts.old_obs_expected_1, check_dtype=False)
    >>> TestCase().assertDictEqual(ts.track_and_obs_report_expected_1, report)


    >>> tracklets = ts.tracklets_sample_4
    >>> old_observations = ts.old_observations_sample_2

    >>> tr_orb_columns = [
    ...    "provisional designation",
    ...    "a",
    ...    "e",
    ...    "i",
    ...    "long. node",
    ...    "arg. peric",
    ...    "mean anomaly",
    ...    "rms_a",
    ...    "rms_e",
    ...    "rms_i",
    ...    "rms_long. node",
    ...    "rms_arg. peric",
    ...    "rms_mean anomaly",
    ... ]

    >>> tracklets[tr_orb_columns] = -1.0
    >>> old_observations[tr_orb_columns] = -1.0
    >>> tracklets["not_updated"] = np.ones(len(tracklets), dtype=np.bool_)

    >>> tk, old, report = old_observations_with_tracklets_associations(tracklets, old_observations, 3, 1.5 * u.degree, 0.1, 0.3, 30)

    >>> assert_frame_equal(tk, ts.tracklets_obs_expected_2, check_dtype=False)
    >>> assert_frame_equal(old.reset_index(drop=True), ts.old_obs_expected_2, check_dtype=False)
    >>> TestCase().assertDictEqual(ts.track_and_obs_report_expected_2, report)


    >>> tracklets = ts.tracklets_sample_5
    >>> old_observations = ts.old_observations_sample_3

    >>> tr_orb_columns = [
    ...    "provisional designation",
    ...    "a",
    ...    "e",
    ...    "i",
    ...    "long. node",
    ...    "arg. peric",
    ...    "mean anomaly",
    ...    "rms_a",
    ...    "rms_e",
    ...    "rms_i",
    ...    "rms_long. node",
    ...    "rms_arg. peric",
    ...    "rms_mean anomaly",
    ... ]

    >>> tracklets[tr_orb_columns] = -1.0
    >>> old_observations[tr_orb_columns] = -1.0
    >>> tracklets["not_updated"] = np.ones(len(tracklets), dtype=np.bool_)

    >>> tk, old, report = old_observations_with_tracklets_associations(tracklets, old_observations, 3, 1.5 * u.degree, 0.1, 0.3, 30)

    >>> assert_frame_equal(tk, ts.tracklets_obs_expected_3, check_dtype=False)
    >>> assert_frame_equal(old.reset_index(drop=True), ts.old_obs_expected_3, check_dtype=False)
    >>> TestCase().assertDictEqual(ts.track_and_obs_report_expected_3, report)
    """

    if len(tracklets) == 0 or len(old_observations) == 0:
        return tracklets, old_observations, {}
    else:

        track_and_obs_report = dict()

        # get all the old night id sort by descending order to begin the associations
        # with the recently ones
        old_obs_nid = np.sort(np.unique(old_observations["nid"]))[::-1]

        two_first_obs_tracklets = get_n_last_observations_from_trajectories(
            tracklets, 2, False
        )

        updated_tracklets = []
        all_nid_report = []

        for obs_nid in old_obs_nid:

            current_nid_report = dict()

            current_old_obs = old_observations[old_observations["nid"] == obs_nid]

            current_nid_report["old nid"] = int(obs_nid)

            diff_night = next_nid - obs_nid
            norm_sep_crit = sep_criterion * diff_night
            norm_same_fid = mag_criterion_same_fid * diff_night
            norm_diff_fid = mag_criterion_diff_fid * diff_night

            # association between the new tracklets and the old observations
            (
                track_left_assoc,
                old_obs_right_assoc,
                track_to_old_obs_report,
            ) = night_to_night_trajectory_associations(
                two_first_obs_tracklets,
                current_old_obs,
                norm_sep_crit,
                norm_same_fid,
                norm_diff_fid,
                angle_criterion,
            )

            # remove tmp_traj column added by the cone_search_associations function in  night_to_night_trajectory_associations
            if "tmp_traj" in old_obs_right_assoc:
                old_obs_right_assoc = old_obs_right_assoc.drop("tmp_traj", axis=1)

            # remove the associateds observations from the set of new observations
            old_observations = old_observations[
                ~old_observations["candid"].isin(old_obs_right_assoc["candid"])
            ]

            nb_track_to_obs_assoc_with_duplicates = len(old_obs_right_assoc)
            track_to_old_obs_report["number of duplicated association"] = 0
            track_to_old_obs_report["metrics"] = {}

            if len(track_left_assoc) > 0:

                # creates a dataframe for each duplicated trajectory associated with the tracklets
                duplicates = old_obs_right_assoc["trajectory_id"].duplicated()
                all_duplicate_traj = []

                # get the duplicated tracklets
                duplicate_obs = old_obs_right_assoc[duplicates]

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
                    max_traj_id = np.max(np.unique(tracklets["trajectory_id"]))

                    for _, rows in duplicate_obs.iterrows():
                        max_traj_id += 1

                        # silence the copy warning
                        with pd.option_context("mode.chained_assignment", None):

                            # get the trajectory associated with the tracklets and update the trajectory id
                            duplicate_track = tracklets[
                                tracklets["trajectory_id"] == rows["trajectory_id"]
                            ]
                            duplicate_track[orbit_column] = -1.0
                            duplicate_track["trajectory_id"] = max_traj_id
                            duplicate_track["not_updated"] = False

                            # get the observations and update the trajectory id
                            obs_duplicate = pd.DataFrame(rows.copy()).T

                            obs_duplicate[orbit_column] = -1.0
                            obs_duplicate["trajectory_id"] = max_traj_id
                            obs_duplicate["not_updated"] = False

                        # append the trajectory and the tracklets into the list
                        all_duplicate_traj.append(duplicate_track)
                        all_duplicate_traj.append(obs_duplicate)
                        updated_tracklets.append(max_traj_id)

                    # creates a dataframe with all duplicates and adds it to the trajectories dataframe
                    all_duplicate_traj = pd.concat(all_duplicate_traj)

                    tracklets = pd.concat([tracklets, all_duplicate_traj])

                # remove the duplicates()
                # remove duplicates associations
                old_obs_right_assoc = old_obs_right_assoc[~duplicates]
                # old_obs_right_assoc = old_obs_right_assoc.drop_duplicates(["trajectory_id"])
                # track_left_assoc = track_left_assoc.drop_duplicates(["trajectory_id"])

                track_to_old_obs_report[
                    "number of duplicated association"
                ] = nb_track_to_obs_assoc_with_duplicates - len(old_obs_right_assoc)

                # add the associated old observations to the tracklets
                tracklets = pd.concat([tracklets, old_obs_right_assoc])

                # remove the associated tracklets for the next loop turn
                two_first_obs_tracklets = two_first_obs_tracklets[
                    ~two_first_obs_tracklets["trajectory_id"].isin(
                        track_left_assoc["trajectory_id"]
                    )
                ]

                updated_tracklets = np.union1d(
                    updated_tracklets, np.unique(old_obs_right_assoc["trajectory_id"])
                ).tolist()

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
                current_old_obs = current_old_obs[
                    ~current_old_obs["candid"].isin(old_obs_right_assoc["candid"])
                ]

            if run_metrics:
                last_traj_obs = (
                    two_first_obs_tracklets.groupby(["trajectory_id"])
                    .last()
                    .reset_index()
                )
                inter_night_metric = compute_inter_night_metric(
                    last_traj_obs,
                    current_old_obs,
                    track_left_assoc,
                    old_obs_right_assoc,
                )
                track_to_old_obs_report["metrics"] = inter_night_metric

            current_nid_report[
                "old observation to tracklets report"
            ] = track_to_old_obs_report

            all_nid_report.append(current_nid_report)

        track_and_obs_report["track and obs report"] = all_nid_report
        track_and_obs_report["updated tracklets"] = updated_tracklets

        return (
            tracklets.reset_index(drop=True).infer_objects(),
            old_observations.reset_index(drop=True).infer_objects(),
            track_and_obs_report,
        )


def old_with_new_observations_associations(
    old_observations,
    new_observations,
    next_nid,
    last_trajectory_id,
    sep_criterion,
    mag_criterion_same_fid,
    mag_criterion_diff_fid,
    run_metrics=False,
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
    run_metrics : boolean
        run the inter night association metrics : trajectory_df, traj_next_night, old_observations and new_observations
        parameters should have the ssnamenr column.

    Return
    ------
    trajectory_df : dataframe
        the new trajectory detected by observations associations,
        They are only two points trajeectory so we don't compute orbital elements with this trajectory_df
    old_observations : dataframe
        remaining old observations after associations
    new_observations : dataframe
        remaining new observations after associations
    observations_associations_report : dictionnary
        statistics about the observations association process, contains the following entries :

            list of new trajectories

            all nid report with the following entries for each reports :

                        current nid

                        observations associations report

                        metrics, no sense if run_metrics is set to False

    Examples
    --------
    >>> new_tr, remain_old, remain_new, report = old_with_new_observations_associations(ts.old_observations_sample_4, ts.new_observation_sample_4, 3, 0, 1.5 * u.degree, 0.1, 0.3, 30,)

    >>> assert_frame_equal(new_tr.reset_index(drop=True), ts.new_trajectory, check_dtype=False)
    >>> assert_frame_equal(remain_old.reset_index(drop=True), ts.remaining_old_obs, check_dtype=False)
    >>> assert_frame_equal(remain_new.reset_index(drop=True), ts.remaining_new_obs, check_dtype=False)
    >>> TestCase().assertDictEqual(ts.expected_obs_report, report)
    """
    if len(old_observations) == 0 or len(new_observations) == 0:
        return old_observations, new_observations, {}
    else:

        observations_associations_report = dict()
        all_nid_report = []
        new_trajectories = []

        old_obs_nid = np.sort(np.unique(old_observations["nid"]))[::-1]
        trajectory_df = pd.DataFrame()

        for obs_nid in old_obs_nid:

            current_nid_report = dict()

            current_old_obs = old_observations[old_observations["nid"] == obs_nid]

            current_nid_report["old nid"] = int(obs_nid)

            diff_night = next_nid - obs_nid
            norm_sep_crit = sep_criterion * diff_night
            norm_same_fid = mag_criterion_same_fid * diff_night
            norm_diff_fid = mag_criterion_diff_fid * diff_night

            (
                left_assoc,
                right_assoc,
                old_to_new_assoc_report,
            ) = night_to_night_observation_association(
                current_old_obs,
                new_observations,
                norm_sep_crit,
                norm_same_fid,
                norm_diff_fid,
            )

            if run_metrics:
                inter_night_metric = compute_inter_night_metric(
                    current_old_obs, new_observations, left_assoc, right_assoc
                )
                old_to_new_assoc_report["metrics"] = inter_night_metric

            old_to_new_assoc_report["number of duplicated association"] = 0
            old_to_new_assoc_report[
                "number of inter night angle filtered association"
            ] = 0

            if len(left_assoc) > 0:

                new_trajectory_id = np.arange(
                    last_trajectory_id, last_trajectory_id + len(left_assoc)
                )
                new_trajectories = np.union1d(
                    new_trajectories, new_trajectory_id
                ).tolist()

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

                trajectory_df = pd.concat([trajectory_df, left_assoc, right_assoc])

            current_nid_report["observation associations"] = old_to_new_assoc_report
            all_nid_report.append(current_nid_report)

        observations_associations_report["new trajectories"] = new_trajectories
        observations_associations_report["all nid report"] = all_nid_report

        return (
            trajectory_df,
            old_observations,
            new_observations,
            observations_associations_report,
        )


def time_window_management(
    trajectory_df, old_observation, last_nid, nid_next_night, time_window
):
    """
    Management of the old observation and trajectories. Remove the old observation with a nid difference with
    the nid of the next night greater than the time window. Perform the same process for trajectories but take
    the most recent trajectory extremity.

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
    time_window : integer
        limit to keep old observation and trajectories

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
    ... "trajectory_id" : [1, 1, 1, 2, 2, 2],
    ... "a": [2.5, 2.5, 2.5, -1, -1, -1]
    ... })

    >>> test_obs = pd.DataFrame({
    ... "nid" : [1, 4, 11, 12],
    ... "candid" : [16, 17, 18, 19]
    ... })

    >>> (test_old_traj, test_most_recent_traj), test_old_obs = time_window_management(test_traj, test_obs, 12, 17, 3)

    >>> expected_old_traj = pd.DataFrame({
    ... "candid" : [10, 11, 12],
    ... "nid" : [1, 2, 3],
    ... "jd" : [1, 2, 3],
    ... "trajectory_id" : [1, 1, 1],
    ... "a" : [2.5, 2.5, 2.5]
    ... })

    >>> expected_most_recent_traj = pd.DataFrame({
    ... "candid" : [13, 14, 15],
    ... "nid" : [10, 11, 12],
    ... "jd" : [10, 11, 12],
    ... "trajectory_id" : [2, 2, 2],
    ... "a": [-1.0, -1.0, -1.0]
    ... })

    >>> expected_old_obs = pd.DataFrame({
    ... "nid" : [12],
    ... "candid" : [19]
    ... })

    >>> assert_frame_equal(expected_old_traj.reset_index(drop=True), test_old_traj.reset_index(drop=True))
    >>> assert_frame_equal(expected_most_recent_traj.reset_index(drop=True), test_most_recent_traj.reset_index(drop=True))
    >>> assert_frame_equal(expected_old_obs.reset_index(drop=True), test_old_obs.reset_index(drop=True))
    """
    most_recent_traj = pd.DataFrame()
    oldest_traj = pd.DataFrame()

    if len(trajectory_df) > 0:

        # get the lost observation of all trajectories
        last_obs_of_all_traj = get_n_last_observations_from_trajectories(
            trajectory_df, 1
        )

        # if the night difference exceed the time window
        if nid_next_night - last_nid > time_window:

            # get last observation of trajectories from the last nid
            last_obs_of_all_traj = last_obs_of_all_traj[
                last_obs_of_all_traj["nid"] == last_nid
            ]

            mask_traj = trajectory_df["trajectory_id"].isin(
                last_obs_of_all_traj["trajectory_id"]
            )
            most_recent_traj = trajectory_df[mask_traj]
            oldest_traj = trajectory_df[~mask_traj & (trajectory_df["a"] != -1.0)]
            old_observation = old_observation[old_observation["nid"] == last_nid]

            # return the trajectories from the last night id
            return (oldest_traj, most_recent_traj), old_observation

        last_obs_of_all_traj["diff_nid"] = nid_next_night - last_obs_of_all_traj["nid"]

        most_recent_last_obs = last_obs_of_all_traj[
            last_obs_of_all_traj["diff_nid"] <= time_window
        ]

        mask_traj = trajectory_df["trajectory_id"].isin(
            most_recent_last_obs["trajectory_id"]
        )

        most_recent_traj = trajectory_df[mask_traj]
        oldest_traj = trajectory_df[~mask_traj & (trajectory_df["a"] != -1.0)]

    diff_nid_old_observation = nid_next_night - old_observation["nid"]
    old_observation = old_observation[diff_nid_old_observation < time_window]

    return (oldest_traj, most_recent_traj), old_observation


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
