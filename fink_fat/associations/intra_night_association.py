import astropy.units as u
from astropy.coordinates import SkyCoord
import pandas as pd
import numpy as np
from collections import Counter
from astropy.coordinates import search_around_sky


def get_n_last_observations_from_trajectories(trajectories, n, ascending=True):
    """
    Get n extremity observations from trajectories

    Parameters
    ----------
    trajectories : dataframe
        a dataframe with a trajectory_id column that identify trajectory observations. (column trajectory_id and jd have to be present)
    n : integer
        the number of extremity observations to return.
    ascending : boolean
        if set to True, return the most recent extremity observations, return the oldest ones otherwise, default to True.

    Returns
    -------
    last_trajectories_observations : dataframe
        the n last observations from the recorded trajectories

    Examples
    --------
    >>> from pandera import Check, Column, DataFrameSchema

    >>> test = pd.DataFrame({
    ... "candid" : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    ... "jd" : [0.0, 1.0, 2, 3, 4, 5, 6, 7, 8, 9],
    ... "trajectory_id" : [1, 1, 1, 1, 2, 2, 3, 3, 3, 3]
    ... })

    >>> df_schema = DataFrameSchema({
    ... "jd": Column(float),
    ... "trajectory_id": Column(int)
    ... })

    >>> res = get_n_last_observations_from_trajectories(test, 1, False)

    >>> res = df_schema.validate(res).reset_index(drop=True)

    >>> df_expected = pd.DataFrame({
    ... "candid" : [6, 4, 0],
    ... "jd" : [6.0, 4.0, 0.0],
    ... "trajectory_id" : [3, 2, 1]
    ... })

    >>> assert_frame_equal(res, df_expected)

    >>> res = get_n_last_observations_from_trajectories(test, 1, True)

    >>> res = df_schema.validate(res).reset_index(drop=True)

    >>> df_expected = pd.DataFrame({
    ... "candid" : [3, 5, 9],
    ... "jd" : [3.0, 5.0, 9.0],
    ... "trajectory_id" : [1, 2, 3]
    ... })

    >>> assert_frame_equal(res, df_expected)

    >>> res = get_n_last_observations_from_trajectories(test, 2, True)

    >>> res = df_schema.validate(res).reset_index(drop=True)

    >>> df_expected = pd.DataFrame({
    ... "candid" : [2, 3, 4, 5, 8, 9],
    ... "jd" : [2.0, 3.0, 4.0, 5.0, 8.0, 9.0],
    ... "trajectory_id" : [1, 1, 2, 2, 3, 3]
    ... })

    >>> assert_frame_equal(res, df_expected)

    >>> res = get_n_last_observations_from_trajectories(test, 2, False)

    >>> res = df_schema.validate(res).reset_index(drop=True)

    >>> df_expected = pd.DataFrame({
    ... "candid" : [7, 6, 5, 4, 1, 0],
    ... "jd" : [7.0, 6.0, 5.0, 4.0, 1.0, 0.0],
    ... "trajectory_id" : [3, 3, 2, 2, 1, 1]
    ... })

    >>> assert_frame_equal(res, df_expected)
    """

    return (
        trajectories.sort_values(["jd"], ascending=ascending)
        .groupby(["trajectory_id"])
        .tail(n)
        .sort_values(["jd", "trajectory_id"], ascending=ascending)
    )


def intra_night_separation_association(night_alerts, separation_criterion):
    """
    Perform intra-night association based on the spearation between the alerts. The separation criterion was computed by a data analysis on the MPC object.

    Parameters
    ----------
    night_alerts : dataframe
        observation of one night (column ra and dec have to be present)
    separation_criterion : float
        the separation limit between the alerts to be associated, must be in arcsecond

    Returns
    -------
    left_assoc : dataframe
        Associations are a binary relation with left members and right members, return the left members of the associations
    right_assoc : dataframe
        return right members of the associations
    sep2d : list
        return the separation between the associated alerts

    Examples
    --------
    >>> test = pd.DataFrame({
    ... 'ra' : [100, 100.003, 100.007, 14, 14.003, 14.007],
    ... 'dec' : [8, 8.002, 8.005, 16, 15.998, 15.992],
    ... 'candid' : [1, 2, 3, 4, 5, 6],
    ... 'traj' : [1, 1, 1, 2, 2, 2]
    ... })

    >>> left, right, sep = intra_night_separation_association(test, 100*u.arcsecond)

    >>> left_expected = ts.intra_sep_left_expected
    >>> right_expected = ts.intra_sep_right_expected

    >>> assert_frame_equal(left.reset_index(drop=True), left_expected)
    >>> assert_frame_equal(right.reset_index(drop=True), right_expected)

    >>> sep_expected = [0.00358129, 0.00854695, 0.00358129, 0.00496889, 0.00854695,
    ...        0.00496889, 0.00350946, 0.01045366, 0.00350946, 0.00712637,
    ...        0.01045366, 0.00712637]

    >>> np.any(np.equal(np.around(sep.value, 8), sep_expected))
    True

    >>> len(np.where(sep == 0)[0])
    0
    """

    c1 = SkyCoord(night_alerts["ra"], night_alerts["dec"], unit=u.degree)

    c1_idx, c2_idx, sep2d, _ = search_around_sky(c1, c1, separation_criterion)

    # remove the associations with the same alerts. (sep == 0)
    nonzero_idx = np.where(sep2d > 0)[0]
    c2_idx = c2_idx[nonzero_idx]
    c1_idx = c1_idx[nonzero_idx]

    left_assoc = night_alerts.iloc[c1_idx]
    right_assoc = night_alerts.iloc[c2_idx]
    return left_assoc, right_assoc, sep2d[nonzero_idx]


def compute_diff_mag(left, right, fid, magnitude_criterion, normalized=False):
    """
    remove the associations based separation that not match the magnitude criterion. This magnitude criterion was computed on the MPC object.

    Parameters
    ----------
    left : dataframe
        left members of the association
    right : dataframe
        right member of the association
    fid : boolean Series
        filter the left and right associations with the boolean series based on the fid, take the alerts with the same fid or not.
    magnitude_criterion : float
        filter the association based on this magnitude criterion
    normalized : boolean
        if is True, normalized the magnitude difference by the jd difference.

    Returns
    -------
    left_assoc : dataframe
        return the left members of the associations filtered on the magnitude
    right_assoc : dataframe
        return the right members of the associations filtered on the magnitude

    Examples
    --------
    >>> test = pd.DataFrame({
    ... 'ra' : [100, 100.003, 100.007, 100.001, 100.003, 100.008],
    ... 'dec' : [12, 11.998, 11.994, 11.994, 11.998, 12.002],
    ... 'jd' : [1, 1, 1, 1, 1, 1],
    ... 'dcmag' : [17, 17.05, 17.09, 15, 15.07, 15.1],
    ... 'fid' : [1, 1, 1, 2, 2, 2],
    ... 'candid' : [1, 2, 3, 4, 5, 6],
    ... 'traj' : [1, 1, 1, 2, 2, 2]
    ... })

    >>> l, r, _ = intra_night_separation_association(test, 100*u.arcsecond)
    >>> same_fid = l['fid'].values == r['fid'].values
    >>> diff_fid = l['fid'].values != r['fid'].values

    >>> same_fid_left, same_fid_right = compute_diff_mag(l, r, same_fid, 0.1)
    >>> diff_fid_left, diff_fid_right = compute_diff_mag(l, r, diff_fid, 0.5)

    >>> same_fid_left_expected = ts.same_fid_left_expected1
    >>> same_fid_right_expected = ts.same_fid_right_expected1

    >>> assert_frame_equal(same_fid_left.reset_index(drop=True), same_fid_left_expected)
    >>> assert_frame_equal(same_fid_right.reset_index(drop=True), same_fid_right_expected)

    >>> diff_fid_left_expected = pd.DataFrame(columns = ['ra', 'dec', 'jd', 'dcmag', 'fid', 'candid', 'traj'])
    >>> diff_fid_right_expected = pd.DataFrame(columns = ['ra', 'dec', 'jd', 'dcmag', 'fid', 'candid', 'traj'])

    >>> assert_frame_equal(diff_fid_left, diff_fid_left_expected, check_index_type=False, check_dtype=False)
    >>> assert_frame_equal(diff_fid_right, diff_fid_right_expected, check_index_type=False, check_dtype=False)



    >>> test = pd.DataFrame({
    ... 'ra' : [100, 99, 99.8, 100.2, 100.7, 100.5],
    ... 'dec' : [12, 11.8, 11.2, 12.3, 11.7, 11.5],
    ... 'jd' : [1.05, 1.08, 1.09, 2.5, 2.6, 2.7],
    ... 'dcmag' : [15, 17, 16, 16.04, 17.03, 15.06],
    ... 'fid' : [1, 1, 1, 1, 1, 1],
    ... 'candid' : [1, 2, 3, 4, 5, 6],
    ... 'traj' : [1, 2, 3, 3, 2, 1]
    ... })

    >>> l, r, sep = intra_night_separation_association(test, 2*u.degree)
    >>> same_fid = l['fid'].values == r['fid'].values
    >>> diff_fid = l['fid'].values != r['fid'].values

    >>> same_fid_left, same_fid_right = compute_diff_mag(l, r, same_fid, 0.1, normalized=True)
    >>> diff_fid_left, diff_fid_right = compute_diff_mag(l, r, diff_fid, 0.5, normalized=True)

    >>> same_fid_left_expected = ts.same_fid_left_expected2
    >>> same_fid_right_expected = ts.same_fid_right_expected2

    >>> assert_frame_equal(same_fid_left.reset_index(drop=True), same_fid_left_expected)
    >>> assert_frame_equal(same_fid_right.reset_index(drop=True), same_fid_right_expected)

    >>> diff_fid_left_expected = pd.DataFrame(columns = ['ra', 'dec', 'jd', 'dcmag', 'fid', 'candid', 'traj'])
    >>> diff_fid_right_expected = pd.DataFrame(columns = ['ra', 'dec', 'jd', 'dcmag', 'fid', 'candid', 'traj'])

    >>> assert_frame_equal(diff_fid_left, diff_fid_left_expected, check_index_type=False, check_dtype=False)
    >>> assert_frame_equal(diff_fid_right, diff_fid_right_expected, check_index_type=False, check_dtype=False)
    """

    left_assoc = left[fid]
    right_assoc = right[fid]

    if normalized:
        diff_mag = np.abs(
            left_assoc["dcmag"].values - right_assoc["dcmag"].values
        ) / np.abs(left_assoc["jd"].values - right_assoc["jd"].values)

    else:
        diff_mag = np.abs(left_assoc["dcmag"].values - right_assoc["dcmag"].values)

    left_assoc = left_assoc[diff_mag <= magnitude_criterion]
    right_assoc = right_assoc[diff_mag <= magnitude_criterion]

    return left_assoc, right_assoc


def magnitude_association(
    left_assoc,
    right_assoc,
    mag_criterion_same_fid,
    mag_criterion_diff_fid,
    jd_normalization=False,
):
    """
    Perform magnitude based association twice, one for the alerts with the same fid and another for the alerts with a different fid.

    Parameters
    ----------
    left_assoc : dataframe
        left members of the associations (column dcmag and fid have to be present, jd must be present if normalize is set to True)
    right_assoc : dataframe
        right members of the associations (column dcmag and fid have to be present, jd must be present if normalize is set to True)
    mag_criterion_same_fid : float
        magnitude criterion between the alerts with the same filter id
    mag_criterion_diff_fid : float
        magnitude criterion between the alerts with a different filter id
    Returns
    -------
    left_assoc : dataframe
        return the left members of the associations filtered by magnitude
    right_assoc : dataframe
        return the right members of the associations filtered by magnitude

    Examples
    --------
    >>> left_test = pd.DataFrame({
    ... 'dcmag' : [17.03, 15, 17, 18, 18, 17, 17, 15],
    ... 'fid' : [1, 2, 1, 2, 1, 2, 1, 2]
    ... })

    >>> right_test = pd.DataFrame({
    ... 'dcmag' : [17, 15.09, 16, 19, 15, 12, 16, 14],
    ... 'fid' : [1, 2, 2, 1, 2, 1, 1, 2]
    ... })

    >>> l, r = magnitude_association(left_test, right_test, 0.1, 1)

    >>> l_expected = pd.DataFrame({
    ... "dcmag" : [17.03,15.00,17.00,18.00],
    ... "fid" : [1,2, 1, 2]
    ... })

    >>> r_expected = pd.DataFrame({
    ... "dcmag" : [17.00,15.09,16.00,19.00],
    ... "fid" : [1,2,2,1]
    ... })

    >>> assert_frame_equal(l, l_expected)
    >>> assert_frame_equal(r, r_expected)
    """

    same_fid = left_assoc["fid"].values == right_assoc["fid"].values
    diff_fid = left_assoc["fid"].values != right_assoc["fid"].values

    same_fid_left, same_fid_right = compute_diff_mag(
        left_assoc, right_assoc, same_fid, mag_criterion_same_fid, jd_normalization
    )
    diff_fid_left, diff_fid_right = compute_diff_mag(
        left_assoc, right_assoc, diff_fid, mag_criterion_diff_fid, jd_normalization
    )

    return (
        pd.concat([same_fid_left, diff_fid_left]),
        pd.concat([same_fid_right, diff_fid_right]),
    )


def compute_inter_night_metric(
    real_obs1, real_obs2, left_assoc, right_assoc, from_inter_night=False
):
    """
    Compute the inter night association metrics.
    Used only on test dataset where the column 'ssnamenr' are present and the real association are provided.

    Parameters
    ----------
    real_obs1 : dataframe
        left_members of the real associations
    real_obs2 : dataframe
        right_members of the real associations
    left_assoc : dataframe
        left members of the detected associations
    right_members : dataframe
        right members of the detected associations

    Returns
    -------
    metrics : dictionary
        dictionary containing the association metrics :
            precision : quality of the associations
            recall : capacity of the system to find all the real associations
            TP : true positive
            FP : false positive
            FN : false negative

    Examples
    --------
    >>> real_left = pd.DataFrame({
    ... 'candid': [0, 1],
    ... 'ssnamenr': [0, 1]
    ... })

    >>> real_right = pd.DataFrame({
    ... 'candid': [2, 3],
    ... 'ssnamenr': [0, 1]
    ... })

    >>> metrics = compute_inter_night_metric(real_left, real_right, real_left, real_right)
    >>> expected_metrics = {'precision': 100.0, 'recall': 100.0, 'True Positif': 2, 'False Positif': 0, 'False Negatif': 0, 'total real association': 2}
    >>> TestCase().assertDictEqual(expected_metrics, metrics)

    >>> detected_left = pd.DataFrame({
    ... 'candid': [0, 1],
    ... 'ssnamenr': [0, 1]
    ... })

    >>> detected_right = pd.DataFrame({
    ... 'candid': [3, 2],
    ... 'ssnamenr': [1, 0]
    ... })

    >>> metrics = compute_inter_night_metric(real_left, real_right, detected_left, detected_right)
    >>> expected_metrics = {'precision': 0.0, 'recall': 0.0, 'True Positif': 0, 'False Positif': 2, 'False Negatif': 2, 'total real association': 2}
    >>> TestCase().assertDictEqual(expected_metrics, metrics)

    >>> empty_assoc = pd.DataFrame(columns=['candid', 'ssnamenr'])
    >>> metrics = compute_inter_night_metric(real_left, real_right, empty_assoc, empty_assoc)
    >>> expected_metrics = {'precision': 0, 'recall': 0.0, 'True Positif': 0, 'False Positif': 0, 'False Negatif': 2, 'total real association': 2}
    >>> TestCase().assertDictEqual(expected_metrics, metrics)

    >>> real_left = pd.DataFrame({
    ... 'candid': [0, 1, 2, 3, 4],
    ... 'ssnamenr': [0, 1, 2, 3, 4]
    ... })

    >>> real_right = pd.DataFrame({
    ... 'candid': [5, 6, 7, 8, 9],
    ... 'ssnamenr': [0, 1, 2, 3, 4]
    ... })

    >>> detected_left = pd.DataFrame({
    ... 'candid': [0, 1, 2, 4],
    ... 'ssnamenr': [0, 1, 2, 4]
    ... })

    >>> detected_right = pd.DataFrame({
    ... 'candid': [5, 6, 8, 7],
    ... 'ssnamenr': [0, 1, 3, 2]
    ... })

    >>> metrics = compute_inter_night_metric(real_left, real_right, detected_left, detected_right)
    >>> expected_metrics = {'precision': 50.0, 'recall': 40.0, 'True Positif': 2, 'False Positif': 2, 'False Negatif': 3, 'total real association': 5}
    >>> TestCase().assertDictEqual(expected_metrics, metrics)

    >>> detected_left = pd.DataFrame({
    ... 'candid': [0, 2],
    ... 'ssnamenr': [0, 2]
    ... })

    >>> detected_right = pd.DataFrame({
    ... 'candid': [6, 8],
    ... 'ssnamenr': [1, 3]
    ... })

    >>> metrics = compute_inter_night_metric(real_left, real_right, detected_left, detected_right)
    >>> expected_metrics = {'precision': 0.0, 'recall': 0.0, 'True Positif': 0, 'False Positif': 2, 'False Negatif': 5, 'total real association': 5}
    >>> TestCase().assertDictEqual(expected_metrics, metrics)

    >>> detected_left = pd.DataFrame({
    ... 'candid': [0, 3],
    ... 'ssnamenr': [0, 3]
    ... })

    >>> detected_right = pd.DataFrame({
    ... 'candid': [5, 8],
    ... 'ssnamenr': [0, 3]
    ... })

    >>> metrics = compute_inter_night_metric(real_left, real_right, detected_left, detected_right)
    >>> expected_metrics = {'precision': 100.0, 'recall': 40.0, 'True Positif': 2, 'False Positif': 0, 'False Negatif': 3, 'total real association': 5}
    >>> TestCase().assertDictEqual(expected_metrics, metrics)

    >>> real_left = pd.DataFrame({
    ... 'candid': [0, 1, 2, 3, 4],
    ... 'ssnamenr': ["1", "1", "2", "3", "1"]
    ... })

    >>> real_right = pd.DataFrame({
    ... 'candid': [5, 6, 7, 8, 9, 10, 11],
    ... 'ssnamenr': ["1", "1", "2", "3", "4", "5", "1"]
    ... })

    >>> detected_left = pd.DataFrame({
    ... 'candid': [0, 1, 2],
    ... 'ssnamenr': ["0", "1", "2"]
    ... })

    >>> detected_right = pd.DataFrame({
    ... 'candid': [5, 6, 7],
    ... 'ssnamenr': ["1", "1", "2"]
    ... })

    >>> metrics = compute_inter_night_metric(real_left, real_right, detected_left, detected_right)
    >>> expected_metrics = {'precision': 66.66666666666666, 'recall': 40.0, 'True Positif': 2, 'False Positif': 1, 'False Negatif': 3, 'total real association': 5}
    >>> TestCase().assertDictEqual(expected_metrics, metrics)

    >>> real_left = pd.DataFrame({
    ... 'candid': [1520405434515015014, 1520216750115015007],
    ... 'ssnamenr': ["53317", "75539"]
    ... })

    >>> real_right = pd.DataFrame({
    ... 'candid': [1521202363215015009],
    ... 'ssnamenr': ["53317"]
    ... })

    >>> detected_left = pd.DataFrame({
    ... 'candid': [1520405434515015014],
    ... 'ssnamenr': ["53317"]
    ... })

    >>> detected_right = pd.DataFrame({
    ... 'candid': [1521202363215015009],
    ... 'ssnamenr': ["53317"]
    ... })

    >>> metrics = compute_inter_night_metric(real_left, real_right, detected_left, detected_right)
    >>> expected_metrics = {'precision': 100.0, 'recall': 100.0, 'True Positif': 1, 'False Positif': 0, 'False Negatif': 0, 'total real association': 1}
    >>> TestCase().assertDictEqual(expected_metrics, metrics)


    >>> real_left = pd.DataFrame({
    ... 'candid': [1520405434515015014, 1520216750115015007]
    ... })

    >>> real_right = pd.DataFrame({
    ... 'candid': [1521202363215015009]
    ... })

    >>> detected_left = pd.DataFrame({
    ... 'candid': [1520405434515015014],
    ... 'ssnamenr': ["53317"]
    ... })

    >>> detected_right = pd.DataFrame({
    ... 'candid': [1521202363215015009],
    ... 'ssnamenr': ["53317"]
    ... })

    >>> compute_inter_night_metric(real_left, real_right, detected_left, detected_right)
    {}
    """

    # fmt: off
    test_statement = ("ssnamenr" in real_obs1 and "ssnamenr" in real_obs2 and "ssnamenr" in left_assoc and "ssnamenr" in right_assoc)
    # fmt: on

    if test_statement:

        real_obs1 = (
            real_obs1[["candid", "ssnamenr"]]
            .sort_values(["ssnamenr"])
            .reset_index(drop=True)
        )
        real_obs2 = (
            real_obs2[["candid", "ssnamenr"]]
            .sort_values(["ssnamenr"])
            .reset_index(drop=True)
        )

        left_assoc = left_assoc[["candid", "ssnamenr"]].reset_index(drop=True)
        right_assoc = right_assoc[["candid", "ssnamenr"]].reset_index(drop=True)

        real_obs1 = real_obs1.rename(
            {g: g + "_left" for g in real_obs1.columns}, axis=1
        )
        real_obs2 = real_obs2.rename(
            {g: g + "_right" for g in real_obs2.columns}, axis=1
        )

        real_assoc = pd.concat([real_obs1, real_obs2], axis=1, join="inner")

        real_assoc = real_assoc[
            real_assoc["ssnamenr_left"] == real_assoc["ssnamenr_right"]
        ]

        left_assoc = left_assoc.rename(
            {g: "left_" + g for g in left_assoc.columns}, axis=1
        )
        right_assoc = right_assoc.rename(
            {g: "right_" + g for g in right_assoc.columns}, axis=1
        )

        detected_assoc = pd.concat([left_assoc, right_assoc], axis=1, join="inner")

        assoc_metrics = real_assoc.merge(
            detected_assoc,
            left_on=["candid_left", "ssnamenr_left", "candid_right", "ssnamenr_right"],
            right_on=["left_candid", "left_ssnamenr", "right_candid", "right_ssnamenr"],
            how="outer",
            indicator=True,
        )

        FP = len(assoc_metrics[assoc_metrics["_merge"] == "right_only"])
        TP = len(assoc_metrics[assoc_metrics["_merge"] == "both"])
        FN = len(assoc_metrics[assoc_metrics["_merge"] == "left_only"])

        try:
            precision = (TP / (FP + TP)) * 100
        except ZeroDivisionError:
            precision = 0

        try:
            recall = (TP / (FN + TP)) * 100
        except ZeroDivisionError:
            recall = 0

        return {
            "precision": precision,
            "recall": float(recall),
            "True Positif": TP,
            "False Positif": FP,
            "False Negatif": int(FN),
            "total real association": len(real_assoc),
        }
    else:
        return {}


def compute_intra_night_metrics(left_assoc, right_assoc, observations_with_real_labels):
    """
    computes performance metrics of the associations methods.
    Used only on test dataset where the column 'ssnamenr' are present and the real association are provided.

    Return 6 performance metrics :
        - precision : True positive divided by the sum of true positive and false positive. monitor the performance of the algorithm to do the good association.
        - recall : True positive divided by the sum of true positive and false negative. monitor the performance of the algorithm to find the association.
        - True positive
        - False positive
        - False Negative
        - total real association : the number of true association in the solar system MPC dataset.

    There is no true negative in the solar system dataset due to the only presence of true association.

    Parameters
    ----------
    left_assoc : dataframe
        left members of the predict associations
    right_members : dataframe
        right members of the predicts associations
    observations_with_real_labels : dataframe
        observations from known objects from MPC, real labels are ssnamenr

    Returns
    -------
    metrics_dict : dictionary
        dictionary where entries are performance metrics

    Examples
    --------
    >>> real_assoc = pd.DataFrame({
    ... 'jd' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    ... 'candid' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    ... 'ssnamenr' : [1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4]
    ... })

    >>> left_assoc = pd.DataFrame({
    ... 'jd' : [0, 2, 3, 5, 6, 7, 9],
    ... 'candid' : [0, 2, 3, 5, 6, 7, 9],
    ... 'ssnamenr' : [1, 2, 2, 3, 3, 3, 4]
    ... })

    >>> right_assoc = pd.DataFrame({
    ... 'jd' : [1, 3, 4, 6, 7, 8, 10],
    ... 'candid' : [1, 3, 4, 6, 7, 8, 10],
    ... 'ssnamenr' : [1, 2, 2, 3, 3, 3, 4]
    ... })

    >>> actual_dict = compute_intra_night_metrics(left_assoc, right_assoc, real_assoc)
    >>> expected_dict = {'precision': 100.0, 'recall': 100.0, 'True Positif': 7, 'False Positif': 0, 'False Negatif': 0, 'total real association': 7}

    >>> TestCase().assertDictEqual(expected_dict, actual_dict)

    >>> left_assoc = pd.DataFrame(columns=['jd', 'candid', 'ssnamenr'], dtype=np.float64)

    >>> right_assoc = pd.DataFrame(columns=['jd', 'candid', 'ssnamenr'], dtype=np.float64)

    >>> real_assoc = pd.DataFrame(columns=['jd', 'candid', 'ssnamenr'], dtype=np.float64)
    >>> actual_dict = compute_intra_night_metrics(left_assoc, right_assoc, real_assoc)
    >>> expected_dict = {'precision': 0, 'recall': 0.0, 'True Positif': 0, 'False Positif': 0, 'False Negatif': 0, 'total real association': 0}
    >>> TestCase().assertDictEqual(expected_dict, actual_dict)

    >>> real_assoc = pd.DataFrame({
    ... 'jd' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    ... 'candid' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    ... 'ssnamenr' : [1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4]
    ... })

    >>> left_assoc = pd.DataFrame({
    ... 'jd' : [0, 2, 5, 6, 9],
    ... 'candid' : [0, 2, 5, 6, 9],
    ... 'ssnamenr' : [1, 2, 3, 3, 4]
    ... })

    >>> right_assoc = pd.DataFrame({
    ... 'jd' : [1, 3, 6, 7, 10],
    ... 'candid' : [1, 3, 6, 7, 10],
    ... 'ssnamenr' : [1, 2, 3, 3, 4]
    ... })

    >>> actual_dict = compute_intra_night_metrics(left_assoc, right_assoc, real_assoc)
    >>> expected_dict = {'precision': 100.0, 'recall': 71.42857142857143, 'True Positif': 5, 'False Positif': 0, 'False Negatif': 2, 'total real association': 7}
    >>> TestCase().assertDictEqual(expected_dict, actual_dict)

    >>> left_assoc = pd.DataFrame({
    ... 'jd' : [0, 2, 5, 6, 7, 9],
    ... 'candid' : [0, 2, 5, 6, 7, 9],
    ... 'ssnamenr' : [1, 2, 3, 3, 3, 4]
    ... })

    >>> right_assoc = pd.DataFrame({
    ... 'jd' : [1, 3, 6, 7, 4, 10],
    ... 'candid' : [1, 3, 6, 7, 4, 10],
    ... 'ssnamenr' : [1, 2, 3, 3, 2, 4]
    ... })

    >>> actual_dict = compute_intra_night_metrics(left_assoc, right_assoc, real_assoc)
    >>> expected_dict = {'precision': 83.33333333333334, 'recall': 71.42857142857143, 'True Positif': 5, 'False Positif': 1, 'False Negatif': 2, 'total real association': 7}
    >>> TestCase().assertDictEqual(expected_dict, actual_dict)
    """

    gb_real_assoc = (
        observations_with_real_labels.sort_values(["jd"])
        .groupby(["ssnamenr"])
        .agg({"ssnamenr": list, "candid": list})
    )

    def split_assoc(x, left_ssnamenr, left_candid, right_ssnamenr, right_candid):
        ssnamenr = x["ssnamenr"]
        candid = x["candid"]

        for i in range(len(ssnamenr) - 1):
            left_ssnamenr.append(ssnamenr[i])
            left_candid.append(candid[i])

            right_ssnamenr.append(ssnamenr[i + 1])
            right_candid.append(candid[i + 1])

    left_candid = []
    left_ssnamenr = []
    right_candid = []
    right_ssnamenr = []

    gb_real_assoc.apply(
        split_assoc,
        axis=1,
        args=(left_ssnamenr, left_candid, right_ssnamenr, right_candid),
    )
    real_left = pd.DataFrame({"ssnamenr": left_ssnamenr, "candid": left_candid})

    real_right = pd.DataFrame({"ssnamenr": right_ssnamenr, "candid": right_candid})
    return compute_inter_night_metric(
        real_left, real_right, left_assoc, right_assoc, from_inter_night=True
    )


def restore_left_right(concat_l_r, nb_assoc_column):
    """
    By given a dataframe which is the results of a columns based concatenation of two dataframe, restore the two originals dataframes.
    Work only if the two dataframe have the same numbers of columns.

    Parameters
    ----------
    concat_l_r : dataframe
        dataframe which is the results of the two concatenate dataframes
    nb_assoc_column : integer
        the number of columns in the two dataframe

    Returns
    -------
    left_a : dataframe
        the left dataframe before the concatenation
    right_a : dataframe
        the right dataframe before the concatenation

    Examples
    --------
    >>> df1 = pd.DataFrame({
    ... "column1": [1, 2, 3],
    ... "column2": [4, 5, 6]
    ... })

    >>> df2 = pd.DataFrame({
    ... "column1": [7, 8, 9],
    ... "column2": [10, 11, 12]
    ... })

    >>> concat = pd.concat([df1, df2], axis=1, keys=['left', 'right'])

    >>> l, r = restore_left_right(concat, 2)

    >>> left_expected = pd.DataFrame({
    ... "column1" : [1, 2, 3],
    ... "column2" : [4, 5, 6]
    ... })

    >>> right_expected = pd.DataFrame({
    ... "column1" : [7, 8, 9],
    ... "column2" : [10, 11, 12]
    ... })

    >>> assert_frame_equal(l, left_expected)
    >>> assert_frame_equal(r, right_expected)
    """
    left_col = concat_l_r.columns.values[0:nb_assoc_column]
    right_col = concat_l_r.columns.values[nb_assoc_column:]

    left_a = concat_l_r[left_col]
    left_a.columns = left_a.columns.droplevel()

    right_a = concat_l_r[right_col]
    right_a.columns = right_a.columns.droplevel()

    return left_a, right_a


def removed_mirrored_association(left_assoc, right_assoc):
    """
    Remove the mirrored association (associations like (a, b) and (b, a)) that occurs in the intra-night associations.
    The column id used to detect mirrored association are candid.
    We keep the associations with the smallest jd in the left_assoc dataframe.

    Parameters
    ----------
    left_assoc : dataframe
        left members of the intra-night associations
    right_assoc : dataframe
        right members of the intra-night associations

    Returns
    -------
    drop_left : dataframe
        left members of the intra-night associations without mirrored associations
    drop_right : dataframe
        right members of the intra-night associations without mirrored associations

    Examples
    --------
    >>> test_1 = pd.DataFrame({
    ... "a" : [1, 2, 3, 4],
    ... "candid" : [10, 11, 12, 13],
    ... "jd" : [1, 2, 3, 4]
    ... })

    >>> test_2 = pd.DataFrame({
    ... "a" : [30, 31, 32, 33],
    ... "candid" : [11, 10, 15, 16],
    ... "jd" : [2, 1, 5, 6]
    ... })

    The mirror association is the candid 10 with the 11
    (10 is associated with 11 and 11 is associated with 10 if we look between test_1 and test_2)

    >>> tt_1, tt_2 = removed_mirrored_association(test_1, test_2)

    >>> tt_1_expected = pd.DataFrame({
    ... "a" : [1,3,4],
    ... "candid" : [10,12,13],
    ... "jd" : [1,3,4]
    ... })

    >>> tt_2_expected = pd.DataFrame({
    ... "a" : [30,32,33],
    ... "candid" : [11,15,16],
    ... "jd" : [2,5,6]
    ... })

    >>> assert_frame_equal(tt_1.reset_index(drop=True), tt_1_expected)
    >>> assert_frame_equal(tt_2.reset_index(drop=True), tt_2_expected)


    >>> df1 = pd.DataFrame({
    ... "candid" : [1522165813015015004, 1522207623015015004],
    ... "objectId": ['ZTF21aanxwfq', 'ZTF21aanyhht'],
    ... "jd" : [2459276.66581, 2459276.707627]
    ... })

    >>> df2 = pd.DataFrame({
    ... "candid" : [1522207623015015004, 1522165813015015004],
    ... "objectId": ['ZTF21aanyhht', 'ZTF21aanxwfq'],
    ... "jd" : [2459276.707627, 2459276.66581]
    ... })

    >>> dd1, dd2 = removed_mirrored_association(df1, df2)

    >>> dd1_expected = pd.DataFrame({
    ... "candid" : [1522165813015015004],
    ... "objectId" : ['ZTF21aanxwfq'],
    ... "jd" : [2459276.66581]
    ... })
    >>> dd2_expected = pd.DataFrame({
    ... "candid" : [1522207623015015004],
    ... "objectId" : ['ZTF21aanyhht'],
    ... "jd" : [2459276.707627]
    ... })

    >>> assert_frame_equal(dd1, dd1_expected)
    >>> assert_frame_equal(dd2, dd2_expected)
    """
    left_assoc = left_assoc.reset_index(drop=True)
    right_assoc = right_assoc.reset_index(drop=True)

    # concatanates the associations
    all_assoc = pd.concat(
        [left_assoc, right_assoc], axis=1, keys=["left", "right"]
    ).sort_values([("left", "jd")])

    # function used to detect the mirrored rows
    # taken from : https://stackoverflow.com/questions/58512147/how-to-removing-mirror-copy-rows-in-a-pandas-dataframe
    def key(x):
        """
        Examples
        --------
        >>> t_list = [10, 11]

        >>> key(t_list)
        frozenset({(11, 1), (10, 1)})
        """
        return frozenset(Counter(x).items())

    mask = (
        all_assoc[[("left", "candid"), ("right", "candid")]]
        .apply(key, axis=1)
        .duplicated()
    )

    # remove the mirrored duplicates by applying the mask to the dataframe
    drop_mirrored = all_assoc[~mask]

    left_a, right_a = restore_left_right(drop_mirrored, len(left_assoc.columns.values))

    return left_a, right_a


def removed_multiple_association(left_assoc, right_assoc):
    """
    Remove the multiple associations which can occurs during the intra_night associations.
    If we have three alerts (A, B and C) in the dataframe and the following associations : (A, B), (B, C) and (A, C) then this function remove
    the associations (A, C) to keep only (A, B) and (B, C).

    Warning : The jd between the triplets alerts must be differents, don't work if it is the case.
    Parameters
    ----------
    left_assoc : dataframe
        the left members of the associations
    right_assoc : dataframe
        the right members of the associations

    Returns
    -------
    left_assoc : dataframe
        left members without the multiple associations
    right_members : dataframe
        right members without the multiple associations

    Examples
    --------
    >>> left = pd.DataFrame({
    ... 'candid' : [0, 1, 0],
    ... 'jd' : [1, 2, 1]
    ... })

    >>> right = pd.DataFrame({
    ... 'candid' : [1, 2, 2],
    ... 'jd' : [2, 3, 3]
    ... })

    >>> l, r = removed_multiple_association(left, right)

    >>> l_expected = pd.DataFrame({
    ... "candid" : [1, 0],
    ... "jd" : [2, 1]
    ... })

    >>> r_expected = pd.DataFrame({
    ... "candid" : [2, 1],
    ... "jd" : [3, 2]
    ... })

    >>> assert_frame_equal(l.reset_index(drop=True), l_expected, check_dtype = False)
    >>> assert_frame_equal(r.reset_index(drop=True), r_expected, check_dtype = False)

    >>> left = pd.DataFrame({
    ... 'candid' : [0, 2, 0, 1],
    ... 'jd' : [1, 3, 1, 2]
    ... })

    >>> right = pd.DataFrame({
    ... 'candid' : [2, 3, 1, 2],
    ... 'jd' : [3, 4, 2, 3]
    ... })

    >>> l, r = removed_multiple_association(left, right)

    >>> l_expected = pd.DataFrame({
    ... "candid" : [2, 1, 0],
    ... "jd" : [3, 2, 1]
    ... })
    >>> r_expected = pd.DataFrame({
    ... "candid" : [3, 2, 1],
    ... "jd" : [4, 3, 2]
    ... })

    >>> assert_frame_equal(l.reset_index(drop=True), l_expected, check_dtype = False)
    >>> assert_frame_equal(r.reset_index(drop=True), r_expected, check_dtype = False)
    """
    # reset the index in order to recover the non multiple association
    left_assoc = left_assoc.reset_index(drop=True).reset_index()
    right_assoc = right_assoc.reset_index(drop=True).reset_index()

    # concat left and right members in order to keep the associations
    l_r_concat = pd.concat([left_assoc, right_assoc], axis=1, keys=["left", "right"])

    agg_dict = {col: list for col in l_r_concat.columns.values}

    # group by left candid to detect multiple assoc
    gb_concat = l_r_concat.groupby(by=[("left", "candid")]).agg(agg_dict)
    gb_concat["nb_multiple_assoc"] = gb_concat.apply(lambda x: len(x[0]), axis=1)

    # keep only the multiple association
    multiple_assoc = gb_concat[gb_concat[("nb_multiple_assoc", "")] > 1]
    multiple_assoc = multiple_assoc.drop(labels=[("nb_multiple_assoc", "")], axis=1)

    # sort right member by jd to keep the first association when we will used drop_duplicates
    explode_multiple_assoc = multiple_assoc.explode(
        list(multiple_assoc.columns.values)
    ).sort_values([("right", "jd")])

    # remove the rows from left and right assoc where the rows occurs in a multiple association
    drop_multiple_assoc = explode_multiple_assoc[("left", "index")].values
    left_assoc = left_assoc.drop(index=drop_multiple_assoc)
    right_assoc = right_assoc.drop(index=drop_multiple_assoc)

    # remove useless columns
    multiple_assoc = explode_multiple_assoc.drop(
        labels=[("left", "index"), ("right", "index")], axis=1
    )

    # drop the multiples associations and keep the first ones, as we have sort by jd on the right members, drop_duplicates remove the wrong associations
    single_assoc = explode_multiple_assoc.drop_duplicates(
        [("left", "candid")]
    ).reset_index(drop=True)

    # restore the initial left and right before the column based concatenation and concat the remain association with the old ones.
    single_left, single_right = restore_left_right(
        single_assoc, len(left_assoc.columns.values)
    )

    left_assoc = pd.concat([left_assoc, single_left]).drop(labels=["index"], axis=1)
    right_assoc = pd.concat([right_assoc, single_right]).drop(labels=["index"], axis=1)

    return left_assoc, right_assoc


def intra_night_association(
    night_observation,
    sep_criterion=145 * u.arcsecond,
    mag_criterion_same_fid=2.21,
    mag_criterion_diff_fid=1.75,
):
    """
    Perform intra_night association with separation and magnitude criterion
    Separation and magnitude are not normalised with the jd difference due to a too small difference of jd between the alerts.
    This creates too bigger values of separation and magnitude between the alerts.

    Parameters
    ----------
    night_observation : dataframe
        observations of the current night
    sep_criterion : float
        separation criterion between the alerts to be associated, must be in arcsecond
    mag_criterion_same_fid : float
        magnitude criterion between the alerts with the same filter id
    mag_criterion_diff_fid : float
        magnitude criterion between the alerts with a different filter id
    real_assoc : boolean
        if True, computes performance metrics of the associations based on real labels from ssnamenr columns in the night_observation parameters.
        night_observations must contains observations with ssnamenr from MPC objects.

    Returns
    -------
    left_assoc : dataframe
        left members of the associations
    right_assoc : dataframe
        right_members of the associations

    Examples
    --------
    >>> left, right = intra_night_association(ts.intra_night_test_traj, sep_criterion=145*u.arcsecond, mag_criterion_same_fid=2.21, mag_criterion_diff_fid=1.75)

    >>> assert_frame_equal(left.reset_index(drop=True), ts.intra_night_left, check_dtype = False)
    >>> assert_frame_equal(right.reset_index(drop=True), ts.intra_night_right, check_dtype = False)

    >>> left, right = intra_night_association(ts.intra_night_test_traj, sep_criterion=145*u.arcsecond, mag_criterion_same_fid=2.21, mag_criterion_diff_fid=1.75)

    >>> test_traj = pd.DataFrame({
    ... 'ra' : [106.305259, 106.141905, 169.860467],
    ... 'dec': [18.176682, 15.241181, 15.206360],
    ... 'dcmag': [0.066603, 0.018517, 0.038709],
    ... 'ssnamenr': [3866, 3051, 19743],
    ... 'fid': [1, 2, 1],
    ... 'candid': [1520166712915015010, 1520166711415015012, 1520220641415015001],
    ... 'jd': [2459274.666713, 2459274.666713, 2459274.7206481]
    ... })

    >>> left, right = intra_night_association(test_traj, sep_criterion=145*u.arcsecond, mag_criterion_same_fid=2.21, mag_criterion_diff_fid=1.75)

    >>> len(left)
    0
    >>> len(right)
    0
    """

    left_assoc, right_assoc, _ = intra_night_separation_association(
        night_observation, sep_criterion
    )

    left_assoc = left_assoc.reset_index(drop=True)
    right_assoc = right_assoc.reset_index(drop=True)

    mask_diff_jd = left_assoc["jd"] != right_assoc["jd"]
    left_assoc = left_assoc[mask_diff_jd]
    right_assoc = right_assoc[mask_diff_jd]

    left_assoc, right_assoc = magnitude_association(
        left_assoc, right_assoc, mag_criterion_same_fid, mag_criterion_diff_fid
    )

    if len(left_assoc) == 0:
        return pd.DataFrame(), pd.DataFrame()

    # remove mirrored associations
    left_assoc, right_assoc = removed_mirrored_association(left_assoc, right_assoc)

    # removed wrong multiple association
    left_assoc, right_assoc = removed_multiple_association(left_assoc, right_assoc)

    return left_assoc, right_assoc


def new_trajectory_id_assignation(left_assoc, right_assoc, last_traj_id):
    """
    Assign a trajectory id to all associations, perform a transitive assignation to all tracklets.

    Parameters
    ----------
    left_assoc : dataframe
        left members of the associations
    right_assoc : dataframe
        right members of the associations
    last_traj_id : integer
        the latest trajectory id assigned to all trajectory

    Returns
    -------
    trajectory_df : dataframe
        a single dataframe which are the concatanation of left and right and contains a new columns called 'trajectory_id'.
        This column allows to reconstruct the trajectory by groupby on this column.

    Examples
    --------
    >>> left_assoc1 = pd.DataFrame({
    ... "candid" : [1, 2, 3, 4]
    ... })

    >>> right_assoc1 = pd.DataFrame({
    ... "candid" : [5, 6, 7, 8]
    ... })

    >>> left_assoc2 = pd.DataFrame({
    ... "candid" : [1, 2, 3, 4, 5]
    ... })

    >>> right_assoc2 = pd.DataFrame({
    ... "candid" : [2, 3, 6, 5, 7]
    ... })

    >>> left_assoc3 = pd.DataFrame({
    ... "candid" : [1, 2, 3, 4, 5, 6, 7, 8]
    ... })

    >>> right_assoc3 = pd.DataFrame({
    ... "candid" : [9, 10, 11, 6, 4, 3, 2, 12]
    ... })

    >>> left_assoc4 = pd.DataFrame({
    ... "candid" : [1, 2, 3, 4]
    ... })

    >>> right_assoc4 = pd.DataFrame({
    ... "candid" : [5, 6, 5, 7]
    ... })

    >>> left_assoc5 = pd.DataFrame({
    ... "candid" : [10, 1, 2, 3, 2, 8, 6]
    ... })

    >>> right_assoc5 = pd.DataFrame({
    ... "candid" : [2, 5, 6, 7, 8, 9, 11]
    ... })

    >>> actual_traj_id = new_trajectory_id_assignation(left_assoc1, right_assoc1, 0)
    >>> expected_traj_id = pd.DataFrame({
    ... "candid" : [1, 2, 3, 4, 5, 6, 7, 8],
    ... "trajectory_id" : [0, 1, 2, 3, 0, 1, 2, 3]
    ... })

    >>> assert_frame_equal(actual_traj_id.reset_index(drop=True), expected_traj_id)

    >>> actual_traj_id = new_trajectory_id_assignation(left_assoc2, right_assoc2, 0)
    >>> expected_traj_id = pd.DataFrame({
    ... "candid" : [1, 2, 3, 4, 5, 6, 7],
    ... "trajectory_id" : [0, 0, 0, 3, 3, 0, 3]
    ... })

    >>> assert_frame_equal(actual_traj_id.reset_index(drop=True), expected_traj_id)

    >>> actual_traj_id = new_trajectory_id_assignation(left_assoc3, right_assoc3, 0)
    >>> expected_traj_id = pd.DataFrame({
    ... "candid" : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    ... "trajectory_id" : [0, 1, 3, 3, 3, 3, 1, 7, 0, 1, 3, 7]
    ... })

    >>> assert_frame_equal(actual_traj_id.reset_index(drop=True), expected_traj_id)

    >>> actual_traj_id = new_trajectory_id_assignation(left_assoc4, right_assoc4, 0)
    >>> expected_traj_id = pd.DataFrame({
    ... "candid" : [1, 2, 4, 5, 6, 7],
    ... "trajectory_id" : [0, 1, 2, 0, 1, 2]
    ... })
    >>> assert_frame_equal(actual_traj_id.reset_index(drop=True), expected_traj_id)

    >>> actual_traj_id = new_trajectory_id_assignation(left_assoc5, right_assoc5, 0)
    >>> expected_traj_id = pd.DataFrame({
    ... "candid" : [10, 1, 2, 3, 8, 6, 5, 7, 9, 11],
    ... "trajectory_id" : [0, 1, 0, 3, 4, 0, 1, 3, 4, 0]
    ... })
    >>> assert_frame_equal(actual_traj_id.reset_index(drop=True), expected_traj_id)
    """

    left_assoc = left_assoc
    right_assoc = right_assoc

    left_duplicates = left_assoc.duplicated()
    right_duplicates = right_assoc.duplicated()

    keep_non_duplicates = ~(left_duplicates | right_duplicates)
    left_assoc = left_assoc[keep_non_duplicates].reset_index(drop=True)
    right_assoc = right_assoc[keep_non_duplicates].reset_index(drop=True)

    nb_new_assoc = len(left_assoc)

    new_traj_id = np.arange(last_traj_id, last_traj_id + nb_new_assoc)

    left_assoc["trajectory_id"] = new_traj_id
    right_assoc["trajectory_id"] = new_traj_id

    for i, _ in right_assoc.iterrows():
        right_rows = right_assoc.iloc[i]
        left_rows = left_assoc.iloc[i]

        left_new_obs = left_assoc[left_assoc["candid"] == right_rows["candid"]]
        left_assoc.loc[left_new_obs.index.values, "trajectory_id"] = right_rows[
            "trajectory_id"
        ]
        right_assoc.loc[left_new_obs.index.values, "trajectory_id"] = right_rows[
            "trajectory_id"
        ]

        right_new_obs = right_assoc[right_assoc["candid"] == left_rows["candid"]]
        left_assoc.loc[right_new_obs.index.values, "trajectory_id"] = left_rows[
            "trajectory_id"
        ]
        right_assoc.loc[right_new_obs.index.values, "trajectory_id"] = left_rows[
            "trajectory_id"
        ]

    traj_df = pd.concat([left_assoc, right_assoc]).drop_duplicates(
        ["candid", "trajectory_id"]
    )

    return traj_df


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
