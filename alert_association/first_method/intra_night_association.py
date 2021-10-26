import astropy.units as u
from astropy.coordinates import SkyCoord
import pandas as pd
import numpy as np
from collections import Counter
import time as t



def intra_night_separation_association(night_alerts, separation_criterion):
    """
    Perform intra-night association based on the spearation between the alerts. The separation criterion was computed by a data analysis on the MPC object.

    Parameters
    ----------
    night_alerts : dataframe
        observation of on night
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
    """

    c1 = SkyCoord(night_alerts['ra'], night_alerts['dec'], unit=u.degree)

    # 108.07 arcsecond come from our study of the intra-night separation between the alerts from the same nights. 52 take 99 percents of the objects.
    c2_idx, c1_idx, sep2d, _ = c1.search_around_sky(c1, separation_criterion)

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
    """
    left_assoc = left[fid]
    right_assoc = right[fid]

    if normalized:
        diff_mag = np.abs(left_assoc['dcmag'].values - right_assoc['dcmag'].values) / np.abs(left_assoc['jd'] - right_assoc['jd'])
    else:
        diff_mag = np.abs(left_assoc['dcmag'].values - right_assoc['dcmag'].values)

    left_assoc = left_assoc[diff_mag <= magnitude_criterion]
    right_assoc = right_assoc[diff_mag <= magnitude_criterion]

    return left_assoc, right_assoc

def magnitude_association(left_assoc, right_assoc, mag_criterion_same_fid, mag_criterion_diff_fid):
    """
    Perform magnitude based association twice, one for the alerts with the same fid and another for the alerts with a different fid. 

    Parameters
    ----------
    left_assoc : dataframe
        left members of the associations
    right_assoc : dataframe
        right members of the associations
    
    Returns
    -------
    left_assoc : dataframe
        return the left members of the associations filtered by magnitude
    right_assoc : dataframe
        return the right members of the associations filtered by magnitude
    """
    same_fid = left_assoc['fid'].values == right_assoc['fid'].values
    diff_fid = left_assoc['fid'].values != right_assoc['fid'].values

    same_fid_left, same_fid_right = compute_diff_mag(left_assoc, right_assoc, same_fid, mag_criterion_same_fid)
    diff_fid_left, diff_fid_right = compute_diff_mag(left_assoc, right_assoc, diff_fid, mag_criterion_diff_fid)

    return pd.concat([same_fid_left, diff_fid_left]), pd.concat([same_fid_right, diff_fid_right])


def compute_associations_metrics(left_assoc, right_assoc, observations_with_real_labels):
    """
    computes performance metrics of the associations methods

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
    metrics_dict : dictionnary
        dictionnary where entries are performance metrics    
    """
    gb_real_assoc = observations_with_real_labels.groupby(['ssnamenr']).count()

    nb_real_assoc = len(gb_real_assoc[gb_real_assoc['ra'] > 1]['ra'])

    nb_predict_assoc = len(right_assoc)

    precision_counter = Counter(left_assoc['ssnamenr'].values == right_assoc['ssnamenr'].values)

    # precision_counter[False] == number of false positif
    # precision_counter[True] == number of true positif
    precision = (precision_counter[True] / (precision_counter[True] + precision_counter[False])) * 100

    # max(0, (nb_real_assoc - nb_predict_assoc)) == number of false negatif if nb_predict_assoc < nb_real_assoc else 0 because no false negatif occurs. 
    FN = max(0, (nb_real_assoc - precision_counter[True]))
    recall = (precision_counter[True] / (precision_counter[True] + FN)) * 100

    accuracy = (1-(precision_counter[False] / (nb_predict_assoc + nb_real_assoc))) * 100

    return {"precision" : precision, "recall" : recall, "accuracy" : accuracy, 
            "True Positif" : precision_counter[True], "False Positif" : precision_counter[False], "False Negatif" : FN,
            "total real association" : nb_real_assoc}


def removed_mirrored_association(left_assoc, right_assoc):
    """
    Remove the mirrored association (associations like (a, b) and (b, a)) that occurs in the intra-night associations. 

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
    """
    left_assoc = left_assoc.reset_index(drop=True)
    right_assoc = right_assoc.reset_index(drop=True)

    old_columns = left_assoc.columns.values
    new_left_columns = ["left_" + el for el in old_columns]
    new_right_columns = ["right_" + el for el in old_columns]

    left_columns = {old_col : new_col for old_col, new_col in zip(old_columns, new_left_columns)}
    right_columns = {old_col : new_col for old_col, new_col in zip(old_columns, new_right_columns)}
    left_assoc = left_assoc.rename(left_columns, axis=1)
    right_assoc = right_assoc.rename(right_columns, axis=1)


    all_assoc = pd.concat([left_assoc, right_assoc], axis=1)

    

    mask = all_assoc[['left_candid','right_candid']].apply(lambda x: list(set(x)), axis=1).duplicated()
    drop_mirrored = all_assoc[~mask]

    restore_left_columns = {new_col : old_col for old_col, new_col in zip(old_columns, new_left_columns)}
    restore_right_columns = {new_col : old_col for old_col, new_col in zip(old_columns, new_right_columns)}
    
    drop_left = drop_mirrored[new_left_columns]
    drop_right = drop_mirrored[new_right_columns]
    drop_left = drop_left.rename(restore_left_columns, axis=1)
    drop_right = drop_right.rename(restore_right_columns, axis=1)
    return drop_left, drop_right

def intra_night_association(night_observation, sep_criterion=108.07*u.arcsecond, mag_criterion_same_fid=2.21, mag_criterion_diff_fid=1.75, compute_metrics=False):
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
    """
    left_assoc, right_assoc, _ = intra_night_separation_association(night_observation, sep_criterion)

    left_assoc, right_assoc = magnitude_association(left_assoc, right_assoc, mag_criterion_same_fid, mag_criterion_diff_fid)

    # remove mirrored associations
    left_assoc, right_assoc = removed_mirrored_association(left_assoc, right_assoc)
    
    if compute_metrics:
        metrics = compute_associations_metrics(left_assoc, right_assoc, night_observation)
        return left_assoc, right_assoc, metrics
    else:
        return left_assoc, right_assoc, {}


def new_trajectory_id_assignation(left_assoc, right_assoc, last_traj_id):
    nb_new_assoc = len(left_assoc)

    new_traj_id = np.arange(last_traj_id, last_traj_id + nb_new_assoc)
    left_assoc['trajectory_id'] = new_traj_id
    right_assoc['trajectory_id'] = new_traj_id

    return pd.concat([left_assoc, right_assoc])


if __name__ == "__main__":

    df_sso = pd.read_pickle("../../data/month=03")

    all_night = np.unique(df_sso['nid'])


    #t_before = t.time()
    for night_id in all_night:
        df_one_night = df_sso[(df_sso['nid'] == night_id) & (df_sso['fink_class'] == 'Solar System MPC')]

        t_before = t.time()
        left_assoc, right_assoc, perf_metrics = intra_night_association(df_one_night, compute_metrics=True)        


        new_traj_df = new_trajectory_id_assignation(left_assoc, right_assoc, 0)

        print("performance metrics :\n\t{}".format(perf_metrics))
        print("elapsed time : {}".format(t.time() - t_before))
        print()

    # test removed mirrored
    test_1 = pd.DataFrame({
        "a" : [1, 2, 3, 4],
        "candid" : [10, 11, 12, 13]
    })

    test_2 = pd.DataFrame({
        "a" : [30, 31, 32, 33],
        "candid" : [11, 10, 15, 16]
    })

    print(test_1)
    print(test_2)

    print()
    print()

    tt_1, tt_2 = removed_mirrored_association(test_1, test_2)

    print(tt_1)
    print(tt_2)