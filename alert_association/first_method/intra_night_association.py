import astropy.units as u
from astropy.coordinates import SkyCoord
import pandas as pd
import numpy as np
from collections import Counter
import time as t




def get_n_last_observations_from_trajectories(trajectories, n, ascending=True):
    """
    Get n extremity observations from trajectories

    Parameters
    ----------
    trajectories : dataframe
        a dataframe with a trajectory_id column that identify trajectory observations.
    n : integer
        the number of extremity observations to return.
    ascending : boolean
        if set to True, return the most recent extremity observations, return the oldest ones otherwise, default to True.
    
    Returns
    -------
    last_trajectories_observations : dataframe
        the n last observations from the recorded trajectories
    """

    return trajectories.sort_values(['jd'], ascending=ascending).groupby(['trajectory_id']).tail(n).sort_values(['trajectory_id'])


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
    """
    left_col = concat_l_r.columns.values[0: nb_assoc_column]
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

    >>> tt_1, tt_2 = removed_mirrored_association(test_1, test_2)

    >>> tt_1
       a  candid  jd
    0  1      10   1
    2  3      12   3
    3  4      13   4
    >>> tt_2
        a  candid  jd
    0  30      11   2
    2  32      15   5
    3  33      16   6

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

    >>> df1
                    candid      objectId            jd
    0  1522165813015015004  ZTF21aanxwfq  2.459277e+06
    1  1522207623015015004  ZTF21aanyhht  2.459277e+06

    >>> df2
                    candid      objectId            jd
    0  1522207623015015004  ZTF21aanyhht  2.459277e+06
    1  1522165813015015004  ZTF21aanxwfq  2.459277e+06

    >>> dd1, dd2 = removed_mirrored_association(df1, df2)

    >>> dd1
                    candid      objectId            jd
    0  1522165813015015004  ZTF21aanxwfq  2.459277e+06
    >>> dd2
                    candid      objectId            jd
    0  1522207623015015004  ZTF21aanyhht  2.459277e+06
    """
    left_assoc = left_assoc.reset_index(drop=True)
    right_assoc = right_assoc.reset_index(drop=True)

    # concatanates the associations
    all_assoc = pd.concat([left_assoc, right_assoc], axis=1, keys=['left', 'right']).sort_values([('left', 'jd')])

    # function used to detect the mirrored rows
    # taken from : https://stackoverflow.com/questions/58512147/how-to-removing-mirror-copy-rows-in-a-pandas-dataframe
    def key(x):
        return frozenset(Counter(x).items())

    # create a set then a list of the left and right candid and detect the mirrored duplicates
    mask = all_assoc[[('left', 'candid'), ('right','candid')]].apply(key, axis=1).duplicated()

    # remove the mirrored duplicates by applying the mask to the dataframe
    drop_mirrored = all_assoc[~mask]

    left_a, right_a = restore_left_right(drop_mirrored, len(left_assoc.columns.values))

    return left_a, right_a

def removed_multiple_association(left_assoc, right_assoc):
    """
    Remove the multiple associations which can occurs during the intra_night associations.
    If we have three alerts (A, B and C) in the dataframe and the following associations : (A, B), (B, C) and (A, C) then this function remove
    the associations (A, C) to keep only (A, B) and (B, C). 

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
    """
    # reset the index in order to recover the non multiple association
    left_assoc = left_assoc.reset_index(drop=True).reset_index()
    right_assoc = right_assoc.reset_index(drop=True).reset_index()

    # concat left and right members in order to keep the associations
    l_r_concat = pd.concat([left_assoc, right_assoc], axis=1, keys=['left', 'right'])

    agg_dict = {col : list for col in l_r_concat.columns.values}

    # group by left candid to detect multiple assoc
    gb_concat = l_r_concat.groupby(by=[('left', 'candid')]).agg(agg_dict)
    gb_concat['nb_multiple_assoc'] = gb_concat.apply(lambda x : len(x[0]), axis=1)

    # keep only the multiple association
    multiple_assoc = gb_concat[gb_concat[('nb_multiple_assoc', '')] > 1]
    multiple_assoc = multiple_assoc.drop(labels=[('nb_multiple_assoc', '')], axis=1)

    # sort right member by jd to keep the first association when we will used drop_duplicates
    explode_multiple_assoc = multiple_assoc.explode(list(multiple_assoc.columns.values)).sort_values([('right', 'jd')])

    # remove the rows from left and right assoc where the rows occurs in a multiple association
    drop_multiple_assoc = explode_multiple_assoc[('left', 'index')].values
    left_assoc = left_assoc.drop(index=drop_multiple_assoc)
    right_assoc = right_assoc.drop(index=drop_multiple_assoc)

    # remove useless columns 
    multiple_assoc = explode_multiple_assoc.drop(labels=[('left', 'index'), ('right', 'index')], axis=1)

    # drop the multiples associations and keep the first ones, as we have sort by jd on the right members, drop_duplicates remove the wrong associations
    single_assoc = explode_multiple_assoc.drop_duplicates([('left', 'candid')]).reset_index(drop=True)

    # restore the initial left and right before the column based concatenation and concat the remain association with the old ones.
    single_left, single_right = restore_left_right(single_assoc, len(left_assoc.columns.values))

    left_assoc = pd.concat([left_assoc, single_left]).drop(labels=['index'], axis=1)
    right_assoc = pd.concat([right_assoc, single_right]).drop(labels=['index'], axis=1)

    return left_assoc, right_assoc

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

    if len(left_assoc) == 0:
        return pd.DataFrame(), pd.DataFrame(), {} 

    # remove mirrored associations
    left_assoc, right_assoc = removed_mirrored_association(left_assoc, right_assoc)
    
    # removed wrong multiple association
    left_assoc, right_assoc = removed_multiple_association(left_assoc, right_assoc)
    
    
    if compute_metrics:
        metrics = compute_associations_metrics(left_assoc, right_assoc, night_observation)
        return left_assoc, right_assoc, metrics
    else:
        return left_assoc, right_assoc, {}


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
    """

    left_assoc = left_assoc.reset_index(drop=True)
    right_assoc = right_assoc.reset_index(drop=True)

    nb_new_assoc = len(left_assoc)

    new_traj_id = np.arange(last_traj_id, last_traj_id + nb_new_assoc)

    left_assoc['trajectory_id'] = new_traj_id
    right_assoc['trajectory_id'] = new_traj_id



    for _, rows in right_assoc.iterrows():
        new_obs = left_assoc[left_assoc['candid'] == rows['candid']]
        left_assoc.loc[new_obs.index.values, 'trajectory_id'] = rows['trajectory_id']
        right_assoc.loc[new_obs.index.values, 'trajectory_id'] = rows['trajectory_id']

    traj_df = pd.concat([left_assoc, right_assoc]).drop_duplicates(['candid', 'trajectory_id'])
    return traj_df


if __name__ == "__main__":
    import doctest
    doctest.testmod()