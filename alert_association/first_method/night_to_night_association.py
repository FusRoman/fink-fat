from re import L
from astropy.coordinates.solar_system import get_body_barycentric
import astropy.units as u
from astropy.coordinates import SkyCoord
import pandas as pd
import numpy as np
from collections import Counter
import time as t

from pandas.core.indexes import multi

from intra_night_association import intra_night_association
from intra_night_association import new_trajectory_id_assignation
from intra_night_association import magnitude_association
from intra_night_association import removed_mirrored_association
from intra_night_association import get_n_last_observations_from_trajectories



def night_to_night_separation_association(old_observation, new_observation, separation_criterion):
    """
    Perform night-night association based on the separation between the alerts. The separation criterion was computed by a data analysis on the MPC object.

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
    """

    old_observations_coord = SkyCoord(old_observation['ra'], old_observation['dec'], unit=u.degree)
    new_observations_coord = SkyCoord(new_observation['ra'], new_observation['dec'], unit=u.degree)

    new_obs_idx, old_obs_idx, sep2d, _ = old_observations_coord.search_around_sky(new_observations_coord, separation_criterion)

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
        a rows from a dataframe with the three consecutives alerts to compute the angle
    
    Returns
    -------
    res_angle : float
        the angle between the three consecutives alerts normalized by the jd difference between the second point and the third point.
    """
    ra_x, dec_x, jd_x = x[1], x[2], x[3]

    ra_y, dec_y, jd_y = x[5], x[6], x[7]

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


def cone_search_association(two_last_observations, traj_assoc, new_obs_assoc, angle_criterion):
    """
    Filter the association based on a cone search. The idea is to remove the alerts triplets that have an angle greater than the angle_criterion.

    Parameters
    ----------
    two_last_observations : dataframe
        a dataframe that contains the two last observations for each trajectories
    traj_assoc : dataframe
        a dataframe which contains the trajectories extremity that are associated with the new observations.

    Returns
    -------
    traj_assoc : dataframe
        trajectories extremity associted with the new observation filtered by the angle cone search
    new_obs_assoc : dataframe
        new observations associated with the trajectories filtered by the angle cone search 
    """
    # reset the index of the associated members in order to recovered the right rows after the angle filters. 
    traj_assoc = traj_assoc.reset_index(drop=True).reset_index()
    new_obs_assoc = new_obs_assoc.reset_index(drop=True).reset_index()

    # rename the new trajectory_id column to another name in order to give the trajectory_id of the associated trajectories
    # and keep the new trajectory_id
    new_obs_assoc = new_obs_assoc.rename({'trajectory_id' : 'tmp_traj'}, axis=1)
    new_obs_assoc['trajectory_id'] = traj_assoc['trajectory_id']
    
    # get the two last observations of the associated trajectories in order to compute the cone search angle
    two_last = two_last_observations[two_last_observations['trajectory_id'].isin(traj_assoc['trajectory_id'])]

    # groupby the two last observations in order to prepare the merge with the new observations
    two_last = two_last.groupby(['trajectory_id']).agg({
        "ra" : list,
        "dec" : list,
        "jd" : list,
        "candid" : lambda x : len(x)
    })
    
    # merge the two last observation with the new observations to be associated
    prep_angle = two_last.merge(new_obs_assoc[['index', 'ra', 'dec', 'jd', 'trajectory_id']], on='trajectory_id')

    # compute the cone search angle
    prep_angle['angle'] = prep_angle.apply(angle_df, axis=1)
    # filter by the physical properties angle
    remain_assoc = prep_angle[prep_angle['angle'] <= angle_criterion]

    # keep only the alerts that match with the angle filter
    traj_assoc = traj_assoc.loc[remain_assoc['index'].values]
    new_obs_assoc = new_obs_assoc.loc[remain_assoc['index'].values]

    return traj_assoc.drop(['index'], axis=1), new_obs_assoc.drop(['index'], axis=1)

def night_to_night_observation_association(obs_set1, obs_set2, sep_criterion, mag_criterion_same_fid, mag_criterion_diff_fid):
    """
    Night to night associations between the observations

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
        the left members of the associations, extremity of the trajectories
    new_obs_assoc : dataframe
        new observations to add to the associated trajectories
    """
    # night_to_night association between the last observations from the trajectories and the first tracklets observations of the next night
    traj_assoc, new_obs_assoc, _ = night_to_night_separation_association(obs_set1, obs_set2, sep_criterion)
    traj_assoc, new_obs_assoc = magnitude_association(traj_assoc, new_obs_assoc, mag_criterion_same_fid, mag_criterion_diff_fid)
    traj_assoc, new_obs_assoc = removed_mirrored_association(traj_assoc, new_obs_assoc)
    return traj_assoc, new_obs_assoc

def trajectory_to_extremity_associations(two_last_observations, tracklets_extremity, sep_criterion, mag_criterion_same_fid, mag_criterion_diff_fid, angle_criterion): 
    """
    Associates the extremity of a trajectories with the new observations. The new observations can be a single observations or the extremity of a trajectories.

    Parameters
    ----------
    two_last_observation : dataframe
        the two last observations of all trajectories
    tracklets_extremity : dataframe
        the extremity of the tracklets of the new nights
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
    remain_traj : dataframe
        all non-associated trajectories
    """
    # get the last obeservations of the trajectories to perform the associations 
    last_traj_obs = two_last_observations.groupby(['trajectory_id']).last().reset_index()
    
    traj_assoc, new_obs_assoc = night_to_night_observation_association(last_traj_obs, tracklets_extremity, sep_criterion, mag_criterion_same_fid, mag_criterion_diff_fid)

    if len(traj_assoc) != 0:
        traj_assoc, new_obs_assoc = cone_search_association(two_last_observations, traj_assoc, new_obs_assoc, angle_criterion)
        remain_traj = last_traj_obs[~last_traj_obs['candid'].isin(traj_assoc['candid'])]
        return traj_assoc, new_obs_assoc, remain_traj
    else:
        return traj_assoc, new_obs_assoc, last_traj_obs



# The two functions below are used if we decide to manage the multiple associations that can appears during the process.
# They are not yet used.
def assign_new_trajectory_id_to_new_tracklets(new_obs, traj_next_night):
    """
    Propagate the trajectory id of the new obs to all observations of the corresponding trajectory

    Parameters
    ----------
    new_obs : dataframe
        the new observations that will be associated with a trajectory
    traj_next_night : dataframe
        dataframe that contains all the tracklets of the next night.

    Returns
    -------
    traj_next_night : dataframe
        all tracklets without the ones where new_obs appears.
    all_tracklets : dataframe
        all tracklets where new_obs appears
    """
    all_tracklets = []
    for _, rows in new_obs.iterrows():
        traj_id = rows['tmp_traj']

        # get all the alerts associated with this first observations 
        mask = traj_next_night.trajectory_id.apply(lambda x: any(i == traj_id for i in x))
        multi_traj = traj_next_night[mask.values]

        # change the trajectory_id of the associated alerts with the original trajectory_id of the associated trajectory
        multi_traj['trajectory_id'] = [[rows['trajectory_id']] for _ in range(len(multi_traj))]
        all_tracklets.append(multi_traj)
    
        # remove the associated tracklets
        traj_next_night = traj_next_night[~mask]

    return traj_next_night, pd.concat(all_tracklets)

def tracklets_id_management(traj_left, traj_right, traj_next_night, trajectory_df):
    """
    This functions perform the trajectory_id propagation between the already known trajectory and the new tracklets that will be associates.

    This function remove also the tracklets involved in an association in traj_next_night and update trajectory_df with all tracklets involved in an associations.

    Parameters
    ----------
    traj_left : dataframe
        the trajectories extremity
    traj_right : dataframe
        the tracklets extremity
    traj_next_night : dataframe
        the tracklets observations without the tracklets extremity
    trajectory_id : dataframe
        all the observation involved in the predict trajectories.

    Returns
    -------
    trajectory_df : dataframe
        the trajectory_df updated with the new tracklets associated with a new trajectory.
    traj_next_night : dataframe
        the remain tracklets that have not been associate with a trajectory.
    """
    traj_left = traj_left.reset_index(drop=True).reset_index()
    traj_right = traj_right.reset_index(drop=True)

    # detect the multiple associations with the left members (trajectories)
    multiple_assoc = traj_left.groupby(['trajectory_id']).agg({
        "index" : list,
        "candid" : lambda x : len(x)
    })

    # get the associations not involved in a multiple associations
    single_assoc = traj_right.loc[multiple_assoc[multiple_assoc['candid'] == 1].explode(['index'])['index'].values]

    # get the tracklets extremity involved in the multiple associations
    multiple_obs = traj_right.loc[multiple_assoc[multiple_assoc['candid'] > 1].explode(['index'])['index'].values]

    first_occur = first_occur_tracklets = other_occur = other_occurs_tracklets = pd.DataFrame()

    if len(multiple_obs) > 0:
        # get the first occurence in the multiple associations
        first_occur = multiple_obs[multiple_obs.duplicated(['trajectory_id'])]

        # get the others occurences
        other_occur = multiple_obs[~multiple_obs.duplicated(['trajectory_id'])]
        

        all_other_occur_tracklets = []
        # add the trajectory_id of the all other occurences to the list of trajectory_id of the associated trajectories
        for _, rows in other_occur.iterrows():
            traj_id = rows['trajectory_id']


            # get all rows of the associated trajectory 
            mask = trajectory_df.trajectory_id.apply(lambda x: any(i == traj_id for i in x))
            multi_traj = trajectory_df[mask]

            # get the trajectory id list of the associated trajectory
            multi_traj_id = multi_traj['trajectory_id'].values
            # duplicates the new trajectory_id which will be added to the trajectory id list
            new_traj_id = [[rows['tmp_traj']] for _ in range(len(multi_traj))]

            # concatenate the trajectory id list with the new trajectory id and add this new list to the trajectory_id columns of the associated trajectory
            trajectory_df.loc[multi_traj.index.values, 'trajectory_id'] = [el1 + el2 for el1, el2 in zip(multi_traj_id, new_traj_id)]

            
            # get all rows of the associated tracklets of the next night
            mask = traj_next_night.trajectory_id.apply(lambda x: any(i == rows['tmp_traj'] for i in x))
            next_night_tracklets = traj_next_night[mask]
            all_other_occur_tracklets.append(next_night_tracklets)

            # remove the associated tracklets
            traj_next_night = traj_next_night[~mask]

        other_occurs_tracklets = pd.concat(all_other_occur_tracklets)
        traj_next_night, first_occur_tracklets = assign_new_trajectory_id_to_new_tracklets(first_occur, traj_next_night)
        other_occur = other_occur.drop(['trajectory_id'], axis=1).rename({'tmp_traj' : 'trajectory_id'}, axis=1)
        first_occur = first_occur.drop(['tmp_traj'], axis=1)
        first_occur['trajectory_id'] = [[el] for el in first_occur['trajectory_id'].values]
        other_occur['trajectory_id'] = [[el] for el in other_occur['trajectory_id'].values]


    traj_next_night, single_assoc_tracklets = assign_new_trajectory_id_to_new_tracklets(single_assoc, traj_next_night)
    single_assoc['trajectory_id'] = [[el] for el in single_assoc['trajectory_id'].values]
    # add all the new tracklets in the trajectory dataframe with the right trajectory_id
    # return also all the tracklets without those added in the trajectory dataframe
    all_df_to_concat = [
        trajectory_df, 
        first_occur, 
        first_occur_tracklets, 
        other_occur, 
        other_occurs_tracklets, 
        single_assoc.drop(['tmp_traj'], axis=1),
        single_assoc_tracklets
        ]

    return pd.concat(all_df_to_concat), traj_next_night


def trajectory_associations(
    trajectory_df,
    traj_next_night,
    new_observation,
    next_nid,
    sep_criterion,
    mag_criterion_same_fid,
    mag_criterion_diff_fid,
    angle_criterion):


    # get the oldest extremity of the new tracklets to perform associations with the latest observations in the trajectories 
    tracklets_extremity = get_n_last_observations_from_trajectories(traj_next_night, 1, False)

    # get the last two observations for each trajectories
    two_last_observation_trajectory = get_n_last_observations_from_trajectories(trajectory_df, 2)

    trajectories_nid = np.sort(np.unique(two_last_observation_trajectory['nid']))
    
    for tr_nid in trajectories_nid[-5:]:
        two_last_observation_trajectory = two_last_observation_trajectory[two_last_observation_trajectory['nid'] == tr_nid]

        diff_night = next_nid - tr_nid
        norm_sep_crit = sep_criterion * diff_night
        norm_same_fid = mag_criterion_same_fid * diff_night
        norm_diff_fid = mag_criterion_diff_fid * diff_night
        norm_angle = angle_criterion * diff_night

        # trajectory association with the new tracklets
        traj_left, traj_extremity_associated, remain_traj = trajectory_to_extremity_associations(
            two_last_observation_trajectory,
            tracklets_extremity,
            norm_sep_crit,
            norm_same_fid,
            norm_diff_fid,
            norm_angle
            )

        # remove duplicates associations
        # do somethings with the duplicates later in the project
        traj_extremity_associated = traj_extremity_associated.drop_duplicates(['trajectory_id']).drop_duplicates(['candid'])

        associated_tracklets = []
        for _, rows in traj_extremity_associated.iterrows():
            # get all rows of the associated tracklets of the next night
            mask = traj_next_night.trajectory_id.apply(lambda x: any(i == rows['tmp_traj'] for i in x))
            next_night_tracklets = traj_next_night[mask]

            next_night_tracklets['trajectory_id'] = rows['trajectory_id']
            associated_tracklets.append(next_night_tracklets)

        
        associated_tracklets = pd.concat(associated_tracklets)
        traj_next_night = traj_next_night[~traj_next_night['candid'].isin(associated_tracklets['candid'])]

        # concatenation of trajectory_df with new tracklets doesn't work if we decide to manage the multiples associations.
        # need to keep track of multiple association with a list of trajectory_id.
        trajectory_df = pd.concat([trajectory_df, associated_tracklets])


        remaining_two_last = two_last_observation_trajectory[~two_last_observation_trajectory['trajectory_id'].isin(traj_left['trajectory_id'])]
        traj_assoc, obs_assoc, remain_traj = trajectory_to_extremity_associations(
            remaining_two_last,
            new_observation,
            norm_sep_crit,
            norm_same_fid,
            norm_diff_fid,
            norm_angle
            )
        
        # remove duplicates associations
        # do somethings with the duplicates later in the project
        obs_assoc = obs_assoc.drop_duplicates(['trajectory_id']).drop_duplicates(['candid'])

        new_observation = new_observation[~new_observation['candid'].isin(obs_assoc['candid'])]

        trajectory_df = pd.concat([trajectory_df, obs_assoc])

    return trajectory_df, traj_next_night, new_observation
    



def night_to_night_association(trajectory_df, old_observation, new_observation, sep_criterion=0.24*u.degree, mag_criterion_same_fid=0.18, mag_criterion_diff_fid=0.7, angle_criterion=8.8):
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
    trajectory_df : dataframe
        the updated trajectories with the new observations
    old_observation : dataframe
        the new set of old observations updated with the remaining non-associated new observations.
    """
    
    #  nid of the next night
    next_nid = new_observation['nid'].values[0]

    # get the maximum trajectory_id and increment it to return the new trajectory_id baseline
    last_trajectory_id = np.max(trajectory_df.explode(['trajectory_id'])['trajectory_id'].values) + 1

    # intra-night association of the new observations
    new_left, new_right, _ = intra_night_association(new_observation)
    new_left, new_right = new_left.reset_index(drop=True), new_right.reset_index(drop=True), 
    
    traj_next_night = new_trajectory_id_assignation(new_left, new_right, last_trajectory_id)

    # remove all the alerts that appears in the tracklets
    new_observation_not_associated = new_observation[~new_observation['candid'].isin(traj_next_night['candid'])]

    print(len(trajectory_df), len(traj_next_night), len(new_observation_not_associated))

    trajectory_df, traj_next_night, new_observation_not_associated = trajectory_associations(
        trajectory_df, 
        traj_next_night,
        new_observation_not_associated,
        next_nid, 
        sep_criterion, 
        mag_criterion_same_fid, 
        mag_criterion_diff_fid, 
        angle_criterion
        )

    print(len(trajectory_df), len(traj_next_night), len(new_observation_not_associated))
    
    """TODO
    do the associations between the old observations and the new tracklets, do the associations between the old and new observations
    from a night to night chaining point of view.    
    """


    exit()

    """precision_counter = Counter(traj_left['ssnamenr'].values == traj_extremity_associated['ssnamenr'].values)
    print("trajectory_to_tracklets associations precision : {}".format(precision_counter))"""

    # remove the tracklets extremity involved in an associations
    # reset index is important for the trajectory_id_management
    traj_next_night = traj_next_night[~traj_next_night['candid'].isin(traj_extremity_associated['candid'])].reset_index(drop=True)
    tracklets_extremity = tracklets_extremity[~tracklets_extremity['candid'].isin(traj_extremity_associated['candid'])].reset_index(drop=True)
    if len(traj_left) > 0:
        trajectory_df, tracklets_not_associated = tracklets_id_management(traj_left, traj_extremity_associated, traj_next_night, trajectory_df)
    else:
        tracklets_not_associated = traj_next_night

    two_last_not_associated = two_last_observation_trajectory[two_last_observation_trajectory['trajectory_id'].isin(remain_traj['trajectory_id'])]

    # trajectories associations with the new observations of the next night
    trajectories_left_assoc, new_obs_right, _ = trajectory_to_extremity_associations(
        two_last_not_associated,
        new_observation_not_associated,
        sep_criterion,
        mag_criterion_same_fid,
        mag_criterion_diff_fid,
        angle_criterion
    )

    # remove multiple associations
    new_obs_right = new_obs_right.drop_duplicates(['trajectory_id'], keep='last')

    """precision_counter = Counter(trajectories_left_assoc['ssnamenr'].values == new_obs_right['ssnamenr'].values)
    print("trajectories to new_observation association precision : {}".format(precision_counter))"""

    trajectory_df = pd.concat([trajectory_df, new_obs_right])

    new_observation_not_associated = new_observation_not_associated[~new_observation_not_associated['candid'].isin(new_obs_right['candid'])]

    all_old_night = np.unique(old_observation['nid'])
    next_nid = new_observation['nid'].values[0]
    for old_night in all_old_night:

        two_first_obs_tracklets = get_n_last_observations_from_trajectories(tracklets_not_associated, 2, ascending=False)

        current_old_night = old_observation[old_observation['nid'] == old_night]

        diff_night = next_nid - old_night

        norm_sep_crit = sep_criterion * diff_night
        norm_same_fid = mag_criterion_same_fid * diff_night
        norm_diff_fid = mag_criterion_diff_fid * diff_night
        norm_angle = angle_criterion * diff_night

        # tracklets associations with the old observations
        track_left, old_obs_right, _ = trajectory_to_extremity_associations(
            two_first_obs_tracklets,
            current_old_night,
            norm_sep_crit,
            norm_same_fid,
            norm_diff_fid,
            norm_angle)

        if len(old_obs_right) > 0:

            old_obs_right['trajectory_id'] = track_left['trajectory_id']

            # remove multiple associations
            old_obs_right = old_obs_right.drop_duplicates(['trajectory_id'], keep='last')
            
            old_obs_right['trajectory_id'] = [[el] for el in old_obs_right['trajectory_id'].values]
            # remove the associated observations in original old_observation dataframe
            old_observation = old_observation[~old_observation['candid'].isin(old_obs_right['candid'])]

            """precision_counter = Counter(track_left['ssnamenr'].values == old_obs_right['ssnamenr'].values)
            print("old observation to tracklets association precision : {}".format(precision_counter))"""

            remaining_tracklets = []
            associated_tracklets = []

            # remove all tracklets involved in the currents associations
            for traj_id in old_obs_right['trajectory_id'].values:
                mask = tracklets_not_associated.trajectory_id.apply(lambda x: any(i == traj_id for i in x))
                remaining_tracklets.append(tracklets_not_associated[~mask])
                associated_tracklets.append(tracklets_not_associated[mask])

            associated_tracklets = pd.concat(associated_tracklets)
            tracklets_not_associated = pd.concat(remaining_tracklets)

            # add the tracklets of the next night to the set of all trajectories
            all_new_tracklets = pd.concat([associated_tracklets, old_obs_right])
            trajectory_df = pd.concat([trajectory_df, all_new_tracklets])



        # old_observations to new observations associations
        old_obs_assoc, new_obs_assoc = night_to_night_observation_association(
            old_observation,
            new_observation_not_associated,
            sep_criterion,
            mag_criterion_same_fid,
            mag_criterion_diff_fid
            )


    # Warning, we keep all the multiple associations here, need to improve the alert associations here in order to reduce the memory consumption (maybe ??)

    # get the maximum trajectory_id and increment it to return the new trajectory_id baseline
    last_trajectory_id = np.max(trajectory_df.explode(['trajectory_id'])['trajectory_id'].values) + 1
    new_traj_id = np.arange(last_trajectory_id, last_trajectory_id + len(old_obs_assoc))
    
    # assign a new trajectory_id to new associations
    old_obs_assoc['trajectory_id'] = new_obs_assoc['trajectory_id'] = [[el] for el in new_traj_id]

    trajectory_df = pd.concat([trajectory_df, old_obs_assoc, new_obs_assoc])

    """precision_counter = Counter(old_obs_assoc['ssnamenr'].values == new_obs_assoc['ssnamenr'].values)
    print("old observation to new_observation precision : {}".format(precision_counter))"""

    old_observation = old_observation[~old_observation['candid'].isin(old_obs_assoc['candid'])]
    new_observation_not_associated = new_observation_not_associated[~new_observation_not_associated['candid'].isin(new_obs_assoc['candid'])]

    return trajectory_df, pd.concat([old_observation, new_observation])


def remove_oldest_observations(old_observation, max_night=5):
    oldest_night = np.sort(np.unique(old_observation['nid']))
    if len(old_observation) > max_night:
        return old_observation[old_observation['nid'].isin(oldest_night[1:])]
    else:
        return old_observation
    
if __name__ == "__main__":
    df_sso = pd.read_pickle("../../data/month=03")

    df_sso = df_sso.drop_duplicates(['candid'])

    all_night = np.unique(df_sso['nid'])
    print(all_night)

    n1 = all_night[0]
    n2 = all_night[1]
    df_night1 = df_sso[(df_sso['nid'] == n1) & (df_sso['fink_class'] == 'Solar System MPC')]
    df_night2 = df_sso[(df_sso['nid'] == n2) & (df_sso['fink_class'] == 'Solar System MPC')]
    left, right, _ = intra_night_association(df_night1)
    traj_df = new_trajectory_id_assignation(left, right, 0)
    traj_df = traj_df.reset_index(drop=True)

    old_observation = pd.concat([df_night1[~df_night1['candid'].isin(traj_df['candid'])], df_night2[~df_night2['candid'].isin(traj_df['candid'])]])

    for i in range(2, len(all_night)):
        t_before = t.time()
        new_night = all_night[i]

        df_next_night = df_sso[(df_sso['nid'] == new_night) & (df_sso['fink_class'] == 'Solar System MPC')]

        traj_df, old_observation = night_to_night_association(traj_df, old_observation, df_next_night)
        old_observation = remove_oldest_observations(old_observation)
        print("elapsed time: {}".format(t.time() - t_before))
        print()
        print()


        traj_gb = traj_df.explode(['trajectory_id']).groupby(['trajectory_id']).agg({
            'ssnamenr' : list,
            'candid' : lambda x :len(x)
        })

        print()
        print(traj_gb[traj_gb['candid'] > 3])
        """traj_gb = traj_df.explode(['trajectory_id']).groupby(['trajectory_id']).agg({
            'ra' : list,
            'dec' : list,
            'ssnamenr' : list,
            'candid' : lambda x : len(x)
        })

        print(traj_gb[traj_gb['candid'] > 3])"""
        
        

        

        
        
        
        
        
        