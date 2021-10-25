import astropy.units as u
from astropy.coordinates import SkyCoord
import pandas as pd
import numpy as np
from collections import Counter
import time as t

from intra_night_association import intra_night_association
from intra_night_association import new_trajectory_id_assignation
from intra_night_association import magnitude_association



def night_to_night_separation_association(old_observation, new_observation, separation_criterion):
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

    old_observations_coord = SkyCoord(old_observation['ra'], old_observation['dec'], unit=u.degree)
    new_observations_coord = SkyCoord(new_observation['ra'], new_observation['dec'], unit=u.degree)

    
    # 108.07 arcsecond come from our study of the intra-night separation between the alerts from the same nights. 52 take 99 percents of the objects.
    new_obs_idx, old_obs_idx, sep2d, _ = old_observations_coord.search_around_sky(new_observations_coord, separation_criterion)

    old_obs_assoc = old_observation.iloc[old_obs_idx]
    new_obs_assoc = new_observation.iloc[new_obs_idx]
    return old_obs_assoc, new_obs_assoc, sep2d


def get_last_observations_from_trajectories(trajectories):
    sort_jd = trajectories.sort_values(['jd'])

    last_occurence = sort_jd.drop_duplicates(['trajectory_id'], keep='last')

    return last_occurence


def night_to_night_association(last_observation_trajectory, old_observation, new_observation, sep_criterion=0.43*u.degree, mag_criterion_same_fid=1.36, mag_criterion_diff_fid=1.31):

    last_trajectory_id = np.max(last_observation_trajectory['trajectory_id'].values) + 1

    # trajectory association with the new tracklets
    new_left, new_right, _ = intra_night_association(new_observation)
    new_left = new_left.reset_index(drop=True)
    new_right = new_right.reset_index(drop=True)
    new_left['tmp_traj_id'] = new_right['tmp_traj_id'] = np.arange(0, len(new_left))

    traj_assoc, new_obs_assoc, _ = night_to_night_separation_association(last_observation_trajectory, new_left, sep_criterion)
    traj_assoc, new_obs_assoc = magnitude_association(traj_assoc, new_obs_assoc, mag_criterion_same_fid, mag_criterion_diff_fid)

    traj_assoc = traj_assoc.reset_index(drop=True)
    new_obs_assoc = new_obs_assoc.reset_index(drop=True)
    new_obs_assoc['trajectory_id'] = traj_assoc['trajectory_id']

    print(np.unique(traj_assoc.groupby(['trajectory_id']).count()['ra']))
    print()
    print(new_obs_assoc)

    #print(np.unique(pd.concat([last_observation_trajectory, new_obs_assoc]).groupby(['trajectory_id']).count()['ra']))

    return None, None

    track_assoc_left_columns = [el + "_x" for el in list(new_obs_assoc.columns)[:-2]] + ['trajectory_id']
    track_assoc_right_columns = [el + "_y" for el in list(new_obs_assoc.columns)[:-2]]

    track_assoc = new_obs_assoc.merge(new_right, on='tmp_traj_id')

    

    left_tracklets = track_assoc[track_assoc_left_columns]
    right_tracklets = track_assoc[track_assoc_right_columns]
    right_tracklets['trajectory_id'] = left_tracklets['trajectory_id']
    

    left_new_name = {k:v  for v, k in zip(list(new_obs_assoc.columns)[:-2], track_assoc_left_columns)}
    right_new_name = {k:v  for v, k in zip(list(new_obs_assoc.columns)[:-2], track_assoc_right_columns)}
    
    left_tracklets = left_tracklets.rename(left_new_name, axis=1)
    right_tracklets = right_tracklets.rename(right_new_name, axis=1)

    return None, None

    # trajectory association with new_observation
    traj_assoc, new_traj_obs, _ = night_to_night_separation_association(last_observation_trajectory, new_observation, sep_criterion)
    traj_assoc, new_traj_obs = magnitude_association(traj_assoc, new_traj_obs, mag_criterion_same_fid, mag_criterion_diff_fid)

    traj_assoc = traj_assoc.reset_index(drop=True)
    new_traj_obs = new_traj_obs.reset_index(drop=True)

    new_traj_obs['trajectory_id'] = traj_assoc['trajectory_id']
    new_observation = new_observation[~new_observation['candid'].isin(new_traj_obs['candid'])]

    old_obs_assoc, new_obs_assoc, _ = night_to_night_separation_association(old_observation, new_observation, sep_criterion)

    #old_assoc, new_assoc = intra_night_magnitude_association(old_obs_assoc, new_obs_assoc, mag_criterion_same_fid, mag_criterion_diff_fid)

    return traj_assoc, new_traj_obs



if __name__ == "__main__":
    df_sso = pd.read_pickle("../../data/month=03")

    df_sso = df_sso.drop_duplicates(['candid'])

    all_night = np.unique(df_sso['nid'])

    df_night1 = df_sso[(df_sso['nid'] == 1526) & (df_sso['fink_class'] == 'Solar System MPC')]
    df_night2 = df_sso[(df_sso['nid'] == 1527) & (df_sso['fink_class'] == 'Solar System MPC')]

    left, right, _ = intra_night_association(df_night1)
    traj_df = new_trajectory_id_assignation(left, right, 0)
    traj_df = traj_df.reset_index(drop=True)

    old_observation = df_night1[~df_night1['candid'].isin(traj_df['candid'])]

    
    old_assoc, new_assoc = night_to_night_association(traj_df, old_observation, df_night2)

    print("total alerts : {}, {}".format(len(df_night1), len(df_night2)))

    print(old_assoc)
    print()
    print(new_assoc)


