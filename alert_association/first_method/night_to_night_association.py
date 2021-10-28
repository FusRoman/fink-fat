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

    
    # 108.07 arcsecond come from our study of the intra-night separation between the alerts from the same nights. 52 take 99 percents of the objects.
    new_obs_idx, old_obs_idx, sep2d, _ = old_observations_coord.search_around_sky(new_observations_coord, separation_criterion)

    old_obs_assoc = old_observation.iloc[old_obs_idx]
    new_obs_assoc = new_observation.iloc[new_obs_idx]
    return old_obs_assoc, new_obs_assoc, sep2d

def angle_three_point(a, b, c):
    ba = b - a
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)


def angle_df(x):
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

def remove_nan(x):
    return [el for el in x if ~np.isnan(el)]


def night_to_night_association(trajectory_df, old_observation, new_observation, sep_criterion=0.43*u.degree, mag_criterion_same_fid=1.36, mag_criterion_diff_fid=1.31):

    # get the last two observations for each trajectories
    two_last_observation_trajectory = get_n_last_observations_from_trajectories(trajectory_df, 2)

    # get the maximum trajectory_id and increment it to return the new trajectory_id baseline
    last_trajectory_id = np.max(trajectory_df.explode(['trajectory_id'])['trajectory_id'].values) + 1

    # get the last obeservations of the trajectories to perform the associations 
    last_traj_obs = two_last_observation_trajectory.groupby(['trajectory_id']).last().reset_index()

    # trajectory association with the new tracklets

    # intra-night association of the new observations
    new_left, new_right, _ = intra_night_association(new_observation)
    traj_next_night = new_trajectory_id_assignation(new_left, new_right, last_trajectory_id)

    # get the oldest extremity of the new tracklets to perform associations with the youngest observations in the trajectories 
    traj_extremity = get_n_last_observations_from_trajectories(traj_next_night, 1, False)    

    # night_to_night association between the last observations from the trajectories and the first tracklets observations of the next night
    traj_assoc, new_obs_assoc, _ = night_to_night_separation_association(last_traj_obs, traj_extremity, sep_criterion)
    traj_assoc, new_obs_assoc = magnitude_association(traj_assoc, new_obs_assoc, mag_criterion_same_fid, mag_criterion_diff_fid)
    traj_assoc, new_obs_assoc = removed_mirrored_association(traj_assoc, new_obs_assoc)
    
    if len(traj_assoc) == 0:
        return None, None
    else:

        traj_assoc = traj_assoc.reset_index(drop=True).reset_index()
        new_obs_assoc = new_obs_assoc.reset_index(drop=True).reset_index()
        new_obs_assoc = new_obs_assoc.rename({'trajectory_id' : 'tmp_traj'}, axis=1)
        new_obs_assoc['trajectory_id'] = traj_assoc['trajectory_id']
        
        two_last = two_last_observation_trajectory[two_last_observation_trajectory['trajectory_id'].isin(traj_assoc['trajectory_id'])]
        
        ttt = trajectory_df.explode(['trajectory_id'])
        print(ttt[ttt['trajectory_id'] == 3803])
        print()
        print(ttt[ttt['trajectory_id'] == 4942])
        print()

        two_last = two_last.groupby(['trajectory_id']).agg({
            "ra" : list,
            "dec" : list,
            "jd" : list,
            "candid" : lambda x : len(x)
        })
        
        print(two_last[two_last['candid'] == 1])

        prep_angle = two_last.merge(new_obs_assoc[['index', 'ra', 'dec', 'jd', 'trajectory_id']], on='trajectory_id')

        prep_angle['angle'] = prep_angle.apply(angle_df, axis=1)
        remain_assoc = prep_angle[prep_angle['angle'] <= 29.52]

        traj_assoc = traj_assoc.loc[remain_assoc['index'].values]
        new_obs_assoc = new_obs_assoc.loc[remain_assoc['index'].values]

        print("nb_assoc : {}".format(len(traj_assoc)))

        multi_assoc = traj_assoc.groupby(['trajectory_id']).count()
        multi_assoc = multi_assoc[multi_assoc['ra'] > 1]

        print("nb_multi_assoc : {}".format(len(multi_assoc)))

        if len(traj_assoc) > 0:
            print("prop : {}".format(len(multi_assoc) / len(traj_assoc)))

        return None, None

    print(new_obs_assoc.columns.values)


    new_traj_df = pd.concat([two_last_observation_trajectory, new_obs_assoc]).groupby(['trajectory_id']).agg(
        {
            "index" : lambda x : remove_nan(list(x)),
            "ra" : list,
            "dec" : list,
            "jd" : list,
            "tmp_traj" : lambda x : remove_nan(list(x)),
            "candid" : lambda x : len(x)
        }
    )

    new_traj_df['angle'] = new_traj_df.apply(angle_df, axis=1)
    
    print(new_traj_df[new_traj_df['candid'] >= 3])

    explode_angle_df = new_traj_df[new_traj_df['candid'] >= 3].explode(['index', 'angle', 'tmp_traj'])
    cone_search = explode_angle_df[explode_angle_df['angle'] <= 29.52].reset_index()
    print(cone_search)

    print("nb assoc : {}".format(len(cone_search)))

    # get alerts with multiple associations
    gb_traj = cone_search.groupby(['trajectory_id']).agg(
        {
            "index" : lambda x : len(x)
        }
    )

    #print(gb_traj)

    multiple_assoc = gb_traj[gb_traj['index'] > 1]

    if len(gb_traj) > 0:
        print("multiple assoc prop : {}".format(np.sum(multiple_assoc['index']) / np.sum(gb_traj['index']) * 100))
    else:
        print("nb assoc : {}".format(len(gb_traj)))

    return None, None

    all_multiple_candid = multiple_assoc.explode(['index', 'candid'])

    # create new trajectory index for the duplicates
    new_traj_id = np.arange(last_trajectory_id, last_trajectory_id + len(all_multiple_candid))

    # set candid column as new index to recover the right rows.
    traj_assoc.set_index(['candid'], inplace=True)

    # set a new trajectory id to all multiple associations
    # use candid index for traj_assoc 
    # becarefull, np.unique in the all_multiple_candid is required because loc get rows for each instance of candid so that creates duplicates on the results
    traj_assoc.loc[np.unique(all_multiple_candid['candid'].values), 'tmp_traj_id'] = new_traj_id
    new_obs_assoc.loc[np.unique(all_multiple_candid['index'].values), 'tmp_traj_id'] = new_traj_id
    traj_assoc = traj_assoc.reset_index()
    new_obs_assoc = new_obs_assoc.reset_index()

    # aggregate the new trajectory_id with the first trajectory_id
    gb_traj = traj_assoc.groupby(['trajectory_id']).agg(
        traj_size=("candid", lambda x : len(x)),
        tmp_traj=('tmp_traj_id', list)
    )

    gb_traj = gb_traj[gb_traj['traj_size'] > 1]

    # the first multiple associations take the original trajecotry_id 
    agg_traj_id = gb_traj['tmp_traj'].values
    for traj_id, new_traj_idx in zip(gb_traj.index.values, range(len(agg_traj_id))):
        agg_traj_id[new_traj_idx][0] = traj_id


    gb_traj['tmp_traj'] = agg_traj_id

    trajectory_df = trajectory_df.set_index(['trajectory_id'])
    trajectory_df.loc[trajectory_df.index.values, 'trajectory_id'] = gb_traj['tmp_traj']

    print(trajectory_df.loc[171])



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
    print(all_night)

    for i in range(len(all_night)-1):
        t_before = t.time()
        n1 = all_night[i]
        n2 = all_night[i+1]
        if n2 - n1 == 1:
            print(n1)
            print(n2)
            print()
            df_night1 = df_sso[(df_sso['nid'] == n1) & (df_sso['fink_class'] == 'Solar System MPC')]
            df_night2 = df_sso[(df_sso['nid'] == n2) & (df_sso['fink_class'] == 'Solar System MPC')]

            left, right, _ = intra_night_association(df_night1)
            traj_df = new_trajectory_id_assignation(left, right, 0)
            traj_df = traj_df.reset_index(drop=True)

            old_observation = df_night1[~df_night1['candid'].isin(traj_df['candid'])]

            old_assoc, new_assoc = night_to_night_association(traj_df, old_observation, df_night2)
        print("elapsed time: {}".format(t.time() - t_before))
        print()
        print()
        
        
        