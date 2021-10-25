import astropy.units as u
from astropy.coordinates import SkyCoord
import pandas as pd
import numpy as np
from collections import Counter
import time as t

from intra_night_association import intra_night_magnitude_association



def night_to_night_separation_association(trajectoire_last_observations, old_observation, new_observations, separation_criterion=108.07*u.arcsecond):
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

    new_observations_coord = SkyCoord(new_observations['ra'], new_observations['dec'], unit=u.degree)




    # 108.07 arcsecond come from our study of the intra-night separation between the alerts from the same nights. 52 take 99 percents of the objects.
    c2_idx, c1_idx, sep2d, _ = c1.search_around_sky(c1, separation_criterion)

    nonzero_idx = np.where(sep2d > 0)[0]
    c2_idx = c2_idx[nonzero_idx]
    c1_idx = c1_idx[nonzero_idx]

    left_assoc = night_alerts.iloc[c1_idx]
    right_assoc = night_alerts.iloc[c2_idx]
    return left_assoc, right_assoc, sep2d[nonzero_idx]


def get_last_observations_from_trajectories(trajectories):

    


if __name__ == "__main__":
    df_sso = pd.read_pickle("../../data/month=03")

    all_night = np.unique(df_sso['nid'])

