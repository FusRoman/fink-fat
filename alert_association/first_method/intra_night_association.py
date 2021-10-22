import astropy.units as u
from astropy.coordinates import SkyCoord
from numpy.lib.function_base import diff
import pandas as pd
import numpy as np
from collections import Counter



def intra_night_separation_association(night_alerts):
    c1 = SkyCoord(night_alerts['ra'], night_alerts['dec'], unit=u.degree)

    # 52 arcsecond come from our study of the intra-night separation between the alerts from the same nights. 52 take 90 percents of the objects.

    c2_idx, c1_idx, sep2d, _ = c1.search_around_sky(c1, 108.07*u.arcsecond)

    nonzero_idx = np.where(sep2d > 0)[0]
    c2_idx = c2_idx[nonzero_idx]
    c1_idx = c1_idx[nonzero_idx]

    left_assoc = night_alerts.iloc[c1_idx]
    right_assoc = night_alerts.iloc[c2_idx]
    return left_assoc, right_assoc, sep2d[nonzero_idx]


def compute_diff_mag_norm(left, right, fid, magnitude_criterion):
    left_assoc = left[fid]
    right_assoc = right[fid]

    diff_mag = np.abs(left_assoc['dcmag'].values - right_assoc['dcmag'].values)

    left_assoc = left_assoc[diff_mag <= magnitude_criterion]
    right_assoc = right_assoc[diff_mag <= magnitude_criterion]

    return left_assoc, right_assoc

def intra_night_magnitude_association(left_assoc, right_assoc):
    
    same_fid = left_assoc['fid'].values == right_assoc['fid'].values
    diff_fid = left_assoc['fid'].values != right_assoc['fid'].values

    same_fid_left, same_fid_right = compute_diff_mag_norm(left_assoc, right_assoc, same_fid, 2.21)
    diff_fid_left, diff_fid_right = compute_diff_mag_norm(left_assoc, right_assoc, diff_fid, 1.75)

    return pd.concat([same_fid_left, diff_fid_left]), pd.concat([same_fid_right, diff_fid_right])





if __name__ == "__main__":

    df_sso = pd.read_pickle("../data/month=03")

    all_night = np.unique(df_sso['nid'])

    df_one_night = df_sso[(df_sso['nid'] == 1520) & (df_sso['fink_class'] == 'Solar System MPC')]


    df_real_assoc = df_one_night.groupby(['ssnamenr']).count()
    
    nb_real_assoc = np.sum(df_real_assoc[df_real_assoc['ra'] > 1]['ra'])


    left_assoc, right_assoc, sep = intra_night_separation_association(df_one_night)

    left_assoc, right_assoc = intra_night_magnitude_association(left_assoc, right_assoc)

    nb_predict_assoc = len(right_assoc)


    precision_counter = Counter(left_assoc['ssnamenr'].values == right_assoc['ssnamenr'].values)

    print("precision: {}".format((1-(precision_counter[False] / precision_counter[True])) * 100))

    print("accuracy : {}".format((nb_predict_assoc / nb_real_assoc) * 100))
