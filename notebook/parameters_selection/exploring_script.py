import pandas as pd
import matplotlib.pyplot as plt

from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np


def load_data(columns=None):
    """
    Load all the observations of solar system object in Fink 
    from the period between 1st of November 2019 and 16th of January 2022

    Parameters
    ----------
    None

    Return
    ------
    sso_data : Pandas Dataframe
        all sso alerts with the following columns
            - 'objectId', 'candid', 'ra', 'dec', 'jd', 'nid', 'fid', 'ssnamenr',
                'ssdistnr', 'magpsf', 'sigmapsf', 'magnr', 'sigmagnr', 'magzpsci',
                'isdiffpos', 'day', 'nb_detection', 'year', 'month'
    """
    return pd.read_parquet("sso_data", columns=columns)

def plot_nb_det_distribution(df):
    """
    Plot the distribution of the number of detection for each sso in Fink
    """
    unique_nb_detection = df.drop_duplicates("ssnamenr")["nb_detection"]
    plt.hist(unique_nb_detection, 100, alpha=0.75, log=True)
    plt.xlabel('Number of detection')
    plt.ylabel('Number of SSO')
    plt.title('Number of detection of each sso in Fink')
    ax = plt.gca()
    plt.text(0.72, 0.8, 'min={},max={},median={}'.format(min(unique_nb_detection), max(unique_nb_detection), int(unique_nb_detection.median())), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    plt.grid(True)
    plt.show()


def plot_tw_distribution(df):
    """
    Plot the distribution of the observation window for each sso in Fink
    """
    tw = df.groupby("ssnamenr").agg(
    tw=("jd", lambda x: list(x)[-1] - list(x)[0])
        ).sort_values("tw")["tw"]
    plt.hist(tw, 100, alpha=0.75, log=True)
    plt.xlabel('Observation window')
    plt.ylabel('Number of SSO')
    plt.title('Observation window of each sso in Fink')
    ax = plt.gca()
    plt.text(0.72, 0.8, 'min={},max={},median={}'.format(min(tw), max(tw), int(tw.median())), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    plt.grid(True)
    plt.show()


def sep_df(x):
    """
    Compute the speed between two observations of solar system object
    """
    ra, dec, jd = x["ra"], x["dec"], x["jd"]

    c1 = SkyCoord(ra, dec, unit = u.degree)

    diff_jd = np.diff(jd)

    sep = c1[0:-1].separation(c1[1:]).degree

    velocity = np.divide(sep, diff_jd)

    return velocity


def plot_hist_and_cdf(data, hist_range, hist_title, hist_xlabel, hist_ylabel, cdf_range, cdf_title, cdf_xlabel, cdf_ylabel, percent_cdf = [0.8, 0.9], bins=200):
    """
    Plot the distribution and the cumulative from data.

    Parameters
    ----------
    data: Series
    hist_range: list or None
    hist_title: String
    hist_xlabel: String
    hist_ylabel: String
    cdf_range: list or None
    cdf_title: String
    cdf_xlabel: String
    cdf_ylabel: String
    percent_cdf: list , default = [0.8, 0.9]
    bins: integer, default = 200

    Returns
    -------
    None
    """
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 10))

    ax1.set_title(hist_title, fontdict={"size": 30})
    ax1.set_xlabel(hist_xlabel, fontdict={"size": 30})
    ax1.set_ylabel(hist_ylabel, fontdict={"size": 30})
    ax1.set_yscale('log')
    ax1.hist(data, bins=bins, range=hist_range)

    ax2.set_title(cdf_title, fontdict={"size": 30})
    ax2.set_ylabel(cdf_ylabel, fontdict={"size": 30})
    ax2.set_xlabel(cdf_xlabel, fontdict={"size": 30})

    mean_diff_value, mean_diff_bins, _ = ax2.hist(data, range=cdf_range, bins=bins, cumulative=True, density=True, histtype='step')

    x_interp = np.interp(percent_cdf, mean_diff_value, mean_diff_bins[:-1])
    ax2.scatter(x_interp, percent_cdf)

    for i , value in enumerate(zip(percent_cdf, x_interp)):
        txt = str(int(value[0]*100)) + "% = " + str(value[1].round(decimals=2))
        ax2.annotate(txt, (x_interp[i], percent_cdf[i]), fontsize=30)

    ax1.tick_params(axis='x', which='major', labelsize=30)
    ax1.tick_params(axis='y', which='major', labelsize=25)

    ax2.tick_params(axis='x', which='major', labelsize=30)
    ax2.tick_params(axis='y', which='major', labelsize=25)
    plt.show()


def intra_sep_df(x):
    """
    Compute the sky separation from a set of equatorial coordinates
    """
    ra, dec= x["ra"], x["dec"]

    c1 = SkyCoord(ra, dec, unit = u.degree)

    sep = c1[0:-1].separation(c1[1:]).arcsecond

    return sep