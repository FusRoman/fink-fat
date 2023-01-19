import matplotlib.pyplot as plt
import numpy as np

def plot_nb_det_distribution(df):
    """
    Plot the distribution of the number of detection for each sso in Fink
    """
    nb_det = df.groupby("ssoCandId").count()["ra"]
    plt.hist(nb_det, 100, alpha=0.75, log=True)
    plt.xlabel('Number of detection')
    plt.ylabel('Number of SSO')
    plt.title('Number of detection of each sso in Fink')
    ax = plt.gca()
    plt.text(0.72, 0.8, 'min={},max={},median={}'.format(min(nb_det), max(nb_det), int(nb_det.median())), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    plt.grid(True)
    plt.show()

def plot_tw_distribution(df):
    """
    Plot the distribution of the observation window for each sso in Fink
    """
    tw = df.sort_values("jd").groupby("ssoCandId").agg(
    tw=("jd", lambda x: list(x)[-1] - list(x)[0])
        ).sort_values("tw")["tw"]
    plt.hist(tw, 100, alpha=0.75, log=True)
    plt.xlabel('Observation window (days)')
    plt.ylabel('Number of SSO')
    plt.title('Observation window of each sso in Fink')
    ax = plt.gca()
    plt.text(0.72, 0.8, 'min={:.2f},max={:.2f},median={:.2f}'.format(min(tw), max(tw), int(tw.median())), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    plt.grid(True)
    plt.show()


def plot_hist_and_cdf(data, range, percent_cdf = [0.8, 0.9], bins=200):
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
    _, axes = plt.subplots(3, 4, figsize=(50, 30))

    def plot_ax(ax1, ax2, plot_col, hist_title, hist_xlabel, hist_ylabel, cdf_title, cdf_xlabel, cdf_ylabel):
            ax1.set_title(hist_title, fontdict={"size": 30})
            ax1.set_xlabel(hist_xlabel, fontdict={"size": 30})
            ax1.set_ylabel(hist_ylabel, fontdict={"size": 30})
            ax1.set_yscale('log')
            ax1.hist(data[plot_col], bins=bins, range=range)

            ax2.set_title(cdf_title, fontdict={"size": 30})
            ax2.set_ylabel(cdf_ylabel, fontdict={"size": 30})
            ax2.set_xlabel(cdf_xlabel, fontdict={"size": 30})

            mean_diff_value, mean_diff_bins, _ = ax2.hist(data[plot_col], range=range, bins=bins, cumulative=True, density=True, histtype='step')

            x_interp = np.interp(percent_cdf, np.array(mean_diff_value, dtype='float64'), np.array(mean_diff_bins[:-1], dtype='float64'))
            ax2.scatter(x_interp, percent_cdf)

            for i , value in enumerate(zip(percent_cdf, x_interp)):
                txt = str(int(value[0]*100)) + "% = " + str(value[1].round(decimals=2))
                ax2.annotate(txt, (x_interp[i], percent_cdf[i]), fontsize=30)

            ax1.tick_params(axis='x', which='major', labelsize=30)
            ax1.tick_params(axis='y', which='major', labelsize=25)

            ax2.tick_params(axis='x', which='major', labelsize=30)
            ax2.tick_params(axis='y', which='major', labelsize=25)

    rms_label = ['rms_a', 'rms_e', 'rms_i', 'rms_long. node', 'rms_arg. peric', 'rms_mean anomaly']
    i = 0
    for ax1, ax2, ax3, ax4 in axes:

        plot_ax(ax1, ax2, rms_label[i], "Distribution {}".format(rms_label[i]), rms_label[i], "", "Cumulative {}".format(rms_label[i]), rms_label[i], "")
        plot_ax(ax3, ax4, rms_label[i+1], "Distribution {}".format(rms_label[i+1]), rms_label[i+1], "", "Cumulative {}".format(rms_label[i+1]), rms_label[i+1], "")
        i+=2
        
    plt.tight_layout()
    plt.show()

from collections import Counter
def compare_confirmed_and_candidates_rms(confirmed_orbit, confirmed_traj, candidates_orbit):

    orbit_with_error = confirmed_orbit[confirmed_orbit["rms_a"] != -1.0]
    traj_with_error = confirmed_traj[confirmed_traj["ssoCandId"].isin(orbit_with_error["ssoCandId"])]
    count_ssnamenr_with_error = traj_with_error[["ssoCandId", "ssnamenr"]].groupby("ssoCandId").agg(
        ssnamenr=("ssnamenr", list),
        count_ssnamenr=("ssnamenr", lambda x: len(Counter(x)))
    )
    pure_orbit_with_error = confirmed_orbit[confirmed_orbit["ssoCandId"].isin(count_ssnamenr_with_error[count_ssnamenr_with_error["count_ssnamenr"] == 1].reset_index()["ssoCandId"])]

    candidates_orbit_with_error = candidates_orbit[candidates_orbit["d:rms_a"] != -1.0]

    for conf_rms, cand_rms in zip(
    ['rms_a', 'rms_e', 'rms_i', 'rms_long. node', 'rms_arg. peric', 'rms_mean anomaly'], 
    ['d:rms_a', 'd:rms_e', 'd:rms_i', 'd:rms_long_node', 'd:rms_arg_peric', 'd:rms_mean_anomaly']
    ):
        fig = plt.figure(figsize=(20, 10))
        fig.suptitle(conf_rms, y=0.9)
        plt.hist(orbit_with_error[conf_rms], range=[0, 10], bins=200, log=True, alpha=0.6, label="confirmed reconstructed_orbit")
        plt.hist(pure_orbit_with_error[conf_rms],range=[0, 10], bins=200, log=True, alpha=0.75, label="pure confirmed reconstructed orbit")
        plt.hist(candidates_orbit_with_error[cand_rms],range=[0, 10], bins=200, log=True, alpha=0.75, label="candidate reconstructed orbit")
        plt.legend()
        plt.show()