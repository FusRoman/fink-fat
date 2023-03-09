import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from collections import Counter
import datetime
import warnings

import sbpy.data as sso_py
from astropy.time import Time
from astropy.coordinates import SkyCoord
from sbpy.data import Ephem
import astropy.units as ut


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
    return pd.read_parquet("../parameters_selection/sso_data", columns=columns)


def plot_nb_det_distribution(df):
    """
    Plot the distribution of the number of detection for each sso in Fink
    """
    nb_det = df.groupby("ssoCandId").count()["ra"]
    plt.hist(nb_det, 100, alpha=0.75, log=True)
    plt.xlabel("Number of detection")
    plt.ylabel("Number of SSO")
    plt.title("Number of detection of each sso in Fink")
    ax = plt.gca()
    plt.text(
        0.72,
        0.8,
        "min={},max={},median={}".format(
            min(nb_det), max(nb_det), int(nb_det.median())
        ),
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax.transAxes,
    )
    plt.grid(True)
    plt.show()


def plot_tw_distribution(df):
    """
    Plot the distribution of the observation window for each sso in Fink
    """
    tw = (
        df.sort_values("jd")
        .groupby("ssoCandId")
        .agg(tw=("jd", lambda x: list(x)[-1] - list(x)[0]))
        .sort_values("tw")["tw"]
    )
    plt.hist(tw, 100, alpha=0.75, log=True)
    plt.xlabel("Observation window (days)")
    plt.ylabel("Number of SSO")
    plt.title("Observation window of each sso in Fink")
    ax = plt.gca()
    plt.text(
        0.72,
        0.8,
        "min={:.2f},max={:.2f},median={:.2f}".format(
            min(tw), max(tw), int(tw.median())
        ),
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax.transAxes,
    )
    plt.grid(True)
    plt.show()


def plot_hist_and_cdf(data, range, percent_cdf=[0.8, 0.9], bins=200):
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

    def plot_ax(
        ax1,
        ax2,
        plot_col,
        hist_title,
        hist_xlabel,
        hist_ylabel,
        cdf_title,
        cdf_xlabel,
        cdf_ylabel,
    ):
        ax1.set_title(hist_title, fontdict={"size": 30})
        ax1.set_xlabel(hist_xlabel, fontdict={"size": 30})
        ax1.set_ylabel(hist_ylabel, fontdict={"size": 30})
        ax1.set_yscale("log")
        ax1.hist(data[plot_col], bins=bins, range=range)

        ax2.set_title(cdf_title, fontdict={"size": 30})
        ax2.set_ylabel(cdf_ylabel, fontdict={"size": 30})
        ax2.set_xlabel(cdf_xlabel, fontdict={"size": 30})

        mean_diff_value, mean_diff_bins, _ = ax2.hist(
            data[plot_col],
            range=range,
            bins=bins,
            cumulative=True,
            density=True,
            histtype="step",
        )

        x_interp = np.interp(
            percent_cdf,
            np.array(mean_diff_value, dtype="float64"),
            np.array(mean_diff_bins[:-1], dtype="float64"),
        )
        ax2.scatter(x_interp, percent_cdf)

        for i, value in enumerate(zip(percent_cdf, x_interp)):
            txt = str(int(value[0] * 100)) + "% = " + str(value[1].round(decimals=2))
            ax2.annotate(txt, (x_interp[i], percent_cdf[i]), fontsize=30)

        ax1.tick_params(axis="x", which="major", labelsize=30)
        ax1.tick_params(axis="y", which="major", labelsize=25)

        ax2.tick_params(axis="x", which="major", labelsize=30)
        ax2.tick_params(axis="y", which="major", labelsize=25)

    rms_label = [
        "rms_a",
        "rms_e",
        "rms_i",
        "rms_long. node",
        "rms_arg. peric",
        "rms_mean anomaly",
    ]
    i = 0
    for ax1, ax2, ax3, ax4 in axes:

        plot_ax(
            ax1,
            ax2,
            rms_label[i],
            "Distribution {}".format(rms_label[i]),
            rms_label[i],
            "",
            "Cumulative {}".format(rms_label[i]),
            rms_label[i],
            "",
        )
        plot_ax(
            ax3,
            ax4,
            rms_label[i + 1],
            "Distribution {}".format(rms_label[i + 1]),
            rms_label[i + 1],
            "",
            "Cumulative {}".format(rms_label[i + 1]),
            rms_label[i + 1],
            "",
        )
        i += 2

    plt.tight_layout()
    plt.show()


def compare_confirmed_and_candidates_rms(
    confirmed_orbit, confirmed_traj, candidates_orbit
):

    orbit_with_error = confirmed_orbit[confirmed_orbit["rms_a"] != -1.0]
    traj_with_error = confirmed_traj[
        confirmed_traj["ssoCandId"].isin(orbit_with_error["ssoCandId"])
    ]
    count_ssnamenr_with_error = (
        traj_with_error[["ssoCandId", "ssnamenr"]]
        .groupby("ssoCandId")
        .agg(
            ssnamenr=("ssnamenr", list),
            count_ssnamenr=("ssnamenr", lambda x: len(Counter(x))),
        )
    )
    pure_orbit_with_error = confirmed_orbit[
        confirmed_orbit["ssoCandId"].isin(
            count_ssnamenr_with_error[
                count_ssnamenr_with_error["count_ssnamenr"] == 1
            ].reset_index()["ssoCandId"]
        )
    ]

    candidates_orbit_with_error = candidates_orbit[candidates_orbit["d:rms_a"] != -1.0]

    for conf_rms, cand_rms in zip(
        [
            "rms_a",
            "rms_e",
            "rms_i",
            "rms_long. node",
            "rms_arg. peric",
            "rms_mean anomaly",
        ],
        [
            "d:rms_a",
            "d:rms_e",
            "d:rms_i",
            "d:rms_long_node",
            "d:rms_arg_peric",
            "d:rms_mean_anomaly",
        ],
    ):
        fig = plt.figure(figsize=(20, 10))
        fig.suptitle(conf_rms, y=0.9)

        _, bins = np.histogram(candidates_orbit_with_error[cand_rms], bins=200)
        logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))

        plt.hist(
            orbit_with_error[conf_rms],
            range=[0, 10],
            bins=logbins,
            log=True,
            alpha=0.6,
            label="confirmed reconstructed_orbit",
        )
        plt.hist(
            pure_orbit_with_error[conf_rms],
            range=[0, 10],
            bins=logbins,
            log=True,
            alpha=0.75,
            label="pure confirmed reconstructed orbit",
        )
        plt.hist(
            candidates_orbit_with_error[cand_rms],
            range=[0, 10],
            bins=logbins,
            log=True,
            alpha=0.75,
            label="candidate reconstructed orbit",
        )
        plt.legend()
        plt.show()


def get_unique_and_pure(reconstructed_orbit, reconstructed_trajectory):

    # With error (rms)
    orbit_with_error = reconstructed_orbit[reconstructed_orbit["rms_a"] != -1.0]
    traj_with_error = reconstructed_trajectory[
        reconstructed_trajectory["ssoCandId"].isin(orbit_with_error["ssoCandId"])
    ]

    count_ssnamenr = (
        reconstructed_trajectory[["ssoCandId", "ssnamenr"]]
        .groupby("ssoCandId")
        .agg(
            ssnamenr=("ssnamenr", list),
            count_ssnamenr=("ssnamenr", lambda x: len(Counter(x))),
        )
    )

    count_ssnamenr_with_error = (
        traj_with_error[["ssoCandId", "ssnamenr"]]
        .groupby("ssoCandId")
        .agg(
            ssnamenr=("ssnamenr", list),
            count_ssnamenr=("ssnamenr", lambda x: len(Counter(x))),
        )
    )

    pure_orbit, pure_with_error = (
        count_ssnamenr[count_ssnamenr["count_ssnamenr"] == 1],
        count_ssnamenr_with_error[count_ssnamenr_with_error["count_ssnamenr"] == 1],
    )

    unique_orbit, unique_with_error = (
        count_ssnamenr[count_ssnamenr["count_ssnamenr"] == 1]
        .explode("ssnamenr")
        .drop_duplicates("ssnamenr"),
        count_ssnamenr_with_error[count_ssnamenr_with_error["count_ssnamenr"] == 1]
        .explode("ssnamenr")
        .drop_duplicates("ssnamenr"),
    )

    return (
        orbit_with_error,
        traj_with_error,
        pure_orbit,
        pure_with_error,
        unique_orbit,
        unique_with_error,
    )


def results(
    reconstructed_orbit,
    reconstructed_trajectory,
    input_data,
    orbfit_limit_point=6,
    tw=15,
):
    # Confirmed SSO in Input
    nb_input = len(input_data["ssnamenr"].unique())

    is_detectable = (
        input_data.sort_values("jd")
        .groupby("ssnamenr")
        .agg(nb_det=("ra", len), is_in_tw=("jd", lambda x: np.all(np.diff(x) <= tw)))
    )

    # Detectable
    nb_detectable = len(
        is_detectable[
            (is_detectable["nb_det"] >= orbfit_limit_point)
            & (is_detectable["is_in_tw"])
        ]
    )

    # Reconstructed
    nb_reconstruct = len(reconstructed_orbit)

    (
        orbit_with_error,
        traj_with_error,
        pure_orbit,
        pure_with_error,
        unique_orbit,
        unique_with_error,
    ) = get_unique_and_pure(reconstructed_orbit, reconstructed_trajectory)

    # with error
    nb_reconstruct_with_error = len(orbit_with_error)

    # Pure
    nb_pure, nb_pure_with_error = len(pure_orbit), len(pure_with_error)

    # Unique
    nb_unique, nb_unique_with_error = len(unique_orbit), len(unique_with_error)

    # Purity
    purity, purity_with_error = (
        ((nb_pure / nb_reconstruct) * 100),
        ((nb_pure_with_error / nb_reconstruct_with_error) * 100),
    )

    # Efficiency
    efficiency, efficiency_with_error = (
        ((nb_unique / nb_detectable) * 100),
        ((nb_unique_with_error / nb_detectable) * 100),
    )

    return (
        """
|                     | Fink_FAT |                 |
|---------------------|----------|-----------------|
|                     | All      | Only with error |
| Confirmed SSO input | {}       | X               |
| Detectable          | {}       | X               |
| Reconstructed orbit | {}    | {}           |
| - Pure              | {}    | {}           |
| - Unique            | {}    | {}           |
| Purity              | {:.1f} %   | {:.1f} %          |
| Efficiency          | {:.1f} %   | {:.1f} %          |
        """.format(
            nb_input,
            nb_detectable,
            nb_reconstruct,
            nb_reconstruct_with_error,
            nb_pure,
            nb_pure_with_error,
            nb_unique,
            nb_unique_with_error,
            purity,
            purity_with_error,
            efficiency,
            efficiency_with_error,
        ),
        purity,
        purity_with_error,
        efficiency,
        efficiency_with_error,
    )


def plot_rms_distribution(
    reconstructed_orbit, reconstructed_trajectory, mops_orbit=None, mops_traj=None
):

    orbit_with_error, _, _, pure_with_error, _, _ = get_unique_and_pure(
        reconstructed_orbit, reconstructed_trajectory
    )

    if mops_orbit is not None and mops_traj is not None:
        orbit_mops_with_error, _, _, pure_mops_with_error, _, _ = get_unique_and_pure(
            mops_orbit, mops_traj
        )

    for rms in [
        "rms_a",
        "rms_e",
        "rms_i",
        "rms_long. node",
        "rms_arg. peric",
        "rms_mean anomaly",
    ]:
        fig = plt.figure(figsize=(20, 10))
        fig.suptitle(rms, y=0.9)

        _, bins = np.histogram(orbit_with_error[rms], bins=200)
        logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))

        plt.hist(
            orbit_with_error[rms],
            bins=logbins,
            log=True,
            alpha=0.6,
            label="confirmed reconstructed_orbit",
        )

        pure_with_error = reconstructed_orbit[
            reconstructed_orbit["ssoCandId"].isin(
                pure_with_error.reset_index()["ssoCandId"]
            )
        ]
        plt.hist(
            pure_with_error[rms],
            bins=logbins,
            log=True,
            alpha=0.75,
            label="pure confirmed reconstructed orbit",
        )

        if mops_orbit is not None and mops_traj is not None:
            plt.hist(
                orbit_mops_with_error[rms],
                bins=logbins,
                log=True,
                alpha=0.6,
                label="MOPS: confirmed reconstructed_orbit",
            )
            pure_mops_with_error = mops_orbit[
                mops_orbit["ssoCandId"].isin(
                    pure_mops_with_error.reset_index()["ssoCandId"]
                )
            ]
            plt.hist(
                pure_mops_with_error[rms],
                bins=logbins,
                log=True,
                alpha=0.75,
                label="MOPS: pure confirmed reconstructed orbit",
            )

        plt.xscale("log")
        plt.legend()
        plt.show()


def mpc_crossmatch(mpc_orb, ssnamenr):
    explode_other = mpc_orb.explode("Other_desigs")
    # explode_other = explode_other[explode_other["Principal_desig"] != explode_other["Other_desigs"]].reset_index(drop=True)

    t1 = explode_other["Number"].str[1:-1].isin(ssnamenr)
    t2 = explode_other["Principal_desig"].str.replace(" ", "").isin(ssnamenr)
    t3 = explode_other["Name"].str.replace(" ", "").isin(ssnamenr)
    t4 = explode_other["Other_desigs"].str.replace(" ", "").isin(ssnamenr)

    reconstructed_mpc = explode_other[t1 | t2 | t3 | t4].drop_duplicates(
        ["Number", "Principal_desig"]
    )

    a = ~ssnamenr.isin(explode_other["Number"].str[1:-1])
    b = ~ssnamenr.isin(explode_other["Principal_desig"].str.replace(" ", ""))
    c = ~ssnamenr.isin(explode_other["Name"].str.replace(" ", ""))
    d = ~ssnamenr.isin(explode_other["Other_desigs"].str.replace(" ", ""))

    not_in_mpc = ssnamenr[a & b & c & d]

    return reconstructed_mpc, not_in_mpc


def display_mpc_reconstruction(reconstructed_mpc, reconstructed_mops, input_mpc):

    reconstruc_orbit_type = reconstructed_mpc.groupby("Orbit_type").count()[
        "Principal_desig"
    ]
    initial_mpc = input_mpc.groupby("Orbit_type").count()["Principal_desig"]
    percent_mpc_reconstr = (reconstruc_orbit_type / initial_mpc) * 100

    reconstruc_mops_orbit_type = reconstructed_mops.groupby("Orbit_type").count()[
        "Principal_desig"
    ]
    percent_mpc_reconstr_mops = (reconstruc_mops_orbit_type / initial_mpc) * 100

    table = """
|  | Initial orbit Distribution | Fink_FAT | MOPS |
|--|----------------------------|----------|------|
"""

    for i in initial_mpc.items():
        cur_orbit = i[0]
        nb_init = initial_mpc[cur_orbit]

        if cur_orbit not in reconstruc_orbit_type:
            nb_reconstr = 0
        else:
            nb_reconstr = reconstruc_orbit_type[cur_orbit]
        if cur_orbit not in percent_mpc_reconstr:
            percent = 0
        else:
            percent = percent_mpc_reconstr[cur_orbit]

        if cur_orbit not in reconstruc_mops_orbit_type:
            nb_reconstr_mops = 0
        else:
            nb_reconstr_mops = reconstruc_mops_orbit_type[cur_orbit]
        if cur_orbit not in percent_mpc_reconstr_mops:
            percent_mops = 0
        else:
            percent_mops = percent_mpc_reconstr_mops[cur_orbit]

        table += "| {} | {} | {} ({:.2f} %) | {} ({:.2f} %) |\n".format(
            cur_orbit, nb_init, nb_reconstr, percent, nb_reconstr_mops, percent_mops
        )

    return table


def merge_reconstruct_and_mpc(mpc_in_fink, reconstruct_orbit):

    mpc_in_fink["Number"] = mpc_in_fink["Number"].str[1:-1]
    mpc_in_fink["Principal_desig"] = mpc_in_fink["Principal_desig"].str.replace(" ", "")

    mpc_in_fink_explode = mpc_in_fink.explode("Other_desigs")
    mpc_in_fink_explode["Other_desigs"] = mpc_in_fink_explode[
        "Other_desigs"
    ].str.replace(" ", "")

    a = reconstruct_orbit.merge(mpc_in_fink, left_on="ssnamenr", right_on="Number")
    b = reconstruct_orbit.merge(
        mpc_in_fink, left_on="ssnamenr", right_on="Principal_desig"
    )
    c = reconstruct_orbit.merge(
        mpc_in_fink, left_on="ssnamenr", right_on="Other_desigs"
    )

    merge_mpc_orbfit = pd.concat([a, b, c])

    return merge_mpc_orbfit


def plot_orbfit_diff_hist(diff_data, orb_param):
    _ = plt.figure(figsize=(20, 10))

    data = diff_data[orb_param]

    _, bins = np.histogram(data, bins=200)
    tmp_logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))

    logbins = tmp_logbins

    plt.hist(data, bins=logbins, log=True, alpha=0.6)

    # plt.legend(prop={'size': 15})
    plt.title(
        "Distribution of the difference between the estimated orbit parameters with ORBFIT and the MPC orbit parameters",
        fontdict={"size": 20},
    )
    plt.ylabel("Number of orbit (log)", fontdict={"size": 20})
    plt.xlabel(
        "Difference between orbit and MPC (log of the difference in %)",
        fontdict={"size": 20},
    )
    ax = plt.gca()
    ax.tick_params(axis="x", which="major", labelsize=15)
    ax.tick_params(axis="y", which="major", labelsize=15)
    plt.show()


def generate_mpc_results(
    mpc_orb,
    input_data,
    reconstructed_orbit,
    reconstructed_trajectory,
    reconstructed_orbit_mops,
    reconstructed_trajectory_mops,
    orbfit_limit_point=6,
    tw=15,
):

    _, _, _, _, _, unique_with_error = get_unique_and_pure(
        reconstructed_orbit, reconstructed_trajectory
    )

    _, _, _, _, _, unique_with_error_mops = get_unique_and_pure(
        reconstructed_orbit_mops, reconstructed_trajectory_mops
    )

    unique_confirmed_with_error = unique_with_error["ssnamenr"]
    unique_confirmed_with_error_mops = unique_with_error_mops["ssnamenr"]

    reconstructed_mpc, _ = mpc_crossmatch(mpc_orb, unique_confirmed_with_error)
    reconstructed_mpc_mops, _ = mpc_crossmatch(
        mpc_orb, unique_confirmed_with_error_mops
    )

    is_detectable = (
        input_data.sort_values("jd")
        .groupby("ssnamenr")
        .agg(nb_det=("ra", len), is_in_tw=("jd", lambda x: np.all(np.diff(x) <= tw)))
        .reset_index()
    )
    is_detectable = is_detectable[
        (is_detectable["nb_det"] >= orbfit_limit_point) & (is_detectable["is_in_tw"])
    ]
    input_mpc, _ = mpc_crossmatch(
        mpc_orb, pd.Series(is_detectable["ssnamenr"].unique())
    )

    return display_mpc_reconstruction(
        reconstructed_mpc, reconstructed_mpc_mops, input_mpc
    )


def ephem_preparation(reconstructed_orbit, reconstructed_trajectory, mpc_orb):
    _, _, _, pure_with_error, _, _ = get_unique_and_pure(
        reconstructed_orbit, reconstructed_trajectory
    )

    pure_with_error_candid = pure_with_error.reset_index()["ssoCandId"]
    pure_with_error_traj = reconstructed_trajectory[
        reconstructed_trajectory["ssoCandId"].isin(pure_with_error_candid)
    ].reset_index(drop=True)

    pure_with_error_prep_ephem = (
        pure_with_error_traj.sort_values(["ssoCandId", "jd"])[
            ["ssoCandId", "jd", "ssnamenr"]
        ]
        .groupby("ssoCandId")
        .agg(lambda x: list(x)[-1])
        .reset_index()
    )

    mpc_orb["MPCID"] = mpc_orb["Number"].str[1:-1]
    prep_ephem = mpc_orb.merge(
        pure_with_error_prep_ephem, left_on="MPCID", right_on="ssnamenr"
    ).merge(reconstructed_orbit, on="ssoCandId")

    prep_ephem["jd+7"] = prep_ephem["jd"] + 7
    prep_ephem["jd+30"] = prep_ephem["jd"] + 30
    prep_ephem["jd+120"] = prep_ephem["jd"] + 120
    prep_ephem["jd+360"] = prep_ephem["jd"] + 360

    return prep_ephem


def df_to_orb(df_orb, get_column_orbit, map_rename):

    prep_to_orb = df_orb[get_column_orbit].rename(
        map_rename,
        axis=1,
    )

    prep_to_orb["orbtype"] = "KEP"

    prep_to_orb["H"] = 14.45
    prep_to_orb["G"] = 0.15

    orb_dict = prep_to_orb.to_dict(orient="list")

    orb_dict["a"] = orb_dict["a"] * ut.au
    orb_dict["i"] = orb_dict["i"] * ut.deg
    orb_dict["node"] = orb_dict["node"] * ut.deg
    orb_dict["argper"] = orb_dict["argper"] * ut.deg
    orb_dict["M"] = orb_dict["M"] * ut.deg
    orb_dict["epoch"] = Time(orb_dict["epoch"], format="jd")
    orb_dict["H"] = orb_dict["H"] * ut.mag

    ast_orb_db = sso_py.Orbit.from_dict(orb_dict)
    return ast_orb_db


# Time window study
path_tw = "../fink_fat_experiments/time_window_experiments"


def gen_path(exp):
    """
    exp must be 'all' or 'mops'
    """
    assert exp == "all" or exp == "mops"
    if exp == "mops":
        return "{}/confirmed_{}_fink_fat/mpc/".format(exp, exp)
    elif exp == "all":
        return "{}_assoc/confirmed_{}_fink_fat/mpc/".format(exp, exp)


def get_tw_exp(tw_exp, kind_exp):
    assert (
        tw_exp == "15_2_2"
        or tw_exp == "15_2_15"
        or tw_exp == "15_15_2"
        or tw_exp == "15_15_15"
    )
    exp_dir = gen_path(kind_exp)
    reconstructed_orbit = pd.read_parquet(
        os.path.join(path_tw, tw_exp, exp_dir, "orbital.parquet")
    )
    reconstructed_trajectory = pd.read_parquet(
        os.path.join(path_tw, tw_exp, exp_dir, "trajectory_orb.parquet")
    )
    input_data = pd.read_parquet(os.path.join(path_tw, tw_exp, exp_dir, "save"))
    return reconstructed_orbit, reconstructed_trajectory, input_data


def get_stats(tw_exp, kind_exp):
    assert (
        tw_exp == "15_2_2"
        or tw_exp == "15_2_15"
        or tw_exp == "15_15_2"
        or tw_exp == "15_15_15"
    )
    exp_dir = gen_path(kind_exp)
    stats = pd.read_json(os.path.join(path_tw, tw_exp, exp_dir, "stats.json"))
    return stats


def get_api_time(path_tw, tw_exp, kind_exp):
    res = []
    log_path = os.path.join(path_tw, tw_exp, kind_exp, "fink_fat.log")
    with open(log_path, "r") as f:
        for lines in f.readlines():
            split_line = lines.split(":")
            if split_line[0] == "time taken to retrieve alerts from fink broker":
                res.append(float(split_line[1].split("/")[0]))
    return res


def print_time_stats(all_stats, path_tw):

    all_stats["api_time"] = get_api_time(path_tw, "15_2_2", "all_assoc")
    min_orbfit_time = all_stats.iloc[all_stats["orbfit_time"].argmin()]
    max_orbfit_time = all_stats.iloc[all_stats["orbfit_time"].argmax()]
    total_execution_time = (
        all_stats["assoc_time"].sum()
        + all_stats["orbfit_time"].sum()
        + all_stats["api_time"].sum()
    )

    return """
|      |API time|Association time|Orbfit time|Trajectory volume|Total|
|------|-------------|--------------------|---------------|---------------------|---------|
| min (sec)  |    {:0.2f}     |        {:0.2f}        |     {:0.2f} ({:0.0f} trajectories)      |         {:0.0f}        | X |
|median (sec)|    {:0.2f}     |        {:0.2f}        |     {:0.2f}      |         {:0.0f}        | X |
|max (sec)   |    {:0.2f}     |        {:0.2f}        |     {:0.2f} ({:0.0f} trajectories)      |         {:0.0f}        | X |
|total (hh:mm:ss) |    {}     |        {}        |     {}      |         X        |         {}        |
    """.format(
        all_stats["api_time"].min(),
        all_stats["assoc_time"].min(),
        min_orbfit_time["orbfit_time"],
        min_orbfit_time["nb_traj_to_orbfit"],
        all_stats["nb_traj_to_orbfit"].min(),
        all_stats["api_time"].median(),
        all_stats["assoc_time"].median(),
        all_stats["orbfit_time"].median(),
        all_stats["nb_traj_to_orbfit"].median(),
        all_stats["api_time"].max(),
        all_stats["assoc_time"].max(),
        max_orbfit_time["orbfit_time"],
        max_orbfit_time["nb_traj_to_orbfit"],
        all_stats["nb_traj_to_orbfit"].max(),
        str(datetime.timedelta(seconds=all_stats["api_time"].sum())),
        str(datetime.timedelta(seconds=all_stats["assoc_time"].sum())),
        str(datetime.timedelta(seconds=all_stats["orbfit_time"].sum())),
        str(datetime.timedelta(seconds=total_execution_time)),
    )


# def plot_hist_and_cdf(
#     data,
#     hist_range,
#     hist_title,
#     hist_xlabel,
#     hist_ylabel,
#     cdf_range,
#     cdf_title,
#     cdf_xlabel,
#     cdf_ylabel,
#     percent_cdf=[0.8, 0.9],
#     bins=200,
# ):
#     """
#     Plot the distribution and the cumulative from data.

#     Parameters
#     ----------
#     data: Series
#     hist_range: list or None
#     hist_title: String
#     hist_xlabel: String
#     hist_ylabel: String
#     cdf_range: list or None
#     cdf_title: String
#     cdf_xlabel: String
#     cdf_ylabel: String
#     percent_cdf: list , default = [0.8, 0.9]
#     bins: integer, default = 200

#     Returns
#     -------
#     None
#     """
#     _, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 10))

#     ax1.set_title(hist_title, fontdict={"size": 30})
#     ax1.set_xlabel(hist_xlabel, fontdict={"size": 30})
#     ax1.set_ylabel(hist_ylabel, fontdict={"size": 30})
#     ax1.set_yscale("log")
#     ax1.hist(data, bins=bins, range=hist_range)

#     ax2.set_title(cdf_title, fontdict={"size": 30})
#     ax2.set_ylabel(cdf_ylabel, fontdict={"size": 30})
#     ax2.set_xlabel(cdf_xlabel, fontdict={"size": 30})

#     mean_diff_value, mean_diff_bins, _ = ax2.hist(
#         data, range=cdf_range, bins=bins, cumulative=True, density=True, histtype="step"
#     )

#     x_interp = np.interp(
#         percent_cdf,
#         np.array(mean_diff_value, dtype="float64"),
#         np.array(mean_diff_bins[:-1], dtype="float64"),
#     )
#     ax2.scatter(x_interp, percent_cdf)

#     for i, value in enumerate(zip(percent_cdf, x_interp)):
#         txt = str(int(value[0] * 100)) + "% = " + str(value[1].round(decimals=2))
#         ax2.annotate(txt, (x_interp[i], percent_cdf[i]), fontsize=30)

#     ax1.tick_params(axis="x", which="major", labelsize=30)
#     ax1.tick_params(axis="y", which="major", labelsize=25)

#     ax2.tick_params(axis="x", which="major", labelsize=30)
#     ax2.tick_params(axis="y", which="major", labelsize=25)
#     plt.show()


def skybot_analysis(skybot_pdf):

    nb_counterparts = len(skybot_pdf[skybot_pdf["Ast_Name"] != "null"])
    counterparts_percent = (
        len(skybot_pdf[skybot_pdf["Ast_Name"] != "null"]) / len(skybot_pdf)
    ) * 100

    match_skybot = skybot_pdf[skybot_pdf["Ast_Name"] != "null"]

    purity_unknown = (
        skybot_pdf.groupby("d:ssoCandId")
        .agg(
            is_pure=("Ast_Name", lambda x: len(Counter(list(x)))),
            is_unknown=(
                "Ast_Name",
                lambda x: len(Counter(x)) == 1 and list(Counter(x).keys())[0] == "null",
            ),
        )
        .reset_index()
    )

    pure_obj = purity_unknown[
        (purity_unknown["is_pure"] == 1) & ~(purity_unknown["is_unknown"])
    ]

    nb_pure = len(pure_obj)
    pure_percent = len(pure_obj) / len(skybot_pdf.drop_duplicates("d:ssoCandId")) * 100

    not_pure_obj = purity_unknown[purity_unknown["is_pure"] != 1]
    nb_not_pure = len(not_pure_obj)
    not_pure_percent = (
        len(not_pure_obj) / len(skybot_pdf.drop_duplicates("d:ssoCandId")) * 100
    )

    unknown_obj = purity_unknown[
        (purity_unknown["is_pure"] == 1) & (purity_unknown["is_unknown"])
    ]
    nb_unknown = len(unknown_obj)
    unknown_percent = (
        len(unknown_obj) / len(skybot_pdf.drop_duplicates("d:ssoCandId")) * 100
    )

    _ = plt.figure(figsize=(8, 8))
    plt.hist(match_skybot["centerdist"], bins=200)
    plt.title(
        "Distribution of the separation \nbetween the observations and the skybot associations (max: 5 arcsecond)"
    )
    plt.show()

    header = """
|                 |  |
|-----------------|--|
|Total observation|{}|
|Alerts with SkyBot counterparts|{} ({} %)|
|Total orbit|{}|
|Pure orbit candidate|{} ({} %)|
|Not Pure orbit|{} ({} %)|
|Unknown orbit|{} ({} %)|
""".format(
        len(skybot_pdf),
        nb_counterparts,
        counterparts_percent,
        len(skybot_pdf["d:ssoCandId"].unique()),
        nb_pure,
        pure_percent,
        nb_not_pure,
        not_pure_percent,
        nb_unknown,
        unknown_percent,
    )

    return header, pure_obj, not_pure_obj, unknown_obj


def compute_sep_from_ztf(traj, fink_id, mpc_id, id_type=None):
    """Compute the separation between a Fink trajectory and a SSO object

    Parameters
    ----------
    fink_id: str
        Fink-FAT ssoCandId
    mpc_id: str or int
        MPC number or designation
    id_type: None or str
        If the mpc_id is a major body (e.g. planet satellite), specify id_type="id"

    Returns
    -----------
    out: list
        List of separation between the Fink-FAT trajectory alerts and the SSO
    """
    t_ztf = traj[traj["ssoCandId"] == fink_id]["jd"].values
    epoch = Time(t_ztf, format="jd", scale="utc")  # time in UT

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if id_type is None:
            eph = Ephem.from_horizons(mpc_id, location="I41", epochs=epoch)
        else:
            eph = Ephem.from_horizons(
                mpc_id, id_type=id_type, location="I41", epochs=epoch
            )

        t = eph.table
        prediction = SkyCoord(t["RA"], t["DEC"])

        obs = SkyCoord(
            traj[traj["ssoCandId"] == fink_id]["ra"].values,
            traj[traj["ssoCandId"] == fink_id]["dec"].values,
            unit="deg",
        )

    return prediction.separation(obs), prediction, obs


def plot_ff_error(ssoCandId, obs_ztf, *args):
    _ = plt.figure(figsize=(9, 9))
    plt.plot(
        obs_ztf.ra,
        obs_ztf.dec,
        label="Obs ZTF: {}".format(ssoCandId),
        color="C2",
        marker="o",
        linestyle="--",
        linewidth=3,
        markersize=8,
    )

    for (pred, ast_name) in args:

        plt.plot(
            pred.ra,
            pred.dec,
            label="Prediction: {}".format(ast_name),
            alpha=0.5,
            marker="o",
            linestyle="--",
            linewidth=1.5,
            markersize=4,
        )

        # ax = plt.gca()
        # for i, txt in enumerate(sep):
        #     ax.annotate("{:.4f} arcsec".format(txt.arcsecond), (pred.ra[i].value + 0.0005, pred.dec[i].value + 0.0008))

    ax = plt.gca()
    ax.tick_params(axis="x", which="major", labelsize=15)
    ax.tick_params(axis="y", which="major", labelsize=15)

    plt.xlabel("ra", fontdict={"size": 15})
    plt.ylabel("dec", fontdict={"size": 15})
    plt.legend(prop={"size": 18})
    plt.savefig("skybot_deviation_plot/{}.png".format(ssoCandId))
    plt.close()
