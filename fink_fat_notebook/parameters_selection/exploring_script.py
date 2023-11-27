import pandas as pd
import matplotlib.pyplot as plt

from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np

from matplotlib.lines import Line2D
import matplotlib.rcsetup as rcsetup

from bin.stat_cli import compute_residue


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


def load_candidate_data(columns=None):
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
    return pd.read_parquet("sso_candidate_data", columns=columns)


def plot_nb_det_distribution(df):
    """
    Plot the distribution of the number of detection for each sso in Fink
    """
    unique_nb_detection = df.drop_duplicates("ssnamenr")["nb_detection"]
    plt.hist(unique_nb_detection, 100, alpha=0.75, log=True)
    plt.xlabel("Number of detection")
    plt.ylabel("Number of SSO")
    plt.title("Number of detection of each sso in Fink")
    ax = plt.gca()
    plt.text(
        0.72,
        0.8,
        "min={},max={},median={}".format(
            min(unique_nb_detection),
            max(unique_nb_detection),
            int(unique_nb_detection.median()),
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
        df.groupby("ssnamenr")
        .agg(tw=("jd", lambda x: list(x)[-1] - list(x)[0]))
        .sort_values("tw")["tw"]
    )
    plt.hist(tw, 100, alpha=0.75, log=True)
    plt.xlabel("Observation window")
    plt.ylabel("Number of SSO")
    plt.title("Observation window of each sso in Fink")
    ax = plt.gca()
    plt.text(
        0.72,
        0.8,
        "min={},max={},median={}".format(min(tw), max(tw), int(tw.median())),
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax.transAxes,
    )
    plt.grid(True)
    plt.show()


def sep_df(x):
    """
    Compute the speed between two observations of solar system object
    """
    ra, dec, jd = x["ra"], x["dec"], x["jd"]

    c1 = SkyCoord(np.array(ra) * u.degree, np.array(dec) * u.degree)

    diff_jd = np.diff(jd)
    diff_jd = np.where(diff_jd < 1, 1, diff_jd)

    sep = c1[0:-1].separation(c1[1:]).degree

    velocity = np.divide(sep, diff_jd)

    return velocity


def intra_sep_df(x):
    """
    Compute the speed between two observations of solar system object
    """
    ra, dec = x["ra"], x["dec"]

    c1 = SkyCoord(np.array(ra) * u.degree, np.array(dec) * u.degree)

    sep = c1[0:-1].separation(c1[1:]).arcminute

    return sep


def plot_hist_and_cdf(
    data,
    hist_range,
    hist_title,
    hist_xlabel,
    hist_ylabel,
    cdf_range,
    cdf_title,
    cdf_xlabel,
    cdf_ylabel,
    percent_cdf=[0.8, 0.9],
    bins=200,
):
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
    ax1.set_yscale("log")
    ax1.hist(data, bins=bins, range=hist_range)

    ax2.set_title(cdf_title, fontdict={"size": 30})
    ax2.set_ylabel(cdf_ylabel, fontdict={"size": 30})
    ax2.set_xlabel(cdf_xlabel, fontdict={"size": 30})

    mean_diff_value, mean_diff_bins, _ = ax2.hist(
        data, range=cdf_range, bins=bins, cumulative=True, density=True, histtype="step"
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
    plt.show()


def mag_df(x):
    """
    Compute magnitude rate of the SSO for each filter and the color
    """
    mag, fid, jd = np.array(x["magpsf"]), np.array(x["fid"]), np.array(x["jd"])

    fid1 = np.where(fid == 1)[0]
    fid2 = np.where(fid == 2)[0]

    jd_fid1 = jd[fid1]
    jd_fid2 = jd[fid2]

    mag1 = mag[fid1]
    diff_mag1 = np.diff(mag1)

    mag2 = mag[fid2]
    diff_mag2 = np.diff(mag2)

    diff_jd1 = np.diff(jd_fid1)
    diff_jd2 = np.diff(jd_fid2)

    diff_jd1 = np.where(diff_jd1 < 1, 1, diff_jd1)
    diff_jd2 = np.where(diff_jd2 < 1, 1, diff_jd2)

    mask_jd1 = np.where(diff_jd1 < 15)[0]
    mask_jd2 = np.where(diff_jd2 < 15)[0]

    diff_mag1 = np.abs(diff_mag1)
    diff_mag2 = np.abs(diff_mag2)

    dmag_fid1 = np.divide(diff_mag1[mask_jd1], diff_jd1[mask_jd1])
    dmag_fid2 = np.divide(diff_mag2[mask_jd2], diff_jd2[mask_jd2])

    if len(dmag_fid1) == 0:
        return [], dmag_fid2, 0
    elif len(dmag_fid2) == 0:
        return dmag_fid1, [], 0
    else:
        diff_fid_mag = np.subtract(np.mean(mag1), np.mean(mag2))

        return dmag_fid1, dmag_fid2, diff_fid_mag


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


def angle(a, b, c):
    ba = b - a
    ca = c - a

    cosine_angle = np.dot(ba, ca) / (np.linalg.norm(ba) * np.linalg.norm(ca))
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)


def angle_df(x):
    ra, dec, jd = x["ra"], x["dec"], x["jd"]

    all_angle = []

    for i in range(len(ra) - 2):
        a = np.array([ra[i], dec[i]])
        b = np.array([ra[i + 1], dec[i + 1]])
        c = np.array([ra[i + 2], dec[i + 2]])

        jd1 = jd[i + 1]
        jd2 = jd[i + 2]
        diff_jd = jd2 - jd1
        diff_jd = diff_jd if diff_jd >= 1 else 1

        all_angle.append(angle(a, b, c) / diff_jd)

    return all_angle


def plot_ast_distrib(mpc_in_fink, ycol):
    _ = plt.figure(figsize=(25, 13))

    ax = plt.gca()

    for orb in mpc_in_fink["Orbit_type"].unique():
        cur_orb = mpc_in_fink[mpc_in_fink["Orbit_type"] == orb]
        if orb == "Object with perihelion distance < 1.665 AU":
            orb = "Small Peri Dist"

        if ycol == "i":
            ydata = np.sin(np.deg2rad(cur_orb[ycol]))
        else:
            ydata = cur_orb[ycol]
        ax.scatter(cur_orb["a"], ydata, label=orb, alpha=0.5, s=100)

    # ax.set_yscale('log')
    ax.set_xlabel("Semi major axis (AU)", fontdict={"size": 30})
    ax.set_ylabel(
        "Eccentricity" if ycol == "e" else "sin(inclination)", fontdict={"size": 30}
    )
    ax.set_xscale("log")
    ax.tick_params(axis="x", which="major", labelsize=25)
    ax.tick_params(axis="y", which="major", labelsize=20)
    ax.legend(prop={"size": 25})
    plt.show()


# def plot_ast_distrib_bft(df, ycol):
#     plt.set_cmap('viridis')
#     _ = plt.figure(figsize=(25, 13))

#     ax = plt.gca()

#     def get_label(x):
#         if x[0] == "MB":
#             if x[1] in ["Middle", "Outer", "Inner"]:
#                 return x[0]
#             else:
#                 return x[1]
#         elif x[0] == "NEA":
#             return x[1]
#         else:
#             return x[0]


#     df["class_alt"] = df["sso_class"].str.split(">").map(lambda x: get_label(x))

#     unique_lab = df["class_alt"].unique()

#     jet = plt.cm.tab20
#     idx = np.linspace(0, 1, len(unique_lab))
#     ax.set_prop_cycle(rcsetup.cycler('color', jet(idx)))

#     for orb in unique_lab:
#         cur_orb = df[df["class_alt"] == orb]
#         if len(cur_orb) > 0:

#             if ycol == "orbital_elements.inclination.value":
#                 ydata = np.sin(np.deg2rad(cur_orb[ycol]))
#             else:
#                 ydata = cur_orb[ycol]
#             ax.scatter(
#                 cur_orb["orbital_elements.semi_major_axis.value"],
#                 ydata,
#                 label=orb,
#                 alpha=0.5,
#                 s=100,
#                 cmap="veridis"
#             )

#     # ax.set_yscale('log')
#     ax.set_xlabel("Semi major axis (AU)", fontdict={"size": 30})
#     ax.set_ylabel(
#         "Eccentricity" if ycol == "orbital_elements.eccentricity.value" else "sin(inclination)", fontdict={"size": 30}
#     )
#     ax.set_xscale("log")
#     ax.tick_params(axis="x", which="major", labelsize=25)
#     ax.tick_params(axis="y", which="major", labelsize=20)
#     plt.legend(ncol=4, prop={"size": 20})
#     plt.tight_layout()
#     plt.show()


def plot_ast_distrib_with_incl(mpc_in_fink):
    _ = plt.figure(figsize=(20, 10))

    ax = plt.gca()
    ax.set_title(
        "Distribution of the asteroid in the Fink's database", fontdict={"size": 20}
    )

    for orb, mark in zip(mpc_in_fink["Orbit_type"].unique(), Line2D.filled_markers):
        cur_orb = mpc_in_fink[mpc_in_fink["Orbit_type"] == orb]
        ax.scatter(
            cur_orb["a"],
            cur_orb["e"],
            label=orb,
            alpha=0.5,
            c=cur_orb["i"],
            marker=mark,
            s=100,
        )

    # ax.set_yscale('log')
    ax.set_xlabel("Semi major axis (AU)", fontdict={"size": 20})
    ax.set_ylabel("Eccentricity", fontdict={"size": 20})
    ax.set_xscale("log")
    ax.tick_params(axis="x", which="major", labelsize=20)
    ax.tick_params(axis="y", which="major", labelsize=15)
    ax.legend(prop={"size": 15})
    plt.show()


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


def orbfit_perf_results():
    mpc_ast_data = pd.read_parquet(
        "../data/MPC_Database/mpcorb_extended.parquet",
        columns=[
            "Number",
            "Name",
            "Principal_desig",
            "Other_desigs",
            "a",
            "e",
            "i",
            "Node",
            "Peri",
            "M",
            "Epoch",
            "Orbit_type",
        ],
    )
    res_dict = {}
    for i in list(range(3, 11)) + [15]:
        print("current nb point: {}".format(i))

        cur_orb = pd.read_parquet("res_orbit_nb_point/{}_point_orbit.parquet".format(i))
        cur_tra = pd.read_parquet("res_orbit_nb_point/{}_point_traj.parquet".format(i))
        orb_and_traj = cur_tra.merge(cur_orb, on="trajectory_id").drop_duplicates(
            "trajectory_id"
        )
        orb_and_traj = orb_and_traj[orb_and_traj["a"] != -1.0]

        nb_orb = len(orb_and_traj)
        nb_traj = len(cur_orb)
        nb_fail = nb_traj - nb_orb
        nb_with_error = len(orb_and_traj[orb_and_traj["rms_a"] != -1.0])

        mpc_in_fink, _ = mpc_crossmatch(
            mpc_ast_data, pd.Series(orb_and_traj["ssnamenr"].unique())
        )
        merge_mpc_orbfit = merge_reconstruct_and_mpc(mpc_in_fink, orb_and_traj)

        df_with_perf = compute_residue(merge_mpc_orbfit)

        res_dict[i] = {
            "nb_traj": nb_traj,
            "nb_orb": nb_orb,
            "nb_fail": nb_fail,
            "nb_error": nb_with_error,
            "df_with_perf": df_with_perf,
            "orbit": cur_orb,
            "trajectory": cur_tra,
        }

    return res_dict


def plot_orbfit_perf(res_dict):
    efficiency = []
    purity = []
    for i in list(range(3, 11)) + [15]:
        nb_orb = res_dict[i]["nb_orb"]
        nb_traj = res_dict[i]["nb_traj"]
        nb_error = res_dict[i]["nb_error"]
        efficiency.append((nb_orb / nb_traj) * 100)
        purity.append((nb_error / nb_orb) * 100)

    plt.plot((list(np.arange(3, 11)) + [15]), efficiency, label="efficiency")
    plt.plot((list(np.arange(3, 11)) + [15]), purity, label="purity")
    plt.ylabel("(%)")
    plt.xlabel("Number of point")
    plt.legend()
    plt.show()


def plot_orbfit_diff_hist(res_dict, df, orb_param, title="", xlabel="", ylabel=""):
    _ = plt.figure(figsize=(20, 10))
    logbins = None
    for i in list(np.arange(3, 11)) + [15]:
        if df == "orbit":
            data = res_dict[i][df]
            data = data[data[orb_param] != -1.0]
            data = data[orb_param]
        else:
            data = res_dict[i][df][orb_param]

        if type(logbins) == type(None):
            _, bins = np.histogram(data, bins=200)
            tmp_logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))

            logbins = tmp_logbins

        plt.hist(
            data,
            bins=logbins,
            log=True,
            alpha=0.6,
            label="nb_point={}".format(i),
        )

    plt.legend(prop={"size": 15})
    plt.title(
        title,
        fontdict={"size": 20},
    )
    plt.ylabel(ylabel, fontdict={"size": 20})
    plt.xlabel(
        xlabel,
        fontdict={"size": 20},
    )
    ax = plt.gca()
    ax.tick_params(axis="x", which="major", labelsize=15)
    ax.tick_params(axis="y", which="major", labelsize=15)
    plt.show()


# papier version
def plot_ast_distrib_bft(df, ycol):
    # plt.set_cmap('tab10')
    _ = plt.figure(figsize=(15, 10))

    ax = plt.gca()

    def get_label_merged(x):
        if x[0] == "MB":
            if x[1] in ["Middle", "Outer", "Inner"]:
                return "Inner/Middle/Outer"
            else:
                return x[1]
        elif x[0] == "NEA":
            return x[1]
        else:
            return x[0]

    # Class definition
    leglist_nea = ["Amor", "Apollo", "Aten", "Atira"]

    leglist_mc = ["Mars-Crosser"]

    leglist_mb = ["Inner/Middle/Outer", "Cybele", "Hilda", "Hungaria", "Phocaea"]

    leglist_other = ["Trojan", "Centaur", "KBO"]

    # get labels
    df = df.copy()
    df["class_alt"] = df["sso_class"].str.split(">").map(lambda x: get_label_merged(x))
    unique_class_sso = df["class_alt"].unique()

    # Define color cycle
    jet = plt.cm.tab20
    idx = np.linspace(0, 1, 11)

    #     idx = np.arange(20)
    ax.set_prop_cycle(rcsetup.cycler("color", jet(idx)))

    # if orb in leglist_nea:
    #     marker = 'o'
    #     prelabel = 'NEA>'
    # if orb in leglist_mc:
    #     marker = 's'
    #     prelabel = ''
    # if orb in leglist_mb:
    #     marker = '.'
    #     prelabel = 'MB>'
    # if orb in leglist_other:
    #     marker = '^'
    #     prelabel = ''

    color_tag = {
        t: jet(n)
        for t, n in zip(
            leglist_nea + leglist_mc + leglist_mb + leglist_other,
            np.arange(len(unique_class_sso)),
        )
    }

    # tag_color = {
    #     n: t for t, n in zip(unique_class_sso, np.arange(len(unique_class_sso)))
    # }

    def plot_leglist(leglist, marker, loc, legend_title):
        cur_orb = df[df["class_alt"].isin(leglist)]

        color_series = cur_orb["class_alt"].map(color_tag)

        ax.scatter(
            cur_orb["orbital_elements.semi_major_axis.value"],
            cur_orb["orbital_elements.eccentricity.value"],
            c=color_series.values,
            label=leglist,
            alpha=0.5,
            marker=marker,
            s=60,
            cmap=plt.cm.tab20,
        )

        from matplotlib.lines import Line2D

        legend1 = ax.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    color="w",
                    markerfacecolor=v,
                    marker=marker,
                    label=k,
                    markersize=15,
                )
                for k, v in color_tag.items()
                if k in leglist
            ],
            loc=loc,
            title=legend_title,
            title_fontsize=15,
            prop={"size": 20},
        )
        ax.add_artist(legend1)

    locs = [(0.82, 0.615), (0.737, 0.539), (0.67, 0.2), (0.8, 0.01)]
    markers = ["o", "s", ".", "^"]

    plot_leglist(leglist_nea, markers[0], locs[0], "NEA")
    plot_leglist(leglist_mc, markers[1], locs[1], "")
    plot_leglist(leglist_mb, markers[2], locs[2], "MB")
    plot_leglist(leglist_other, markers[3], locs[3], "")

    # plt.gca().add_artist(
    #     plt.legend(
    #         handles=[sc],
    #         loc=(0.8, 0.1)
    #     )
    # )

    # Loop over classes
    # for orb in leglist_nea + leglist_mc + leglist_mb + leglist_other:
    #     cur_orb = df[df["class_alt"] == orb]

    #     if orb in leglist_nea:
    #         marker = 'o'
    #         prelabel = 'NEA>'
    #     if orb in leglist_mc:
    #         marker = 's'
    #         prelabel = ''
    #     if orb in leglist_mb:
    #         marker = '.'
    #         prelabel = 'MB>'
    #     if orb in leglist_other:
    #         marker = '^'
    #         prelabel = ''

    #     if len(cur_orb) > 0:

    #         if ycol == "orbital_elements.inclination.value":
    #             ydata = np.sin(np.deg2rad(cur_orb[ycol]))
    #         else:
    #             ydata = cur_orb[ycol]
    #         sc = ax.scatter(
    #             cur_orb["orbital_elements.semi_major_axis.value"],
    #             ydata,
    #             label=prelabel+orb,
    #             alpha=0.5,
    #             marker=marker,
    #             s=60,
    #             cmap=plt.cm.tab20
    #         )

    #         plt.gca().add_artist(
    #             plt.legend(
    #                 handles=[sc],
    #                 loc=(0.8, ( (y_padding*0.01) * len(all_leglist[y_padding % 4]) + 0.05))
    #             )
    #         )
    #         y_padding += 1

    # ax.set_yscale('log')
    ax.set_xlabel("Semi major axis (AU)")
    ax.set_ylabel(
        "Eccentricity"
        if ycol == "orbital_elements.eccentricity.value"
        else "sin(inclination)"
    )
    ax.set_xscale("log")
    ax.set_ylim(0, 1)

    # ax.legend(markerscale=3, loc='lower right')
    plt.show()
