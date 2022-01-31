import json
from os import path
from os import mkdir
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from astropy.time import Time
import glob
import matplotlib.cm as cm


def convert_dict_to_nested_type(report):
    if type(report) is dict:
        for k, v in report.items():
            print(k)
            convert_dict_to_nested_type(v)
            print()
    elif type(report) is list:
        for el in report:
            convert_dict_to_nested_type(el)
    else:
        print(type(report))


def save_report(report, date):

    dir_path = "report_db/"
    today = Time(date, format="jd")

    current_day = today.iso.split(" ")[0].split("-")
    dir_path = dir_path + str(current_day[1]) + "/"

    report_name = str(current_day[2]) + ".json"
    report_path = dir_path + report_name

    # convert_dict_to_nested_type(report)

    if path.isdir(dir_path):
        with open(report_path, "w") as outfile:
            # warning : serialize dictionary with numpy type doesn't work,
            # convert values to nested python type
            json.dump(report, outfile, indent=4)
    else:
        mkdir(dir_path)
        with open(report_path, "w") as outfile:
            json.dump(report, outfile, indent=4)


def parse_intra_night_report(intra_night_report):
    if len(intra_night_report) > 0:
        nb_sep_assoc = intra_night_report["number of separation association"]
        nb_mag_filter = intra_night_report[
            "number of association filtered by magnitude"
        ]
        nb_tracklets = intra_night_report["number of intra night tracklets"]

        metrics = intra_night_report["association metrics"]
        if len(metrics) > 0:
            pr = metrics["precision"]
            re = metrics["recall"]
            tp = metrics["True Positif"]
            fp = metrics["False Positif"]
            fn = metrics["False Negatif"]
            tot = metrics["total real association"]
        else:
            pr = 100
            re = 100
            tp = 0
            fp = 0
            fn = 0
            tot = 0
        return np.array(
            [nb_sep_assoc, nb_mag_filter, nb_tracklets, pr, re, tp, fp, fn, tot]
        )
    else:
        return np.array([0, 0, 0, 100, 100, 0, 0, 0, 0])


def parse_association_report(association_report):
    if len(association_report) > 0:
        nb_sep_assoc = association_report[
            "number of inter night separation based association"
        ]
        nb_mag_filter = association_report[
            "number of inter night magnitude filtered association"
        ]
        nb_angle_filter = association_report[
            "number of inter night angle filtered association"
        ]
        nb_duplicates = association_report["number of duplicated association"]

        metrics = association_report["metrics"]
        if len(metrics) > 0:
            pr = metrics["precision"]
            re = metrics["recall"]
            tp = metrics["True Positif"]
            fp = metrics["False Positif"]
            fn = metrics["False Negatif"]
            tot = metrics["total real association"]
            if fp == 0 and tot == 0:
                pr = 100
                re = 100
        else:
            pr = 100
            re = 100
            tp = 0
            fp = 0
            fn = 0
            tot = 0

        return np.array(
            [
                nb_sep_assoc,
                nb_mag_filter,
                nb_angle_filter,
                nb_duplicates,
                pr,
                re,
                tp,
                fp,
                fn,
                tot,
            ]
        )
    else:
        return np.array([0, 0, 0, 0, 100, 100, 0, 0, 0, 0])


def parse_trajectories_report(inter_night_report):
    updated_trajectories = inter_night_report["list of updated trajectories"]
    all_assoc_report = []
    if len(inter_night_report["all nid report"]) > 0:
        for report in inter_night_report["all nid report"]:
            traj_to_track_report = report["trajectories_to_tracklets_report"]
            traj_to_obs_report = report["trajectories_to_new_observation_report"]

            all_assoc_report.append(
                np.array(
                    [
                        parse_association_report(traj_to_track_report),
                        parse_association_report(traj_to_obs_report),
                    ]
                )
            )

    return updated_trajectories, np.array(all_assoc_report)


def parse_tracklets_obs_report(inter_night_report):
    updated_trajectories = inter_night_report["list of updated trajectories"]
    all_assoc_report = []
    if len(inter_night_report["all nid report"]) > 0:
        for report in inter_night_report["all nid report"]:
            obs_to_track_report = report["old observation to tracklets report"]
            obs_to_obs_report = report["old observation to new observation report"]

            all_assoc_report.append(
                np.array(
                    [
                        parse_association_report(obs_to_track_report),
                        parse_association_report(obs_to_obs_report),
                    ]
                )
            )

    return updated_trajectories, np.array(all_assoc_report)


def parse_inter_night_report(report):
    intra_report = report["intra night report"]
    traj_report = report["trajectory association report"]
    if "tracklets and observation association report" in report:
        track_report = report["tracklets and observation association report"]
        parse_track_report = parse_tracklets_obs_report(track_report)
    else:
        parse_track_report = [], np.array([])

    nb_traj = report["nb trajectories"]
    nb_most_recent_traj = report["nb most recent traj"]
    nb_old_obs = report["nb old observations"]
    nb_new_obs = report["nb new observations"]
    time = report["computation time of the night"]

    parse_intra_report = parse_intra_night_report(intra_report)
    parse_traj_report = parse_trajectories_report(traj_report)

    return (
        parse_intra_report,
        parse_traj_report,
        parse_track_report,
        np.array([nb_traj, time, nb_most_recent_traj, nb_old_obs, nb_new_obs]),
    )


def open_and_parse_report(path):
    with open(path, "r") as file:
        inter_night_report = json.load(file)
        return parse_inter_night_report(inter_night_report)


def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct * total / 100.0))
        return "{p:.2f}%  ({v:d})".format(p=pct, v=val)

    return my_autopct


def plot_report(parse_report):

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(20, 20))

    intra_assoc_value = parse_report[0]

    traj_assoc_value = parse_report[1]

    track_assoc_value = parse_report[2]

    intra_values = [intra_assoc_value[1], (intra_assoc_value[0] - intra_assoc_value[1])]
    labels = ("magnitude filtering", "remaining associations")
    explode = (0.1, 0.0)

    ax.pie(
        intra_values,
        explode=explode,
        shadow=True,
        labels=labels,
        autopct=make_autopct(intra_values),
    )
    ax.axis("equal")

    def transform_data(data):
        return np.array(
            [data[1], data[2], data[3], (data[0] - data[1] - data[2] - data[3])]
        )

    if len(traj_assoc_value[1]) > 0:
        traj_assoc_value = traj_assoc_value[1].sum(axis=1).sum(axis=0)
        traj_assoc_value = transform_data(traj_assoc_value)
    else:
        traj_assoc_value = np.array([0, 0, 0, 0])

    if len(track_assoc_value[1]) > 0:
        track_assoc_value = track_assoc_value[1].sum(axis=1).sum(axis=0)
        track_assoc_value = transform_data(track_assoc_value)
    else:
        track_assoc_value = np.array([0, 0, 0, 0])

    vals = np.concatenate([[traj_assoc_value], [track_assoc_value]], axis=0)
    slop = 0.0001

    group_size = vals.sum(axis=1) + slop
    subgroup_size = vals.flatten()
    subgroup_names = subgroup_size

    # Create colors
    a, b = [plt.cm.Blues, plt.cm.Reds]

    ax2.axis("equal")
    mypie, _ = ax2.pie(
        group_size, radius=1.3, labels=group_size - slop, colors=[a(0.6), b(0.6)]
    )
    plt.setp(mypie, width=0.3, edgecolor="white")

    # Second Ring (Inside)
    mypie2, _ = ax2.pie(
        subgroup_size,
        radius=1.3 - 0.3,
        labels=subgroup_names,
        labeldistance=0.7,
        colors=[
            a(subgroup_size[0] / group_size[0] - slop),
            a(subgroup_size[1] / group_size[0] - slop),
            a(subgroup_size[2] / group_size[0] - slop),
            a(subgroup_size[3] / group_size[0] - slop),
            b(subgroup_size[0] / group_size[1] - slop),
            b(subgroup_size[1] / group_size[1] - slop),
            b(subgroup_size[2] / group_size[1] - slop),
            b(subgroup_size[3] / group_size[1] - slop),
        ],
    )
    plt.setp(mypie2, width=0.4, edgecolor="white")
    ax2.margins(0, 0)

    subgroup_names_legs = [
        "Trajectory association",
        "Tracklets and observation association",
        "filtered by magnitude",
        "filtered by angle",
        "duplicated association",
        "remaining association",
        "filtered by magnitude",
        "filtered by angle",
        "duplicated association",
        "remaining association",
    ]
    ax2.legend(subgroup_names_legs, loc="best")

    ax.set_title("intra night association")
    ax2.set_title("inter night association")

    plt.show()


def get_intra_night_metrics(parse_report):
    intra_night = parse_report[0]
    return np.array(intra_night)[3:]


def get_intra_night_associations(parse_report):
    intra_night = parse_report[0]
    return np.array(intra_night)[:3]


def get_inter_night_metrics(parse_report):
    traj_assoc_report = parse_report[1][1]

    track_assoc_report = parse_report[2][1]

    if len(traj_assoc_report) > 0:
        traj_to_tracklets = traj_assoc_report[:, 0, 4:]

        traj_to_obs = traj_assoc_report[:, 1, 4:]
    else:
        traj_to_tracklets = np.array([100, 100, 0, 0, 0, 0])
        traj_to_obs = np.array([100, 100, 0, 0, 0, 0])

    if len(track_assoc_report) > 0:
        old_obs_to_track = track_assoc_report[:, 0, 4:]

        old_obs_to_new_obs = track_assoc_report[:, 1, 4:]
    else:
        old_obs_to_track = np.array([100, 100, 0, 0, 0, 0])
        old_obs_to_new_obs = np.array([100, 100, 0, 0, 0, 0])

    return traj_to_tracklets, traj_to_obs, old_obs_to_track, old_obs_to_new_obs


def get_inter_night_stat(parse_report):
    return parse_report[3]


def get_inter_night_associations(parse_report):
    traj_assoc_report = parse_report[1][1]

    track_assoc_report = parse_report[2][1]

    if len(traj_assoc_report) > 0:
        traj_to_tracklets = traj_assoc_report[:, 0, :4]

        traj_to_obs = traj_assoc_report[:, 1, :4]
    else:
        traj_to_tracklets = np.array([0, 0, 0, 0])

        traj_to_obs = np.array([0, 0, 0, 0])

    if len(track_assoc_report) > 0:
        old_obs_to_track = track_assoc_report[:, 0, :4]

        old_obs_to_new_obs = track_assoc_report[:, 1, :4]
    else:
        old_obs_to_track = np.array([0, 0, 0, 0])
        old_obs_to_new_obs = np.array([0, 0, 0, 0])

    return traj_to_tracklets, traj_to_obs, old_obs_to_track, old_obs_to_new_obs


def mean_metrics_over_nights(metrics):
    # test = np.ones(np.shape(metrics), dtype=bool)
    # idx = np.where(metrics[:, -1] == 0)
    # test[idx] = np.zeros(6, dtype=bool)
    return np.mean(metrics, axis=0)


def plot_metrics(fig, metrics, axes, title):
    values_idx = np.arange(1, np.shape(metrics[:, :2])[0] + 1)

    css_color = mcolors.CSS4_COLORS
    axes[0].plot(
        values_idx,
        np.cumsum(metrics[:, 0]) / values_idx,
        label="precision",
        color=css_color["crimson"],
    )
    axes[0].plot(
        values_idx,
        np.cumsum(metrics[:, 1]) / values_idx,
        label="recall",
        color=css_color["chocolate"],
    )
    axes[0].set_title(title)
    axes[1].plot(
        values_idx,
        np.cumsum(metrics[:, 2:-1], axis=0),
        alpha=0.8,
        label=["True Positif", "False Positif", "False Negatif"],
    )

    axes[1].plot(
        values_idx, np.cumsum(metrics[:, -1]), label="total real association", alpha=0.7
    )

    axes[1].set_yscale("log")

    colors = [
        css_color["green"],
        css_color["red"],
        css_color["royalblue"],
        css_color["black"],
    ]
    for i, j in enumerate(axes[1].lines):
        j.set_color(colors[i])

    lines_labels = [ax.get_legend_handles_labels() for ax in axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels, loc=(0.5, 0.45), framealpha=0.2)


def plot_intra_assoc(assoc, axes, title):

    values_idx = np.arange(1, np.shape(assoc[:, :2])[0] + 1)

    axes.plot(
        values_idx,
        assoc,
        label=["separation assoc", "magnitude filter", "detected tracklets"],
    )
    axes.set_title(title)
    axes.legend()


def plot_inter_assoc(assoc, ax, title):
    values_idx = np.arange(1, np.shape(assoc[:, :2])[0] + 1)

    assoc[:, 1] = assoc[:, 0] - assoc[:, 1]
    assoc[:, 2] = assoc[:, 1] - assoc[:, 2]
    assoc[:, 3] = np.cumsum(assoc[:, 2] - assoc[:, 3])

    ax.plot(
        values_idx,
        assoc,
        label=[
            "separation assoc",
            "magnitude filter",
            "angle filter",
            "remain after removing duplicates",
        ],
    )
    ax.set_yscale("log")
    ax.set_title(title)


def plot_inter_stat(stat, axes, title):
    values_idx = np.arange(1, np.shape(stat[:, :2])[0] + 1)

    axes[0].plot(values_idx, np.cumsum(stat[:, 1]))
    axes[0].set_title("cumulative elapsed time")

    axes[1].plot(values_idx, stat[:, 0])
    axes[1].set_title("cumulative number of trajectories")

    axes[2].plot(
        values_idx,
        stat[:, 2:],
        label=[
            "number of most recent trajectories",
            "number of old observations",
            "number of new observations",
        ],
    )
    axes[2].set_title("inputs statistics")
    axes[2].legend()


def plot_trajectories(traj_df, mpc_plot):
    gb_traj = (
        traj_df.groupby(["trajectory_id"])
        .agg(
            {
                "ra": list,
                "dec": list,
                "dcmag": list,
                "fid": list,
                "nid": list,
                "candid": lambda x: len(x),
            }
        )
        .reset_index()
    )

    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(40, 40))

    colors = cm.jet(np.linspace(0, 1, len(mpc_plot)))

    for i, rows in mpc_plot.iterrows():
        ra = rows["ra"]
        dec = rows["dec"]

        ax1.scatter(ra, dec, color=colors[i])

    colors = cm.Set1(np.linspace(0, 1, len(gb_traj)))
    for i, rows in gb_traj.iterrows():
        ra = rows["ra"]
        dec = rows["dec"]
        ax2.scatter(ra, dec, color=colors[i])

    ax1.set_title("real trajectories")
    ax2.set_title("detected trajectories")


def load_performance_stat(only_intra_night=False):
    all_path_report = sorted(glob.glob("report_db/*/*"))

    all_inter_metrics = [[], [], [], []]
    all_intra_metrics = []

    all_inter_assoc = [[], [], [], []]
    all_intra_assoc = []

    all_inter_stat = []

    with open(all_path_report[0], "r") as file:
        intra_night_report = json.load(file)
        intra_night_values = parse_intra_night_report(intra_night_report)

        nb_traj = intra_night_report["nb trajectories"]
        nb_most_recent_traj = intra_night_report["nb most recent traj"]
        nb_old_obs = intra_night_report["nb old observations"]
        nb_new_obs = intra_night_report["nb new observations"]
        time = intra_night_report["computation time of the night"]

        all_intra_metrics.append(intra_night_values[3:])
        all_intra_assoc.append(intra_night_values[:3])
        all_inter_stat.append(
            np.array([nb_traj, time, nb_most_recent_traj, nb_old_obs, nb_new_obs])
        )

    for current_path in all_path_report[1:]:

        if only_intra_night:
            with open(current_path, "r") as file:
                intra_night_report = json.load(file)
                intra_night_values = parse_intra_night_report(
                    intra_night_report["intra night report"]
                )

                nb_traj = intra_night_report["nb trajectories"]
                nb_most_recent_traj = intra_night_report["nb most recent traj"]
                nb_old_obs = intra_night_report["nb old observations"]
                nb_new_obs = intra_night_report["nb new observations"]
                time = intra_night_report["computation time of the night"]

                all_intra_metrics.append(intra_night_values[3:])
                all_intra_assoc.append(intra_night_values[:3])
                all_inter_stat.append(
                    np.array(
                        [nb_traj, time, nb_most_recent_traj, nb_old_obs, nb_new_obs]
                    )
                )
                continue

        parse_report = open_and_parse_report(current_path)

        inter_night_assoc = get_inter_night_associations(parse_report)
        intra_night_assoc = get_intra_night_associations(parse_report)
        all_intra_assoc.append(intra_night_assoc)

        inter_night_metrics = get_inter_night_metrics(parse_report)
        intra_night_metrics = get_intra_night_metrics(parse_report)
        all_intra_metrics.append(intra_night_metrics)

        inter_stat = get_inter_night_stat(parse_report)
        all_inter_stat.append(inter_stat)

        for i in range(4):
            metrics_shape = np.shape(inter_night_metrics[i])
            assoc_shape = np.shape(inter_night_assoc[i])

            if assoc_shape[0] > 1 and len(assoc_shape) == 2:
                mean_assoc = np.nan_to_num(
                    mean_metrics_over_nights(inter_night_assoc[i])
                )
                all_inter_assoc[i].append(mean_assoc.reshape((1, 4)))
            else:
                all_inter_assoc[i].append(inter_night_assoc[i].reshape((1, 4)))

            if metrics_shape[0] > 1 and len(metrics_shape) == 2:
                mean_metrics = np.nan_to_num(
                    mean_metrics_over_nights(inter_night_metrics[i])
                )
                all_inter_metrics[i].append(mean_metrics.reshape((1, 6)))
            else:
                all_inter_metrics[i].append(inter_night_metrics[i].reshape((1, 6)))

    all_intra_assoc = np.stack(all_intra_assoc)
    all_inter_assoc = [np.concatenate(i, axis=0) for i in all_inter_assoc if len(i) > 0]

    all_intra_metrics = np.stack(all_intra_metrics)
    all_inter_metrics = [
        np.concatenate(i, axis=0) for i in all_inter_metrics if len(i) > 0
    ]

    all_inter_stat = np.stack(all_inter_stat)

    return (
        all_intra_assoc,
        all_inter_assoc,
        all_intra_metrics,
        all_inter_metrics,
        all_inter_stat,
    )


def plot_performance_test(
    all_intra_assoc,
    all_inter_assoc,
    all_intra_metrics,
    all_inter_metrics,
    all_inter_stat,
):
    fig1 = plt.figure()
    ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2)
    ax2 = plt.subplot2grid((3, 3), (1, 0))
    ax3 = plt.subplot2grid((3, 3), (2, 0))
    ax4 = plt.subplot2grid((3, 3), (1, 1))
    ax5 = plt.subplot2grid((3, 3), (2, 1))
    ax6 = plt.subplot2grid((3, 3), (0, 2))
    ax7 = plt.subplot2grid((3, 3), (1, 2))
    ax8 = plt.subplot2grid((3, 3), (2, 2))

    stat_axes = [ax1, ax6, ax8]
    assoc_axes = [ax2, ax3, ax4, ax5]

    plot_inter_stat(all_inter_stat, stat_axes, "inter night statistics")
    plot_intra_assoc(all_intra_assoc, ax7, "intra night association")

    fig2, axes = plt.subplots(5, 2)
    metrics_axes = np.array(axes)
    plot_metrics(fig2, all_intra_metrics, metrics_axes[0], "intra night metrics")

    metrics_title = [
        "trajectory to tracklets metrics",
        "trajectory to new observations metrics",
        "old observations to tracklets metrics",
        "old observations to new observations metrics",
    ]

    assoc_title = [
        "trajectory to tracklets associations",
        "trajectory to new observations associations",
        "old observations to tracklets associations",
        "old observations to new observations associations",
    ]

    fig2.suptitle("Metrics")

    if len(all_inter_assoc) > 0 and len(all_inter_metrics) > 0:
        for i, met_ax, met_title, assoc_ax, title in zip(
            range(4), metrics_axes[1:], metrics_title, assoc_axes, assoc_title
        ):
            plot_metrics(fig2, all_inter_metrics[i], met_ax, met_title)
            plot_inter_assoc(all_inter_assoc[i], assoc_ax, title)

    lines_labels = [ax.get_legend_handles_labels() for ax in assoc_axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig1.legend(lines[:4], labels[:4], loc=(0.55, 0.53), framealpha=0.2)

    plt.tight_layout()
    plt.show()
