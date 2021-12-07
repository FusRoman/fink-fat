import json
from os import path
from os import mkdir
import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time


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
            pr = 0
            re = 0
            tp = 0
            fp = 0
            fn = 0
            tot = 0
        return nb_sep_assoc, nb_mag_filter, nb_tracklets, pr, re, tp, fp, fn, tot
    else:
        return 0, 0, 0, 0, 0, 0, 0, 0, 0


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
        else:
            pr = 0
            re = 0
            tp = 0
            fp = 0
            fn = 0
            tot = 0

        return np.array([nb_sep_assoc, nb_mag_filter, nb_angle_filter, nb_duplicates, pr, re, tp, fp, fn, tot])
    else:
        return np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])


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
    track_report = report["tracklets and observation association report"]

    parse_intra_report = parse_intra_night_report(intra_report)
    parse_traj_report = parse_trajectories_report(traj_report)
    parse_track_report = parse_tracklets_obs_report(track_report)

    return parse_intra_report, parse_traj_report, parse_track_report


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

def get_inter_night_metrics(parse_report):
    traj_assoc_report = parse_report[1][1]

    track_assoc_report = parse_report[2][1]

    traj_to_tracklets = traj_assoc_report[:, 0, 4:]
    
    traj_to_obs = traj_assoc_report[:, 1, 4:]
    
    if len(track_assoc_report) > 0:
        old_obs_to_track = track_assoc_report[:, 0, 4:]

        old_obs_to_new_obs = track_assoc_report[:, 1, 4:]
    else:
        old_obs_to_track = np.array([0, 0, 0, 0, 0, 0])
        old_obs_to_new_obs = np.array([0, 0, 0, 0, 0, 0])

    return traj_to_tracklets, traj_to_obs, old_obs_to_track, old_obs_to_new_obs


def mean_metrics_over_nights(metrics):
    test = np.ones(np.shape(metrics), dtype=bool)
    idx = np.where(metrics[:, -1] == 0)
    test[idx] = np.zeros(6, dtype=bool)
    return np.mean(metrics, axis=0, where=test)

if __name__ == "__main__":
    import test_sample as ts  # noqa: F401

    import glob

    all_path_report = glob.glob("report_db/*/*")

    all_metrics = [[], [], [], []]

    for path in all_path_report:
        
        parse_report = open_and_parse_report(path)

        inter_night_metrics = get_inter_night_metrics(parse_report)
        
        for i in range(4):
            metrics_shape = np.shape(inter_night_metrics[i])
            
            if metrics_shape[0] > 1 and len(metrics_shape) == 2:
                mean_metrics = np.nan_to_num(mean_metrics_over_nights(inter_night_metrics[i]))
                all_metrics[i].append(mean_metrics.reshape((1, 6)))
            else:
                all_metrics[i].append(inter_night_metrics[i].reshape((1, 6)))

    all_metrics = [np.concatenate(i, axis=0) for i in all_metrics]

    for metrics in all_metrics:
        print(mean_metrics_over_nights(metrics))
    print()

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 20))

    #,"True Positif", "False Positif", "False Negatif", "total real association"

    ax1.plot(np.arange(np.shape(all_metrics[0][:, :2])[0]), all_metrics[0][:, :2], label=["precision", "recall"])
    ax1.legend()

    ax2.plot(np.arange(np.shape(all_metrics[0][:, 2:])[0]), all_metrics[0][:, 2:], label=["True Positif", "False Positif", "False Negatif", "total real association"])
    ax2.legend()
    plt.show()

