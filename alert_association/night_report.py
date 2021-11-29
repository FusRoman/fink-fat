import json
from os import path
from os import mkdir
import datetime
import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time


def save_report(report, date):
    print(report)
    print()
    print(type(report['nid of the next night']))

    dir_path = "report_db/"
    today = Time(date, format='jd')
    
    current_day = today.iso.split(" ")[0].split("-")
    print(current_day)
    dir_path = dir_path + str(current_day[1]) + "/"
    print(dir_path)
    report_name = str(current_day[2]) + ".json"
    report_path = dir_path + report_name
    if path.isdir(dir_path):
        with open(report_path, "w") as outfile:
            # warning : serialize dictionary with numpy integer doesn't work,
            # convert values to standard integer or float 
            json.dump(report, outfile, indent=4)
    else:
        mkdir(dir_path)
        with open(report_path, "w") as outfile:
            json.dump(report, outfile, indent=4)


def parse_intra_night_report(intra_night_report):
    nb_sep_assoc = intra_night_report["number of separation association"]
    nb_mag_filter = intra_night_report["number of association filtered by magnitude"]
    nb_tracklets = intra_night_report["number of intra night tracklets"]
    return nb_sep_assoc, nb_mag_filter, nb_tracklets


def parse_association_report(association_report):
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

    return np.array([nb_sep_assoc, nb_mag_filter, nb_angle_filter, nb_duplicates])


def parse_trajectories_report(inter_night_report):
    updated_trajectories = inter_night_report["list of updated trajectories"]
    all_assoc_report = []
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
            [data[1], data[2], data[3], (data[0] - data[1] - data[2] - data[3]),]
        )

    traj_assoc_value = traj_assoc_value[1].sum(axis=1).sum(axis=0)
    traj_assoc_value = transform_data(traj_assoc_value)

    track_assoc_value = track_assoc_value[1].sum(axis=1).sum(axis=0)
    print(track_assoc_value)
    track_assoc_value = transform_data(track_assoc_value)
    print(track_assoc_value)

    size = 0.5
    vals = np.concatenate([[traj_assoc_value], [track_assoc_value]], axis=0)

    group_names = ["Trajectory association", "Tracklets and observation association"]
    group_size = vals.sum(axis=1)
    subgroup_size = vals.flatten()
    subgroup_names = subgroup_size

    # Create colors
    a, b = [plt.cm.Blues, plt.cm.Reds]

    ax2.axis("equal")
    mypie, _ = ax2.pie(
        group_size, radius=1.3, labels=group_size, colors=[a(0.6), b(0.6)]
    )
    plt.setp(mypie, width=0.3, edgecolor="white")

    slop = 0.01
    # Second Ring (Inside)
    mypie2, _ = ax2.pie(
        subgroup_size,
        radius=1.3 - 0.3,
        labels=subgroup_names,
        labeldistance=0.7,
        colors=[
            a(subgroup_size[0] / group_size[0]-slop),
            a(subgroup_size[1] / group_size[0]-slop),
            a(subgroup_size[2] / group_size[0]-slop),
            a(subgroup_size[3] / group_size[0]-slop),
            b(subgroup_size[0] / group_size[1]-slop),
            b(subgroup_size[1] / group_size[1]-slop),
            b(subgroup_size[2] / group_size[1]-slop),
            b(subgroup_size[3] / group_size[1]-slop),
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


if __name__ == "__main__":
    import test_sample as ts  # noqa: F401

    # res = open_and_parse_report("report_db/2021-11-26.json")
    # print(res)
    # print()
    # print()

    # plot_report(res)
    # save_report(ts.inter_night_report1)

    save_report({"fake date": 1520,"fake report":{"id1":1, "id2":3}}, "2459274.9054398")

    # plot_report('report_db/2021-11-26.json')
