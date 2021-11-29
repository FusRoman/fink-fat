import json
from os import path
from os import mkdir
from datetime import date
import matplotlib.pyplot as plt
import numpy as np


def save_report(report):
    dir_path = "report_db/"
    report_name = str(date.today()) + ".json"
    report_path = dir_path + report_name
    if path.isdir(dir_path):
        with open(report_path, "w") as outfile:
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
                [parse_association_report(traj_to_track_report),
                parse_association_report(traj_to_obs_report)]
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
                [parse_association_report(obs_to_track_report),
                parse_association_report(obs_to_obs_report)]
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


def plot_report(parse_report):

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    intra_assoc_value = parse_report[0]

    traj_assoc_value = parse_report[1]

    track_assoc_value = parse_report[2]

    labels = ("magnitude filtering", "remaining associations")
    explode = (0.1, 0.0)
    ax.pie(
        [intra_assoc_value[1], (intra_assoc_value[0] - intra_assoc_value[1])],
        explode=explode,
        shadow=True,
        labels=labels,
        autopct="%1.1f%%",
    )
    ax.axis("equal")

    traj_assoc_value = traj_assoc_value[1].sum(axis=1).sum(axis=0)
    track_assoc_value = track_assoc_value[1].sum(axis=1).sum(axis=0)
    print(traj_assoc_value)
    print(track_assoc_value)
    print(np.c_[traj_assoc_value, track_assoc_value])

    plt.show()


if __name__ == "__main__":
    import test_sample as ts  # noqa: F401

    res = open_and_parse_report("report_db/2021-11-26.json")
    print(res)
    print()
    print()

    plot_report(res)
    # save_report(ts.inter_night_report1)

    # plot_report('report_db/2021-11-26.json')
