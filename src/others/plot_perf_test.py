from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from src.others.utils import load_data
from src.others.utils import get_mpc_database
import json
import os


def plot_orbit_type(orbit_param, title, y, ylabel, savefig=False, test_name=""):
    g = sns.scatterplot(data=orbit_param, x="a", y=y, hue="Orbit_type")

    g.set(xlabel="semi-major axis (UA)", ylabel=ylabel)
    g.set_title(title)
    # g.legend(bbox_to_anchor= (1.2,1))

    if savefig:
        if not os.path.exists(test_name):
            os.mkdir(test_name)

        g.set(xlim=(0, 7))
        plt.savefig(os.path.join(test_name, title + "_" + ylabel), dpi=500)
        g.set(xlim=(6, 500))
        plt.savefig(os.path.join(test_name, title + "_" + ylabel + "_distant"), dpi=500)

    plt.close()
    # g.set(xlim=(0, 7))
    # plt.show()


def plot_stat(stat_df, test_name):
    g = sns.barplot(data=stat_df, x="night", y="time")
    plt.xticks(rotation=90)
    g.set(xlabel="night identifier", ylabel="time (sec)")
    g.set_title("Computation time of the associations algorithm over nights")
    plt.savefig(os.path.join(test_name, "time_plot"), dpi=500)
    plt.close()

    cum_time = np.cumsum(stat_df["time"])
    stat_df["cum_time"] = cum_time

    g = sns.lineplot(data=stat_df, x="night", y="cum_time")
    plt.xticks(rotation=90)
    g.set(xlabel="night identifier", ylabel="time (sec)")
    g.set_title("Cumulative computation time of the associations algorithm over nights")
    plt.savefig(os.path.join(test_name, "cum_time_plot"), dpi=500)
    plt.close()

    g = sns.barplot(data=stat_df, x="night", y="trajectory_size")
    g.set(xlabel="night identifier", ylabel="number of recorded trajectory")
    g.set_title("Size of the recorded trajectory set over nights")
    plt.xticks(rotation=90)
    plt.savefig(os.path.join(test_name, "trajectory_plot"), dpi=500)
    plt.close()


if __name__ == "__main__":
    sns.set_context("talk")
    sns.set(rc={"figure.figsize": (20, 9)})

    test_name = "perf_test_2"

    df_sso = load_data("Solar System MPC", 0)

    trajectory_df = pd.read_parquet("src/others/perf_test/{}.parquet".format(test_name))

    with open("src/others/perf_test/{}.json".format(test_name), "r") as json_file:
        stat = json.load(json_file)

    test_night = np.unique(trajectory_df["nid"])

    test_night = np.append(test_night, test_night[-1] + 1)

    print("number of processed night: {}".format(len(test_night)))

    df_sso = df_sso[df_sso["nid"].isin(test_night)]

    nb_traj = len(np.unique(df_sso["ssnamenr"]))
    print("number of objects in these nights: {}".format(nb_traj))

    traj_size = df_sso.groupby(["ssnamenr"]).count().reset_index()

    detected_traj = traj_size[traj_size["ra"] >= 5]

    traj_not_observed = traj_size[traj_size["ra"] < 5]

    print("number of objects that can be detected: {}".format(len(detected_traj)))
    print()

    traj_with_orb = trajectory_df[trajectory_df["a"] != -1.0]

    traj_cand_size = traj_with_orb.groupby(["trajectory_id"]).agg(
        trajectory_size=("candid", lambda x: len(list(x))),
        error=("ssnamenr", lambda x: len(np.unique(x))),
        ssnamenr=("ssnamenr", np.unique),
        nid=("nid", list),
        assoc=("nid", lambda x: Counter(x)),
        track=("nid", lambda x: len(np.unique(x))),
    )

    true_candidate = traj_cand_size[traj_cand_size["error"] == 1]

    false_candidate = traj_cand_size[traj_cand_size["error"] != 1]

    real_mpc_trajectories = np.unique(
        traj_cand_size[traj_cand_size["error"] == 1]["ssnamenr"]
    )

    not_detected_object = detected_traj[
        ~detected_traj["ssnamenr"].isin(real_mpc_trajectories)
    ]

    print(
        "number of trajectories with orbital elements: {}".format(len(traj_cand_size))
    )

    print("number of true detected trajectories: {}".format(len(true_candidate)))

    print("number of overlaping trajectories: {}".format(len(false_candidate)))

    print()
    print("purity: {}".format((len(true_candidate) / len(traj_cand_size)) * 100))

    print()
    print()
    print("number of mpc object recover: {}".format(len(real_mpc_trajectories)))

    print(
        "efficiency: {}".format((len(real_mpc_trajectories) / len(detected_traj)) * 100)
    )

    piece_of_traj = true_candidate.groupby(["ssnamenr"]).count().reset_index()

    print()
    print()
    print("Fragmented trajectories")
    print(piece_of_traj[piece_of_traj["error"] > 1])

    print()
    print()

    real_mpc_object = traj_cand_size[traj_cand_size["error"] == 1]

    def detect_tracklets(x):

        counter = x["assoc"]

        most_c = np.array(counter.most_common())

        most_c = most_c[most_c[:, 0].argsort()]

        if most_c[0][1] == x["trajectory_size"]:
            return ["tracklets"]
        elif np.any(most_c[:, 1] == 5):
            return ["only detected with tracklets"]
        elif np.all(most_c[:, 1] > 1):
            return ["tracklets_with_trajectories_associations only"]
        elif np.all(most_c[:, 1] == 1):
            return ["observations_associations"]
        else:
            counter = np.array([i for i in counter.values()])

            assoc_dict = list()

            if counter[0] == 1 and counter[1] == 1:
                assoc_dict.append("obs_assoc")
            elif counter[0] > 1 and counter[1] == 1:
                assoc_dict.append("traj_with_new_obs")
            elif counter[0] == 1 and counter[1] > 1:
                assoc_dict.append("old_obs_with_track")
            elif counter[0] > 1 and counter[1] > 1:
                assoc_dict.append("traj_with_track")

            for i in range(2, len(counter)):

                if counter[i - 1] > 1 and counter[i] == 1:
                    assoc_dict.append("traj_with_new_obs")
                elif counter[i - 1] == 1 and counter[i] > 1:
                    assoc_dict.append("traj_with_track")
                elif counter[i - 1] > 1 and counter[i] > 1:
                    assoc_dict.append("traj_with_track")
                elif counter[i - 1] == 1 and counter[i] == 1:
                    assoc_dict.append("traj_with_new_obs")

            return assoc_dict

    def association_stat(df, test_name):
        with pd.option_context("mode.chained_assignment", None):
            df["assoc_type"] = df.apply(detect_tracklets, axis=1)

        assoc_type = Counter(df.explode(["assoc_type"])["assoc_type"])

        data = [v for v in assoc_type.values()]
        labels = [k for k in assoc_type.keys()]

        # define Seaborn color palette to use
        colors = sns.color_palette("pastel")[0:len(data)]
        print(colors)
        # create pie chart
        plt.pie(data, labels=labels, colors=colors, autopct="%.0f%%")
        plt.title("Distribution of the associations done over nights")
        plt.savefig(os.path.join(test_name, "assoc_type"), dpi=500)
        plt.show()

    association_stat(real_mpc_object, test_name)

    # traj_d_size = detected_traj.groupby(["ssnamenr"]).agg(
    #     trajectory_size=("candid", lambda x: len(list(x))),
    #     error=("ssnamenr", lambda x: len(np.unique(x))),
    #     ssnamenr=("ssnamenr", np.unique),
    #     nid=("nid", list),
    #     assoc=("nid", lambda x : Counter(x)),
    #     track=("nid", lambda x : len(np.unique(x)))
    # )

    # print(traj_d_size[["nid", "assoc"]])

    # association_stat(traj_d_size)

    exit()

    stat_df = pd.DataFrame(stat)
    stat_df["night"] = test_night

    plot_stat(stat_df, test_name)

    print("Object analysis")

    mpc_database = get_mpc_database(0)

    detected_mpc = mpc_database[mpc_database["Number"].isin(real_mpc_trajectories)]
    not_detected_mpc = mpc_database[
        mpc_database["Number"].isin(not_detected_object["ssnamenr"])
    ]
    not_seen_mpc = mpc_database[
        mpc_database["Number"].isin(traj_not_observed["ssnamenr"])
    ]

    print()
    print()
    print("Class of objects detected by FAT")
    print(Counter(detected_mpc["Orbit_type"]))

    print()
    print("Class of objects not detected by FAT")
    print(Counter(not_detected_mpc["Orbit_type"]))

    plot_orbit_type(
        detected_mpc, "Asteroids detected by FAT", "e", "eccentricity", True, test_name
    )

    plot_orbit_type(
        detected_mpc, "Asteroids detected by FAT", "i", "inclination", True, test_name
    )

    plot_orbit_type(
        not_detected_mpc,
        "Asteroids not detected by FAT",
        "e",
        "eccentricity",
        True,
        test_name,
    )

    plot_orbit_type(
        not_detected_mpc,
        "Asteroids not detected by FAT",
        "i",
        "inclination",
        True,
        test_name,
    )

    plot_orbit_type(
        not_seen_mpc, "Asteroids not seen by FAT", "e", "eccentricity", True, test_name,
    )

    plot_orbit_type(
        not_seen_mpc, "Asteroids not seen by FAT", "i", "inclination", True, test_name,
    )
