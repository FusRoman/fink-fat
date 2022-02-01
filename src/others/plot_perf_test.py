from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from src.others.utils import load_data
from src.others.utils import get_mpc_database
import json


def plot_orbit_type(orbit_param, title, y, ylabel):
    g = sns.scatterplot(data=orbit_param, x="a", y=y, hue="Orbit_type")

    g.set(xlim=(0, 7))
    g.set(xlabel="semi-major axis (UA)", ylabel=ylabel)
    g.set_title(title)

    plt.show()


if __name__ == "__main__":
    test_name = "perf_test_1"

    df_sso = load_data("Solar System MPC", 0)

    trajectory_df = pd.read_parquet("src/others/perf_test/{}.parquet".format(test_name))

    with open("src/others/perf_test/{}.json".format(test_name), "r") as json_file:
        stat = json.load(json_file)

    test_night = np.unique(trajectory_df["nid"])

    print("number of processed night: {}".format(len(test_night)))

    df_sso = df_sso[df_sso["nid"].isin(test_night)]

    nb_traj = len(np.unique(df_sso["ssnamenr"]))
    print("number of objects in these nights: {}".format(nb_traj))

    traj_size = df_sso.groupby(["ssnamenr"]).count().reset_index()

    detected_traj = traj_size[traj_size["ra"] >= 5]

    print("number of objects that can be detected: {}".format(len(detected_traj)))
    print()

    traj_with_orb = trajectory_df[trajectory_df["a"] != -1.0]

    traj_cand_size = traj_with_orb.groupby(["trajectory_id"]).agg(
        trajectory_size=("candid", lambda x: len(list(x))),
        error=("ssnamenr", lambda x: len(np.unique(x))),
        ssnamenr=("ssnamenr", np.unique),
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
    print("Object analysis")

    mpc_database = get_mpc_database(0)

    detected_mpc = mpc_database[mpc_database["Number"].isin(real_mpc_trajectories)]
    not_detected_mpc = mpc_database[
        mpc_database["Number"].isin(not_detected_object["ssnamenr"])
    ]

    print()
    print()
    print("Class of objects detected by FAT")
    print(Counter(detected_mpc["Orbit_type"]))

    print()
    print("Class of objects not detected by FAT")
    print(Counter(not_detected_mpc["Orbit_type"]))

    plot_orbit_type(detected_mpc, "Asteroids detected by FAT", "e", "eccentricity")

    plot_orbit_type(detected_mpc, "Asteroids detected by FAT", "i", "inclination")

    plot_orbit_type(
        not_detected_mpc, "Asteroids not detected by FAT", "e", "eccentricity"
    )

    plot_orbit_type(
        not_detected_mpc, "Asteroids not detected by FAT", "i", "inclination"
    )
