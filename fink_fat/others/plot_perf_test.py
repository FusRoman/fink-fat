import time as t
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from fink_fat.others.utils import load_data
from fink_fat.others.utils import get_mpc_database
import json
import os
from astropy.coordinates import SkyCoord
from astropy import units as u


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
    plt.savefig(os.path.join(test_name, "time_plot"))
    plt.close()

    cum_time = np.cumsum(stat_df["time"])
    stat_df["cum_time"] = cum_time

    g = sns.lineplot(data=stat_df, x="night", y="cum_time")
    plt.xticks(rotation=90)
    g.set(xlabel="night identifier", ylabel="time (sec)")
    g.set_title("Cumulative computation time of the associations algorithm over nights")
    plt.savefig(os.path.join(test_name, "cum_time_plot"))
    plt.close()

    g = sns.barplot(data=stat_df, x="night", y="trajectory_size")
    g.set(xlabel="night identifier", ylabel="number of recorded trajectory")
    g.set_title("Size of the recorded trajectory set over nights")
    plt.xticks(rotation=90)
    plt.savefig(os.path.join(test_name, "trajectory_plot"))
    plt.close()


def detect_tracklets(x, traj_time_window, obs_time_window):

    counter = x["assoc"]

    most_c = np.array(counter.most_common())

    most_c = most_c[most_c[:, 0].argsort()]

    if most_c[0][1] == x["trajectory_size"]:
        return ["tracklets"]
    # elif np.any(most_c[:, 1] == orbfit_limit):
    #     return ["only detected with tracklets"]
    # elif np.all(most_c[:, 1] > 1):
    #     return ["tracklets_with_trajectories_associations only"]
    # elif np.all(most_c[:, 1] == 1):
    #     return ["observations_associations only"]
    else:
        counter = np.array([i for i in counter.values()])
        diff_nid = np.diff(np.unique(x["nid"]))

        assoc_dict = list()

        assoc = ""

        if counter[0] == 1 and counter[1] == 1:
            if diff_nid[0] <= obs_time_window:
                assoc = "begin by obs_assoc"
            else:
                assoc = "assoc_not_seen"
        elif counter[0] > 1 and counter[1] == 1:
            if diff_nid[0] <= traj_time_window:
                assoc = "traj_with_new_obs"
            else:
                assoc = "assoc_not_seen"
        elif counter[0] == 1 and counter[1] > 1:
            if diff_nid[0] <= obs_time_window:
                assoc = "old_obs_with_track"
            else:
                assoc = "assoc_not_seen"
        elif counter[0] > 1 and counter[1] > 1:
            if diff_nid[0] <= traj_time_window:
                assoc = "traj_with_track"
            else:
                assoc = "assoc_not_seen"

        assoc_dict.append(assoc)
        pred_assoc = assoc

        for i in range(2, len(counter)):

            if diff_nid[i - 1] <= traj_time_window:
                if counter[i - 1] > 1 and counter[i] == 1:
                    if pred_assoc == "assoc_not_seen":
                        assoc = "assoc_not_seen"
                    else:
                        assoc = "traj_with_new_obs"
                elif counter[i - 1] == 1 and counter[i] > 1:
                    if pred_assoc == "assoc_not_seen":
                        assoc = "old_obs_with_track"
                    else:
                        assoc = "traj_with_track"
                elif counter[i - 1] > 1 and counter[i] > 1:
                    if pred_assoc == "assoc_not_seen":
                        assoc = "assoc_not_seen"
                    else:
                        assoc = "traj_with_track"
                elif counter[i - 1] == 1 and counter[i] == 1:
                    if pred_assoc == "assoc_not_seen":
                        assoc = "begin by obs_assoc"
                    else:
                        assoc = "traj_with_new_obs"
            else:
                assoc = "assoc_not_seen"

            pred_assoc = assoc
            assoc_dict.append(assoc)

        return assoc_dict


def association_stat(
    df, traj_time_window, obs_time_window, test_name, df_name, pie_chart=False
):
    with pd.option_context("mode.chained_assignment", None):
        df["assoc_type"] = df.apply(
            detect_tracklets, axis=1, args=(traj_time_window, obs_time_window,)
        )

    # fmt: off
    begin_by_obs_assoc_percent = (
        np.sum(
            df.apply(
                lambda x: 1 if x["assoc_type"][0] == "begin by obs_assoc" else 0, axis=1
            )
        ) / len(df)
    ) * 100

    print(
        "Number of trajectories beginning by observations associations: {0:10.1f}".format(
            begin_by_obs_assoc_percent
        )
    )

    begin_by_old_obs_assoc_percent = (
        np.sum(
            df.apply(
                lambda x: 1 if x["assoc_type"][0] == "old_obs_with_track" else 0, axis=1
            )
        ) / len(df)
    ) * 100

    # fmt: on

    print(
        "Number of trajectories beginning by associations of old observations with tracklets: {0:10.1f}".format(
            begin_by_old_obs_assoc_percent
        )
    )

    def new_obs_track_prop(rows):
        assoc_type = Counter(rows["assoc_type"])

        # fmt: off
        test = "traj_with_new_obs" in assoc_type and "traj_with_track" in assoc_type and assoc_type["traj_with_track"] != 0
        # fmt: on
        if test:
            return [
                assoc_type["traj_with_new_obs"] / assoc_type["traj_with_track"],
                "ratio",
            ]
        elif "traj_with_new_obs" in assoc_type:
            return [float(assoc_type["traj_with_new_obs"]), "nb_traj_with_new_obs"]
        elif "traj_with_track" in assoc_type:
            return [float(assoc_type["traj_with_track"]), "nb_track_assoc"]
        else:
            return [float(0), "none of them"]

    df[["prop", "type"]] = pd.DataFrame(
        np.array(df.apply(new_obs_track_prop, axis=1).to_list()), index=df.index
    )

    _ = sns.histplot(data=df, x="prop", hue="type", stat="percent")
    plt.savefig(os.path.join(test_name, "ratio_assoc_type"))
    plt.close()

    assoc_type = Counter(df.explode(["assoc_type"])["assoc_type"])
    tt = df.apply(lambda x: np.all(np.array(x["assoc_type"]) == "tracklets"), axis=1)
    print(df[tt])
    if pie_chart:
        data = [v for v in assoc_type.values()]
        labels = [k for k in assoc_type.keys()]

        print(data)

        print(labels)

        # define Seaborn color palette to use
        # fmt: off
        colors = sns.color_palette("pastel")[0:len(data)]
        # fmt: on

        # create pie chart
        plt.pie(data, labels=labels, colors=colors, autopct="%.0f%%")
        plt.title("Distribution of the associations done over nights")
        # plt.show()
        plt.savefig(os.path.join(test_name, df_name))
        plt.close()
    else:
        return assoc_type


def angle(a, b, c):
    ba = b - a
    ca = c - a

    cosine_angle = np.round_(
        np.dot(ba, ca) / (np.linalg.norm(ba) * np.linalg.norm(ca)), decimals=8
    )
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)


def compute_speed(rows):
    ra, dec, jd = rows["ra"], rows["dec"], rows["jd"]

    c = SkyCoord(ra, dec, unit=u.degree)

    diff_jd = np.diff(jd)

    sep = c[0:-1].separation(c[1:]).degree

    velocity = sep / diff_jd

    return velocity


def compute_angle(rows):
    ra, dec, jd = rows["ra"], rows["dec"], rows["jd"]
    all_angle = []

    for i in range(len(ra) - 2):
        a = np.array([ra[i], dec[i]])
        b = np.array([ra[i + 1], dec[i + 1]])
        c = np.array([ra[i + 2], dec[i + 2]])

        jd1 = jd[i + 1]
        jd2 = jd[i + 2]
        diff_jd = jd2 - jd1

        if diff_jd > 0:
            all_angle.append(angle(a, b, c) / diff_jd)
        else:
            all_angle.append(0)

    return all_angle


def rm_list(a):
    np_array = np.array(a["diff_night"])
    np_mask = np.ma.masked_array(np_array, np_array > params["traj_time_window"])
    for i in range(len(np_mask) - 2):
        current = np_mask[0]
        n_next_ = np_mask[i + 2]
        if current is np.ma.masked and n_next_ is np.ma.masked:
            np_mask[i + 1] = np.ma.masked

    not_mask = np.logical_not(np_mask.mask)
    count_consecutif = np.diff(
        np.where(
            np.concatenate(([not_mask[0]], not_mask[:-1] != not_mask[1:], [True]))
        )[0]
    )[::2]

    return np.any(count_consecutif >= params["orbfit_limit"])


if __name__ == "__main__":
    sns.set_context("talk")
    # sns.set(rc={"figure.figsize": (40, 15)})

    test_name = "perf_test_6"
    if not os.path.isdir(test_name):
        os.mkdir(test_name)

    df_sso = load_data("Solar System MPC")

    trajectory_df = pd.read_parquet(
        "fink_fat/others/perf_test/{}.parquet".format(test_name)
    )

    print(trajectory_df[trajectory_df["trajectory_id"] == 4584])

    with open("fink_fat/others/perf_test/{}.json".format(test_name), "r") as json_file:
        stat = json.load(json_file)

    with open(
        "fink_fat/others/perf_test/params_{}.json".format(test_name), "r"
    ) as json_file:
        params = json.load(json_file)

    print("-----------------")
    print()
    print("Parameters of the associations algorithm for this test : ")
    print(params)
    print()
    print("-----------------")

    test_night = np.unique(trajectory_df["nid"])

    test_night = np.append(test_night, test_night[-1] + 1)

    print("number of processed night: {}".format(len(test_night)))

    df_sso = df_sso[df_sso["nid"].isin(test_night)]

    nb_traj = len(np.unique(df_sso["ssnamenr"]))
    print("number of objects in these nights: {}".format(nb_traj))

    traj_size = df_sso.groupby(["ssnamenr"]).count().reset_index()

    # number of object that can be detected in the sso dataset
    detected_traj = traj_size[traj_size["ra"] >= params["orbfit_limit"]]

    detected_object = df_sso[
        df_sso["ssnamenr"].isin(detected_traj["ssnamenr"])
    ].sort_values(["ssnamenr", "jd"])

    traj_can_be_detected = detected_object.groupby(["ssnamenr"]).agg(
        trajectory_size=("candid", lambda x: len(list(x))),
        nid=("nid", list),
        diff_night=("nid", lambda x: list(np.diff(list(x)))),
    )

    traj_can_be_detected["diff_night"] = traj_can_be_detected.apply(rm_list, axis=1)

    detected_traj = traj_can_be_detected[
        traj_can_be_detected["diff_night"]
    ].reset_index()

    # number of object that cannot be detected
    traj_not_observed = traj_size[traj_size["ra"] < params["orbfit_limit"]]

    print("number of objects that can be detected: {}".format(len(detected_traj)))
    print()

    traj_with_orb = trajectory_df[trajectory_df["a"] != -1.0]

    traj_cand_size = (
        traj_with_orb.sort_values(["jd"])
        .groupby(["trajectory_id"])
        .agg(
            trajectory_size=("candid", lambda x: len(list(x))),
            error=("ssnamenr", lambda x: len(np.unique(x))),
            ssnamenr=("ssnamenr", np.unique),
            nid=("nid", list),
            assoc=("nid", lambda x: Counter(x)),
            track=("nid", lambda x: len(np.unique(x))),
        )
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
    print("purity: {0:10.1f}".format((len(true_candidate) / len(traj_cand_size)) * 100))

    print()
    print()
    print("number of mpc object recover: {}".format(len(real_mpc_trajectories)))

    print(
        "efficiency: {0:10.1f}".format(
            (len(real_mpc_trajectories) / len(detected_traj)) * 100
        )
    )

    piece_of_traj = true_candidate.groupby(["ssnamenr"]).count().reset_index()

    print()
    print()
    print("Fragmented trajectories")
    print(piece_of_traj[piece_of_traj["error"] > 1])

    print()
    print()

    real_mpc_object = traj_cand_size[traj_cand_size["error"] == 1]

    detected_sso = df_sso[df_sso["ssnamenr"].isin(detected_traj["ssnamenr"])]

    not_recompute = True
    not_detected_path_data = os.path.join(test_name, "not_detected_traj.parquet")
    detected_path_data = os.path.join(test_name, "detected_traj.parquet")
    # fmt: off
    test = os.path.exists(detected_path_data) and os.path.exists(not_detected_path_data) and not_recompute
    # fmt: on
    if test:
        not_detected_traj = pd.read_parquet(not_detected_path_data)
        detected_traj = pd.read_parquet(detected_path_data)
    else:
        t_before = t.time()
        not_detected_sso = detected_sso[
            ~detected_sso["ssnamenr"].isin(real_mpc_trajectories)
        ]

        detected_mpc = detected_sso[
            detected_sso["ssnamenr"].isin(real_mpc_trajectories)
        ]

        not_detected_traj = (
            not_detected_sso.sort_values(["jd"])
            .groupby(["ssnamenr"])
            .agg(
                trajectory_size=("candid", lambda x: len(list(x))),
                ssnamenr=("ssnamenr", np.unique),
                nid=("nid", list),
                ra=("ra", list),
                dec=("dec", list),
                jd=("jd", list),
                dcmag=("dcmag", list),
            )
        )

        detected_traj = (
            detected_mpc.sort_values(["jd"])
            .groupby(["ssnamenr"])
            .agg(
                trajectory_size=("candid", lambda x: len(list(x))),
                ssnamenr=("ssnamenr", np.unique),
                nid=("nid", list),
                ra=("ra", list),
                dec=("dec", list),
                jd=("jd", list),
                dcmag=("dcmag", list),
            )
        )
        not_detected_traj["speed"] = not_detected_traj.apply(compute_speed, axis=1)
        not_detected_traj["angle"] = not_detected_traj.apply(compute_angle, axis=1)

        detected_traj["speed"] = detected_traj.apply(compute_speed, axis=1)
        detected_traj["angle"] = detected_traj.apply(compute_angle, axis=1)
        print("stat computation elapsed time: {}".format(t.time() - t_before))
        not_detected_traj.to_parquet(not_detected_path_data)
        detected_traj.to_parquet(detected_path_data)

    n_speed_explode = not_detected_traj["speed"].explode(["speed"])
    n_speed_data = n_speed_explode.to_numpy()
    sns.histplot(
        n_speed_data, bins=200, binrange=(0, 1), cbar_kws=dict(alpha=0.5), color="red"
    )

    d_speed_explode = detected_traj["speed"].explode(["speed"])
    d_speed_data = d_speed_explode.to_numpy()
    sns.histplot(
        d_speed_data[d_speed_data < params["sep_criterion"]],
        bins=200,
        binrange=(0, 1),
        cbar_kws=dict(alpha=0.5),
    )
    plt.savefig(os.path.join(test_name, "speed_result"))
    plt.close()

    n_angle_explode = not_detected_traj["angle"].explode(["angle"])
    n_angle_data = n_angle_explode.to_numpy()
    sns.histplot(
        n_angle_data, bins=200, binrange=(0, 8), cbar_kws=dict(alpha=0.5), color="red"
    )

    d_angle_explode = detected_traj["angle"].explode(["angle"])
    d_angle_data = d_angle_explode.to_numpy()
    sns.histplot(
        d_angle_data[d_angle_data < params["angle_criterion"]],
        bins=200,
        binrange=(0, 8),
        cbar_kws=dict(alpha=0.5),
    )
    plt.savefig(os.path.join(test_name, "angle_result"))
    plt.close()

    # assoc_sso = (
    #     detected_sso.sort_values(["jd"])
    #     .groupby(["ssnamenr"])
    #     .agg(
    #         trajectory_size=("candid", lambda x: len(list(x))),
    #         error=("ssnamenr", lambda x: len(np.unique(x))),
    #         ssnamenr=("ssnamenr", np.unique),
    #         nid=("nid", list),
    #         assoc=("nid", lambda x: Counter(x)),
    #         track=("nid", lambda x: len(np.unique(x))),
    #     )
    # )

    # all_assoc_type = {}
    # all_assoc_type["params"] = []
    # all_assoc_type["results"] = []
    # import time as t
    # for i in [5, 10, 15, 20, 30, 40]:
    #     for j in [2, 3, 5, 10]:
    #         print(i, " ", j)
    #         all_assoc_type["params"].append("({},{})".format(i, j))
    #         t_before = t.time()
    #         res = association_stat(assoc_sso, i, j, test_name, "assoc_type_real_tw={}_ow={}".format(i, j))
    #         print(t.time() - t_before)
    #         all_assoc_type["results"].append(res)
    #         print()
    #         print()

    # all_possible_key = [
    #     "tracklets",
    #     "tracklets not seen",
    #     "only detected with tracklets",
    #     "tracklets_with_trajectories_associations only",
    #     "observations_associations only",
    #     "begin by obs_assoc",
    #     "assoc_not_seen",
    #     "old_obs_with_track",
    #     "traj_with_track",
    #     "traj_with_new_obs"
    # ]

    # label_color = dict()
    # nb_data = np.arange(len(all_assoc_type["results"]))
    # proc_data = dict()

    # for counter in all_assoc_type["results"]:
    #     for key in all_possible_key:
    #         color = label_color.setdefault(key, np.random.random(size=3))
    #         l = proc_data.setdefault(key, [])
    #         if key in counter:
    #             proc_data[key].append(counter[key])
    #         else:
    #             proc_data[key].append(0)

    # # print(proc_data)
    # data_df = pd.DataFrame(proc_data).reset_index(drop=True)

    # data_df = data_df.divide(data_df.sum(axis=1), axis=0)# .multiply(100)

    # g = sns.lineplot(data=data_df)
    # g.set_xlabel("time windows parameters")
    # g.set_ylabel("number of associations types (percentage)")
    # # g.set_xticklabels(all_assoc_type["params"])
    # # plt.tight_layout()
    # plt.show()
    # plt.close()

    # best windows parameters : 15 for the trajectories and 2 for the observations

    # association_stat(
    #     real_mpc_object,
    #     params["traj_time_window"],
    #     params["obs_time_window"],
    #     test_name,
    #     "assoc_type_candidates",
    #     True,
    # )

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

    # stat_df = pd.DataFrame(stat)

    # stat_df["night"] = np.arange(len(stat_df))

    # plot_stat(stat_df, test_name)

    # exit()

    # print("Object analysis")

    # mpc_database = get_mpc_database(0)

    # print(mpc_database.info())

    # mpc_reduce = mpc_database[["a", "e", "i", "Number", "Orbit_type"]]

    # detected_real_mpc = mpc_reduce[mpc_reduce["Number"].isin(detected_traj["ssnamenr"])]
    # detected_real_mpc["detected"] = True
    # not_detected_real_mpc = mpc_reduce[
    #     mpc_reduce["Number"].isin(not_detected_traj["ssnamenr"])
    # ]
    # not_detected_real_mpc["detected"] = False
    # synthesis_mpc_result = pd.concat([detected_real_mpc, not_detected_real_mpc])

    # g = sns.scatterplot(
    #     data=synthesis_mpc_result,
    #     x="a",
    #     y="e",
    #     hue="Orbit_type",
    #     markers=["v", "o"],
    #     style="detected",
    # )
    # g.set(xlim=(0, 7))
    # plt.show()
