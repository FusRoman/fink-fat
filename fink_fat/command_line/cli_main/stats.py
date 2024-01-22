import numpy as np
import pandas as pd
import os
from collections import Counter
from collections import OrderedDict
import glob

from terminaltables import DoubleTable, AsciiTable, SingleTable

from fink_fat.command_line.stat_cli import print_assoc_table, describe
from fink_fat.command_line.stat_cli import compute_residue, test_detectable
from fink_fat.command_line.utils_cli import get_class


def cli_stats(arguments, config, output_path):
    """
    Print summary statistics about fink-fat using the command line

    Parameters
    ----------
    arguments : dict
        command_line arguments
    config : ConfigParser
        object containing the data from the config file
    output_path : string
        path where are located the fink-fat data
    """
    output_path, object_class = get_class(arguments, output_path)
    tr_df_path = os.path.join(output_path, "trajectory_df.parquet")
    orb_res_path = os.path.join(output_path, "orbital.parquet")
    obs_df_path = os.path.join(output_path, "old_obs.parquet")
    traj_orb_path = os.path.join(output_path, "trajectory_orb.parquet")

    if os.path.exists(tr_df_path):
        trajectory_df = pd.read_parquet(tr_df_path)

        if len(trajectory_df) == 0:
            print("No trajectories detected.")
            exit()

        print(
            "Number of observations, all trajectories candidates combined: {}".format(
                len(trajectory_df)
            )
        )
        print(
            "Number of trajectories candidates: {}".format(
                len(np.unique(trajectory_df["trajectory_id"]))
            )
        )
        gb = trajectory_df.groupby(["trajectory_id"]).count()["ra"]
        c = Counter(gb)
        table_data = [["Size", "Number of trajectories candidates"]]
        table_data += [
            [size, number_size]
            for size, number_size in OrderedDict(sorted(c.items())).items()
        ]
        table_instance = AsciiTable(
            table_data, "Trajectories candidates size distribution"
        )
        table_instance.justify_columns[1] = "right"
        print()
        print(table_instance.table)
        print()

        print_assoc_table(trajectory_df)

    else:
        print(
            "Trajectory file doesn't exist, run 'fink_fat association (mpc | candidates)' to create it."
        )

    if os.path.exists(obs_df_path):
        old_obs_df = pd.read_parquet(obs_df_path)
        print("Number of old observations: {}".format(len(old_obs_df)))
        print()
    else:
        print("No old observations exists.")

    if os.path.exists(orb_res_path) and os.path.exists(traj_orb_path):
        orb_df = pd.read_parquet(orb_res_path)
        traj_orb_df = pd.read_parquet(traj_orb_path)

        if len(orb_df) == 0 or len(traj_orb_df) == 0:
            print("No trajectories with orbital elements found.")
            exit()

    else:
        print("No trajectories with orbital elements found")
        exit()

    # trajectories with orbits size comparation
    trajectories_gb = traj_orb_df.groupby(["trajectory_id"]).agg(
        count=("ra", len), tags=("assoc_tag", list)
    )

    trajectories_size = Counter(trajectories_gb["count"])
    table_data = [["Size", "Number of orbits candidates"]]
    table_data += [
        [size, number_size]
        for size, number_size in OrderedDict(sorted(trajectories_size.items())).items()
    ]
    table_instance = AsciiTable(table_data, "Orbits candidates size distribution")
    table_instance.justify_columns[1] = "right"
    print()
    print(table_instance.table)
    print()

    print_assoc_table(traj_orb_df)

    # orbital type statistics
    orb_stats = describe(
        orb_df[["a", "e", "i", "long. node", "arg. peric", "mean anomaly"]],
        ["median"],
    ).round(decimals=3)

    print("Number of orbit candidates: {}".format(int(orb_stats["a"]["count"])))

    orbit_distrib_data = (
        ("orbital elements", "Metrics", "Values"),
        ("semi-major-axis (AU)", "median", orb_stats["a"]["median"]),
        ("", "min", orb_stats["a"]["min"]),
        ("", "max", orb_stats["a"]["max"]),
        ("eccentricity", "median", orb_stats["e"]["median"]),
        ("", "min", orb_stats["e"]["min"]),
        ("", "max", orb_stats["e"]["max"]),
        ("inclination (degrees)", "median", orb_stats["i"]["median"]),
        ("", "min", orb_stats["i"]["min"]),
        ("", "max", orb_stats["i"]["max"]),
        ("long. node (degrees)", "median", orb_stats["long. node"]["median"]),
        ("", "min", orb_stats["long. node"]["min"]),
        ("", "max", orb_stats["long. node"]["max"]),
        ("arg. peric (degrees)", "median", orb_stats["arg. peric"]["median"]),
        ("", "min", orb_stats["arg. peric"]["min"]),
        ("", "max", orb_stats["arg. peric"]["max"]),
        ("mean anomaly (degrees)", "median", orb_stats["mean anomaly"]["median"]),
        ("", "min", orb_stats["mean anomaly"]["min"]),
        ("", "max", orb_stats["mean anomaly"]["max"]),
    )

    orb_table = SingleTable(orbit_distrib_data, "orbit candidates distribution")
    print()
    print(orb_table.table)
    print()

    main_belt_candidates = orb_df[(orb_df["a"] <= 4.5) & (orb_df["a"] >= 1.7)]
    distant_main_belt = orb_df[orb_df["a"] > 4.5]
    close_asteroids = orb_df[orb_df["a"] < 1.7]
    earth_crosser = close_asteroids[
        (close_asteroids["a"] < 1.7) & (close_asteroids["e"] > 0.1)
    ]
    no_earth_crosser = close_asteroids[
        (close_asteroids["a"] < 1.7) & (close_asteroids["e"] <= 0.1)
    ]

    orbit_type_data = (
        ("Orbit type", "Number of candidates", "Notes"),
        (
            "Main belt",
            len(main_belt_candidates),
            "Main belt asteroids are asteroids with a semi major axis between 1.7 AU and 4.5 AU",
        ),
        (
            "Distant",
            len(distant_main_belt),
            "Distant asteroids are asteroids with a semi major axis greater than 4.5 AU",
        ),
        (
            "Earth crosser",
            len(earth_crosser),
            "An asteroids is considered as an earth crosser when his semi major axis is less than 1.7 and his eccentricity is greater than 0.1",
        ),
        (
            "No earth crosser",
            len(no_earth_crosser),
            "Asteroids with a semi major axis less than 1.7 and an eccentricity less than 0.1",
        ),
    )
    orb_type_table = SingleTable(orbit_type_data, "orbit candidates type")
    print()
    print(orb_type_table.table)
    print()

    # subtraction with the mean of each rms computed with
    # the trajectories from MPC.
    orb_df["rms_dist"] = np.linalg.norm(
        orb_df[
            [
                "rms_a",
                "rms_e",
                "rms_i",
                "rms_long. node",
                "rms_arg. peric",
                "rms_mean anomaly",
            ]
        ].values
        - [0.018712, 0.009554, 0.170369, 0.383595, 4.314636, 3.791175],
        axis=1,
    )

    orb_df["chi_dist"] = np.abs(orb_df["chi_reduced"].values - 1)

    orb_df["score"] = np.linalg.norm(
        orb_df[["rms_dist", "chi_dist"]].values - [0, 0], axis=1
    )

    orb_df = orb_df.sort_values(["score"]).reset_index(drop=True)
    best_orb = orb_df.loc[:9]

    best_orbit_data = [
        [
            "Trajectory id",
            "Orbit ref epoch",
            "a (AU)",
            "error",
            "e",
            "error",
            "i (deg)",
            "error",
            "Long. node (deg)",
            "error",
            "Arg. peri (deg)",
            "error",
            "Mean Anomaly (deg)",
            "error",
            "chi",
            "score",
        ],
    ] + np.around(
        best_orb[
            [
                "trajectory_id",
                "ref_epoch",
                "a",
                "rms_a",
                "e",
                "rms_e",
                "i",
                "rms_i",
                "long. node",
                "rms_long. node",
                "arg. peric",
                "rms_arg. peric",
                "mean anomaly",
                "rms_mean anomaly",
                "chi_reduced",
                "score",
            ]
        ].values,
        3,
    ).tolist()

    best_table = DoubleTable(best_orbit_data, "Best orbit")

    print(best_table.table)
    print("* a: Semi major axis, e: eccentricity, i: inclination")
    print()

    if arguments["mpc"]:
        print()
        path_alert = os.path.join(output_path, "save", "")
        if os.path.exists(path_alert):
            all_path_alert = glob.glob(os.path.join(path_alert, "alert_*"))
            alerts_pdf = pd.DataFrame()
            for path in all_path_alert:
                pdf = pd.read_parquet(path)
                alerts_pdf = pd.concat([alerts_pdf, pdf])

            alerts_pdf["ssnamenr"] = alerts_pdf["ssnamenr"].astype("string")

            gb = (
                alerts_pdf.sort_values(["jd"])
                .groupby(["ssnamenr"])
                .agg(
                    trajectory_size=("candid", lambda x: len(list(x))),
                    nid=("nid", list),
                    diff_night=("nid", lambda x: list(np.diff(list(x)))),
                )
                .reset_index()
            )

            detectable_test = gb["trajectory_size"] >= int(
                config["SOLVE_ORBIT_PARAMS"]["orbfit_limit"]
            )

            trivial_detectable_sso = gb[detectable_test]
            trivial_detectable_sso.insert(
                len(trivial_detectable_sso.columns),
                "detectable",
                trivial_detectable_sso.apply(
                    test_detectable,
                    axis=1,
                    args=(
                        int(config["TW_PARAMS"]["trajectory_keep_limit"]),
                        int(config["SOLVE_ORBIT_PARAMS"]["orbfit_limit"]),
                    ),
                ),
            )

            detectable_sso = trivial_detectable_sso[
                trivial_detectable_sso["detectable"]
            ]

            obs_with_orb = orb_df.merge(traj_orb_df, on="trajectory_id")

            true_cand = (
                obs_with_orb.groupby(["trajectory_id"])
                .agg(
                    error=("ssnamenr", lambda x: len(np.unique(x))),
                    ssnamenr=("ssnamenr", list),
                )
                .reset_index()
                .explode(["ssnamenr"])
            )

            true_orbit = true_cand[true_cand["error"] == 1]

            orb_cand = len(orb_df)
            pure_orb = len(np.unique(true_orbit["trajectory_id"]))
            purity = np.round_((pure_orb / orb_cand) * 100, decimals=2)

            detectable = len(np.unique(detectable_sso["ssnamenr"]))
            detected = len(np.unique(true_orbit["ssnamenr"]))
            efficiency = np.round_((detected / detectable) * 100, decimals=2)

            table_data = (
                ("Metrics", "Values", "Notes"),
                (
                    "True SSO",
                    len(np.unique(alerts_pdf["ssnamenr"])),
                    "Number of solar system objects (SSO) observed by ZTF since the first associations date with fink_fat.",
                ),
                (
                    "Detectable True SSO",
                    detectable,
                    "Number of SSO detectable with fink_fat according to the config file.\n(trajectory_keep_limit={} days / orbfit_limit={} points.".format(
                        config["TW_PARAMS"]["trajectory_keep_limit"],
                        config["SOLVE_ORBIT_PARAMS"]["orbfit_limit"],
                    ),
                ),
                (
                    "Orbit candidates",
                    orb_cand,
                    "Number of orbit detected with fink_fat",
                ),
                (
                    "Pure objects orbit",
                    pure_orb,
                    "Number of orbit candidates that contains only observations of the same SSO.",
                ),
                (
                    "Detected SSO",
                    detected,
                    "Number of unique SSO detected with fink_fat.\n(removes the SSO seen multiple time with fink_fat)",
                ),
                (
                    "Purity",
                    "{} %".format(purity),
                    "ratio between the number of orbit candidates and the number of pure orbits",
                ),
                (
                    "Efficiency",
                    "{} %".format(efficiency),
                    "ratio between the number of detectable sso and the number of detected sso with fink_fat.",
                ),
            )

            table_instance = DoubleTable(table_data, "fink_fat performances")
            table_instance.justify_columns[2] = "right"
            print(table_instance.table)

            if arguments["--mpc-data"] is not None:
                if os.path.exists(arguments["--mpc-data"]):
                    print()
                    print()
                    print("Load mpc database...")
                    mpc_data = pd.read_json(arguments["--mpc-data"])
                    mpc_data["Number"] = mpc_data["Number"].astype("string").str[1:-1]

                    sub_set_mpc = alerts_pdf.merge(
                        mpc_data, left_on="ssnamenr", right_on="Number", how="inner"
                    )

                    detectable_mpc = sub_set_mpc[
                        sub_set_mpc["ssnamenr"].isin(detectable_sso["ssnamenr"])
                    ].drop_duplicates(subset=["ssnamenr"])
                    pure_mpc = sub_set_mpc[
                        sub_set_mpc["ssnamenr"].isin(true_orbit["ssnamenr"])
                    ].drop_duplicates(subset=["ssnamenr"])

                    count_detect_orbit = Counter(detectable_mpc["Orbit_type"])
                    total_orbit = len(detectable_mpc)
                    count_pure_orbit = Counter(pure_mpc["Orbit_type"])
                    table_rows = [
                        ["Orbit type", "Known orbit distribution", "Recovery"]
                    ]
                    for detect_key, detect_value in count_detect_orbit.items():
                        if detect_key in count_pure_orbit:
                            pure_value = count_pure_orbit[detect_key]
                        else:
                            pure_value = 0

                        table_rows.append(
                            [
                                detect_key,
                                "{} % ({})".format(
                                    np.round_(
                                        (detect_value / total_orbit) * 100,
                                        decimals=2,
                                    ),
                                    detect_value,
                                ),
                                "{} % ({})".format(
                                    np.round_(
                                        (pure_value / detect_value) * 100,
                                        decimals=2,
                                    ),
                                    pure_value,
                                ),
                            ]
                        )

                    orbit_type_table = DoubleTable(
                        table_rows, "Orbit type recovery performance"
                    )
                    print()
                    print(orbit_type_table.table)
                    print(
                        "\t*Ratio computed between the detectable object and the pure detected objects with fink_fat."
                    )

                    true_obs = obs_with_orb[
                        obs_with_orb["trajectory_id"].isin(true_cand["trajectory_id"])
                    ]
                    detect_orb_with_mpc = true_obs.drop_duplicates(
                        subset=["trajectory_id"]
                    ).merge(
                        sub_set_mpc,
                        left_on="ssnamenr",
                        right_on="Number",
                        how="inner",
                    )

                    orbital_residue = compute_residue(detect_orb_with_mpc)[
                        ["da", "de", "di", "dNode", "dPeri", "dM"]
                    ]

                    residue_stats = describe(orbital_residue, ["median"]).round(
                        decimals=3
                    )

                    orbit_residue_data = (
                        ("orbital elements", "Metrics", "Values"),
                        (
                            "residue semi-major-axis (AU) (%)",
                            "median",
                            residue_stats["da"]["median"],
                        ),
                        ("", "std", residue_stats["da"]["std"]),
                        ("", "min", residue_stats["da"]["min"]),
                        ("", "max", residue_stats["da"]["max"]),
                        (
                            "residue eccentricity (%)",
                            "median",
                            residue_stats["de"]["median"],
                        ),
                        ("", "std", residue_stats["de"]["std"]),
                        ("", "min", residue_stats["de"]["min"]),
                        ("", "max", residue_stats["de"]["max"]),
                        (
                            "residue inclination (degrees) (%)",
                            "median",
                            residue_stats["di"]["median"],
                        ),
                        ("", "std", residue_stats["di"]["std"]),
                        ("", "min", residue_stats["di"]["min"]),
                        ("", "max", residue_stats["di"]["max"]),
                        (
                            "residue long. node (degrees) (%)",
                            "median",
                            residue_stats["dNode"]["median"],
                        ),
                        ("", "std", residue_stats["dNode"]["std"]),
                        ("", "min", residue_stats["dNode"]["min"]),
                        ("", "max", residue_stats["dNode"]["max"]),
                        (
                            "residue arg. peric (degrees) (%)",
                            "median",
                            residue_stats["dPeri"]["median"],
                        ),
                        ("", "std", residue_stats["dPeri"]["std"]),
                        ("", "min", residue_stats["dPeri"]["min"]),
                        ("", "max", residue_stats["dPeri"]["max"]),
                        (
                            "residue mean anomaly (degrees) (%)",
                            "median",
                            residue_stats["dM"]["median"],
                        ),
                        ("", "std", residue_stats["dM"]["std"]),
                        ("", "min", residue_stats["dM"]["min"]),
                        ("", "max", residue_stats["dM"]["max"]),
                    )

                    residue_table = SingleTable(orbit_residue_data, "orbit residuals")
                    print(residue_table.table)
                    print(
                        "\t*Residues computed between the orbital elements from the pure detected objets and the orbital elements from the mpc database for the corresponding object."
                    )

                else:
                    print()
                    print("The indicated path for the mpc database doesn't exist.")
                    exit()

            print(
                "\t**Reminder: These performance statistics exists as fink_fat has been run in mpc mode."
            )

            exit()
