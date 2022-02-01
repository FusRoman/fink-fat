import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import time as t

from astropy.coordinates import SkyCoord
import astropy.units as u

import src.orbit_fitting.orbfit_management as om

# import src.utils as utils
import multiprocessing as mp
from src.others.utils import load_data
import json
import glob
import os
from src.orbit_fitting.orbfit_management import compute_df_orbit_param

# constant to locate the ram file system
ram_dir = "/media/virtuelram/"


def compute_orbital_element(traj_df):

    orbit_elem = om.compute_df_orbit_param(traj_df, int(mp.cpu_count() - 2), ram_dir)
    return traj_df.merge(orbit_elem, on="trajectory_id")


def write_target_json(trajectory_df):
    tr_gb = trajectory_df.groupby(["trajectory_id"]).first().reset_index()

    for _, rows in tr_gb.iterrows():
        dict_param = dict()
        dict_param["type"] = "Asteroid"
        dynamical_parameters = dict()

        dynamical_parameters["ref_epoch"] = rows["ref_epoch"]
        dynamical_parameters["semi_major_axis"] = rows["a"]
        dynamical_parameters["eccentricity"] = rows["e"]
        dynamical_parameters["inclination"] = rows["i"]

        # dynamical_parameters["node_longitude"] = rows["Node"]
        # dynamical_parameters["perihelion_argument"] = rows["Peri"]
        # dynamical_parameters["mean_anomaly"] = rows["M"]

        dynamical_parameters["node_longitude"] = rows["long. node"]
        dynamical_parameters["perihelion_argument"] = rows["arg. peric"]
        dynamical_parameters["mean_anomaly"] = rows["mean anomaly"]

        dict_param["dynamical_parameters"] = dynamical_parameters

        with open(
            os.path.join(ram_dir, "@aster_{}.json".format(rows["trajectory_id"])), "w"
        ) as outfile:
            json.dump(dict_param, outfile, indent=4)


def generate_ephemeris(trajectory_df):

    url = "https://ssp.imcce.fr/webservices/miriade/api/ephemcc.php"
    write_target_json(trajectory_df)

    all_param_path = glob.glob(os.path.join(ram_dir, "@aster_*.json"))

    all_ephem = []

    for path in all_param_path:

        trajectory_id = int(path.split(".")[0].split("_")[1])
        jd = trajectory_df[trajectory_df["trajectory_id"] == trajectory_id][
            "jd_ephem"
        ].values

        params = {
            "-name": "",
            "-type": "Asteroid",
            "-tscale": "UTC",
            "-observer": "I41",
            "-theory": "INPOP",
            "-teph": 1,
            "-tcoor": 5,
            "-oscelem": "MPCORB",
            "-mime": "json",
            "-output": "--jd",
            "-from": "MiriadeDoc",
        }

        files = {
            "target": open(path, "rb").read(),
            "epochs": ("epochs", "\n".join(["%.6f" % epoch for epoch in jd])),
        }

        r = requests.post(url, params=params, files=files, timeout=2000)

        j = r.json()
        ephem = pd.DataFrame.from_dict(j["data"])

        coord = SkyCoord(ephem["RA"], ephem["DEC"], unit=(u.deg, u.deg))

        ephem["cRA"] = coord.ra.value * 15
        ephem["cDec"] = coord.dec.value
        ephem["trajectory_id"] = trajectory_id
        all_ephem.append(ephem)

        os.remove(path)

    return pd.concat(all_ephem)


if __name__ == "__main__":
    print("ephem")

    n_trajectories = 30
    n_points = 3

    print("Load sso data")
    df_sso = load_data("Solar System MPC", nb_indirection=0)

    gb_ssn = df_sso.groupby(["ssnamenr"]).agg({"candid": len}).sort_values(["candid"])
    all_track = gb_ssn[gb_ssn["candid"] == n_points].reset_index()["ssnamenr"].values
    mpc = df_sso[df_sso["ssnamenr"].isin(all_track[:n_trajectories])][
        ["ra", "dec", "dcmag", "fid", "jd", "nid", "candid", "ssnamenr"]
    ]
    all_ssnamenr = np.unique(mpc["ssnamenr"].values)
    ssnamenr_translate = {
        ssn: i for ssn, i in zip(all_ssnamenr, range(len(all_ssnamenr)))
    }
    mpc["trajectory_id"] = mpc.apply(
        lambda x: ssnamenr_translate[x["ssnamenr"]], axis=1
    )
    mpc["ssnamenr"] = mpc["ssnamenr"].astype("string")

    mpc = mpc.sort_values(["ssnamenr", "jd"]).reset_index(drop=True)

    t_before = t.time()
    orbit_results = compute_df_orbit_param(mpc, 10, ram_dir)

    multiprocess_time = t.time() - t_before
    print("total multiprocessing orbfit time: {}".format(multiprocess_time))

    mpc_orb = mpc.merge(orbit_results, on="trajectory_id")
    mpc_orb["jd_ephem"] = mpc["jd"]

    # print("MPC DATABASE loading")
    # t_before = t.time()
    # mpc_database = utils.get_mpc_database()
    # print("MPC DATABASE end loading, elapsed time: {}".format(t.time() - t_before))

    # print("cross match with mpc database")
    # cross_match_mpc = mpc.merge(
    #     mpc_database, how="inner", left_on="ssnamenr", right_on="Number"
    # )

    # cross_match_mpc["jd_ephem"] = cross_match_mpc["jd"]

    # print(
    #     cross_match_mpc[
    #         [
    #             "ra",
    #             "dec",
    #             "jd",
    #             "jd_ephem",
    #             "ssnamenr",
    #             "trajectory_id",
    #             "a",
    #             "e",
    #             "i",
    #             "Peri",
    #             "Node",
    #             "M",
    #         ]
    #     ]
    # )
    # print()
    # print()

    ephemeris = generate_ephemeris(mpc_orb)

    mpc_orb["jd"] = np.around(mpc_orb["jd"].values, decimals=6)

    ephemeris["trajectory_id"] = ephemeris["trajectory_id"].astype(int)

    ephem_and_obs = ephemeris.merge(
        mpc_orb, left_on=("trajectory_id", "Date"), right_on=("trajectory_id", "jd")
    )

    # fmt: off
    deltaRAcosDEC = (ephem_and_obs["ra"] - ephem_and_obs["cRA"]) * np.cos(np.radians(ephem_and_obs["dec"])) * 3600
    # fmt: on

    deltaDEC = (ephem_and_obs["dec"] - ephem_and_obs["cDec"]) * 3600

    colors = ["#15284F", "#F5622E"]

    fig, ax = plt.subplots(figsize=(10, 10), sharex=True,)

    fig.suptitle(
        "Trajectories / ephemeris : {} trajectories of {} points".format(
            n_trajectories, n_points
        )
    )

    ax.scatter(
        ephem_and_obs["ra"],
        ephem_and_obs["dec"],
        label="ZTF",
        alpha=0.2,
        color=colors[1],
    )

    ax.plot(
        ephem_and_obs["cRA"],
        ephem_and_obs["cDec"],
        ls="",
        color="black",
        marker="x",
        alpha=0.2,
        label="Ephemerides",
    )
    ax.legend(loc="best")
    ax.set_xlabel("RA ($^o$)")
    ax.set_ylabel("DEC ($^o$)")

    axins = ax.inset_axes([0.2, 0.2, 0.45, 0.45])

    axins.plot(deltaRAcosDEC, deltaDEC, ls="", color=colors[0], marker="x", alpha=0.5)
    axins.errorbar(
        np.mean(deltaRAcosDEC),
        np.mean(deltaDEC),
        xerr=np.std(deltaRAcosDEC),
        yerr=np.std(deltaDEC),
    )
    axins.axhline(0, ls="--", color="black")
    axins.axvline(0, ls="--", color="black")
    axins.set_xlabel(r"$\Delta$RA ($^{\prime\prime}$)")
    axins.set_ylabel(r"$\Delta$DEC ($^{\prime\prime}$)")

    plt.show()
