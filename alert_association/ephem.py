import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import time as t

from astropy.coordinates import SkyCoord
import astropy.units as u

import alert_association.orbit_fitting.plot_orbstat as po
import alert_association.orbit_fitting.orbfit_management as om
import alert_association.utils as utils
import multiprocessing as mp
from alert_association.utils import load_data
import json
import glob
import os

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

        dynamical_parameters["ref_epoch"] = rows["jd"]
        dynamical_parameters["semi_major_axis"] = rows["a"]
        dynamical_parameters["eccentricity"] = rows["e"]
        dynamical_parameters["inclination"] = rows["i"]

        dynamical_parameters["node_longitude"] = rows["Node"]
        dynamical_parameters["perihelion_argument"] = rows["Peri"]
        dynamical_parameters["mean_anomaly"] = rows["M"]

        # dynamical_parameters["node_longitude"] = rows["long. node"]
        # dynamical_parameters["perihelion_argument"] = rows["arg. peric"]
        # dynamical_parameters["mean_anomaly"] = rows["mean anomaly"]

        dict_param["dynamical_parameters"] = dynamical_parameters

        with open(
            os.path.join(ram_dir, "@aster_{}.json".format(rows["trajectory_id"])), "w"
        ) as outfile:
            json.dump(dict_param, outfile, indent=4)


def generate_ephemeris(trajectory_df):

    url = "https://ssp.imcce.fr/webservices/miriade/api/ephemcc.php"
    write_target_json(trajectory_df)

    all_param_path = glob.glob(os.path.join(ram_dir, "@aster_*.json"))

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

        with open(all_param_path[0], "rb") as fp:
            print(json.load(fp))
            print()
            print()

        for epoch in jd:
            print(epoch, end=", ")

        print()
        files = {
            "target": open(all_param_path[0], "rb").read(),
            "epochs": ("epochs", "\n".join(["%.6f" % epoch for epoch in jd])),
        }

        r = requests.post(url, params=params, files=files, timeout=2000)

        j = r.json()
        ephem = pd.DataFrame.from_dict(j["data"])
        print(ephem)
        print(ephem.info())
        print()
        print()
        coord = SkyCoord(ephem["RA"], ephem["DEC"], unit=(u.deg, u.deg))

        ephem["cRA"] = coord.ra.value * 15
        ephem["cDec"] = coord.dec.value

        print(ephem)
        print()
        print()

        os.remove(path)

    return ephem


if __name__ == "__main__":
    print("ephem")


    print("Load sso data")
    df_sso = load_data("Solar System MPC", nb_indirection=0).reset_index(drop=True)

    one_obs = df_sso[df_sso["ssnamenr"] == "795"].reset_index(drop=True).loc[0]

    dynamical_parameters = dict()

    dynamical_parameters["ref_epoch"] = 2459274.6455671
    dynamical_parameters["semi_major_axis"] = 2.7497334
    dynamical_parameters["eccentricity"] = 0.1020056
    dynamical_parameters["inclination"] = 19.05051

    dynamical_parameters["node_longitude"] = 17.35846
    dynamical_parameters["perihelion_argument"] = 190.01857
    dynamical_parameters["mean_anomaly"] = 336.48578

    dict_param = dict()
    dict_param["type"] = "Asteroid"
    dict_param["dynamical_parameters"] = dynamical_parameters

    print("Write json orbital parameter on disk")

    json_orb_elem_path = "aster_{}.json".format(one_obs["ssnamenr"])

    with open(
            json_orb_elem_path, "w"
        ) as outfile:
            json.dump(dict_param, outfile, indent=4)


    url = "https://ssp.imcce.fr/webservices/miriade/api/ephemcc.php"
    params = {
            "-name": "",
            "-type": "Asteroid",
            "-ep": 2459274.6455671,
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
            "target": open(json_orb_elem_path, "rb").read()
        }

    print("Ephemeris request")

    r = requests.post(url, params=params, files=files, timeout=2000)

    j = r.json()

    ephem = pd.DataFrame.from_dict(j["data"])    
    coord = SkyCoord(ephem["RA"], ephem["DEC"], unit=(u.deg, u.deg))

    ephem["cRA"] = coord.ra.value * 15
    ephem["cDec"] = coord.dec.value
    
    deltaRAcosDEC = (one_obs["ra"] - ephem["cRA"]) * np.cos(
        np.radians(one_obs["dec"])
    )
    deltaDEC = one_obs["dec"] - ephem["cDec"]

    colors = ['#15284F', '#F5622E']
    
    fig, ax = plt.subplots(
    figsize=(10, 10),
    sharex=True,
    )

    ax.scatter(one_obs['ra'], one_obs['dec'], label='ZTF', alpha=0.5, color=colors[1])

    ax.plot(ephem['cRA'], ephem["cDec"], ls='', color='black', marker='x', alpha=0.5, label='Ephemerides')
    ax.legend(loc='best')
    ax.set_xlabel('RA ($^o$)')
    ax.set_ylabel('DEC ($^o$)')

    axins = ax.inset_axes([0.2, 0.2, 0.45, 0.45])

    axins.plot( deltaRAcosDEC, deltaDEC, ls='', color=colors[0], marker='x', alpha=0.8)
    axins.errorbar( np.mean(deltaRAcosDEC), np.mean(deltaDEC), xerr=np.std(deltaRAcosDEC), yerr=np.std(deltaDEC) )
    axins.axhline(0, ls='--', color='black')
    axins.axvline(0, ls='--', color='black')
    axins.set_xlabel(r'$\Delta$RA ($^o$)')
    axins.set_ylabel(r'$\Delta$DEC ($^o$)')

    plt.show()
    os.remove(json_orb_elem_path)