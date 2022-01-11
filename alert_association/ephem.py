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
import pycurl as pcu
from io import BytesIO
from urllib.parse import urlencode
import xml.etree.ElementTree as ET
import xmltodict

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

    import requests

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
            "-oscelem": "",
            "-mime": "json",
            "-output": "--jd",
            "-from": "MiriadeDoc",
        }

        with open(all_param_path[0], "rb") as fp:
            print(json.load(fp))
            print()
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

        eph = ephem.drop(columns=["RA", "DEC"])
        eph["RA"] = coord.ra.value * 15
        eph["Dec"] = coord.dec.value

        # print(eph)
        # print()
        # print()

        os.remove(path)

    return 0
    buf_data = BytesIO()

    traj_id_list = []
    ra_list = []
    dec_list = []

    for path in all_param_path:
        trajectory_id = int(path.split(".")[0].split("_")[1])

        jd = trajectory_df[trajectory_df["trajectory_id"] == trajectory_id][
            "jd_ephem"
        ].values

        for tmp_jd in jd:

            crl = pcu.Curl()
            crl.setopt(crl.URL,)

            print(tmp_jd)
            data = {
                "-name": "",
                "-type": "",
                "-ep": str(tmp_jd),
                "-nbd": 1,
                "-step": "1d",
                "-tscale": "UTC",
                "-observer": "I41",
                "-theory": "INPOP",
                "-teph": 1,
                "-tcoor": 3,
                "-oscelem": "",
                "-mime": "json",
                "-output": "",
                "-from": "MiriadeDoc",
            }

            pf = urlencode(data)

            print(pf)

            crl.setopt(pcu.POST, 1)
            crl.setopt(pcu.VERBOSE, 1)

            crl.setopt(pcu.POSTFIELDS, pf)

            crl.setopt(pcu.HTTPPOST, [("target", (pcu.FORM_FILE, path,)),])

            crl.setopt(pcu.WRITEDATA, buf_data)

            crl.perform()

            response_data = buf_data.getvalue().decode("UTF-8")
            buf_data.seek(0)
            buf_data.truncate(0)

            tree = ET.fromstring(response_data)
            tr_table = tree[-1][-1][-1][-1][0][0:3]
            print("ephemeris date: {}".format(tr_table[0].text))
            coord = ""
            for child in tr_table[1:]:
                coord += child.text + " "

            print()
            print(coord)
            print()
            print()
            traj_id_list.append(trajectory_id)
            ephemeris_coord = SkyCoord(coord, unit=(u.hourangle, u.deg))

            print(ephemeris_coord)
            print()
            crl.close()
            break

            ra_list.append(ephemeris_coord.ra.degree)
            dec_list.append(ephemeris_coord.dec.degree)

        os.remove(path)

    buf_data.close()

    return pd.DataFrame(
        {
            "trajectory_id": traj_id_list,
            "ephemeris_ra": ra_list,
            "ephemeris_dec": dec_list,
        }
    )


if __name__ == "__main__":
    print("ephem")

    n_trajectories = 1
    n_points = 50

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

    # print(mpc)

    print("MPC DATABASE loading")
    t_before = t.time()
    mpc_database = utils.get_mpc_database()
    print("MPC DATABASE end loading, elapsed time: {}".format(t.time() - t_before))

    print("cross match with mpc database")
    cross_match_mpc = mpc.merge(
        mpc_database, how="inner", left_on="ssnamenr", right_on="Number"
    )

    cross_match_mpc["jd_ephem"] = cross_match_mpc["jd"]

    print(
        cross_match_mpc[
            [
                "ra",
                "dec",
                "jd",
                "jd_ephem",
                "ssnamenr",
                "trajectory_id",
                "a",
                "e",
                "i",
                "Peri",
                "Node",
                "M",
            ]
        ]
    )
    print()
    print()

    ephemeris = generate_ephemeris(cross_match_mpc)
    ephemeris["trajectory_id"] = ephemeris["trajectory_id"].astype(int)

    ephem_and_obs = ephemeris.merge(cross_match_mpc, on="trajectory_id")

    deltaRAcosDEC = (ephem_and_obs["ra"] - ephem_and_obs["ephemeris_ra"]) * np.cos(
        np.radians(ephem_and_obs["dec"])
    )
    deltaDEC = ephem_and_obs["dec"] - ephem_and_obs["ephemeris_dec"]

    print("ephemeris")
    print(
        ephem_and_obs[
            ["ephemeris_ra", "ephemeris_dec", "ra", "dec", "nid", "jd", "jd_ephem"]
        ]
    )
    print(ephem_and_obs["jd"].values)
    print()
    print("ephemeris residuals")
    print(deltaRAcosDEC)
    print()
    print(deltaDEC)

    exit()

    next_obs = mpc[(n_points - 1) :: n_points]

    traj_to_orb = mpc[~mpc["candid"].isin(next_obs["candid"])]
    print(mpc)
    print()
    print()
    print(next_obs)
    print()
    print()
    print(traj_to_orb)
    print()
    print()

    orbital_elem = compute_orbital_element(traj_to_orb)
    orbital_elem = orbital_elem[orbital_elem["a"] != -1.0]
    print("orbital elements")
    print(
        orbital_elem[
            [
                "trajectory_id",
                "ssnamenr",
                "a",
                "e",
                "i",
                "long. node",
                "arg. peric",
                "mean anomaly",
            ]
        ]
    )

    print("MPC DATABASE loading")
    t_before = t.time()
    mpc_database = utils.get_mpc_database()
    print("MPC DATABASE end loading, elapsed time: {}".format(t.time() - t_before))

    print()
    print("cross match with mpc database")
    cross_match_mpc = orbital_elem.merge(
        mpc_database, how="inner", left_on="ssnamenr", right_on="Number"
    )

    df_residue = po.compute_residue(cross_match_mpc)

    print()
    print("orbital elements residuals")
    print(df_residue[["da", "de", "di", "dNode", "dPeri", "dM"]])
    print()
    print()

    next_jd = next_obs[["trajectory_id", "jd"]]
    orbital_elem = orbital_elem.merge(
        next_jd, on="trajectory_id", suffixes=("", "_ephem")
    )

    print(orbital_elem)

    ephemeris = generate_ephemeris(orbital_elem)
    ephemeris["trajectory_id"] = ephemeris["trajectory_id"].astype(int)

    ephem_and_obs = ephemeris.merge(next_obs, on="trajectory_id")

    deltaRAcosDEC = (ephem_and_obs["ra"] - ephem_and_obs["ephemeris_ra"]) * np.cos(
        np.radians(ephem_and_obs["dec"])
    )
    deltaDEC = ephem_and_obs["dec"] - ephem_and_obs["ephemeris_dec"]

    print("ephemeris")
    print(ephem_and_obs)
    print(ephem_and_obs["jd"].values)
    print()
    print("ephemeris residuals")
    print(deltaRAcosDEC)
    print()
    print(deltaDEC)
