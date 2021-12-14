from astropy import coordinates
import numpy as np
import pandas as pd
from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as u
import csv
import re


def time_to_decimal(time):
    return str(int(time[0]) * 3600 + int(time[1]) * 60 + int(time[0]))


def split_string(string):
    return string.split("-")


def join_string(list_string, join_string):
    return join_string.join(list_string)


def concat_date(list_date):
    first_list = join_string(list_date[:-2], " ")
    return join_string([first_list] + list_date[-2:], "")


def band_to_str(band):
    if band == 1:
        return "g"
    elif band == 2:
        return "r"


def write_observation_file(filename, obs_df):
    obs_df = obs_df.sort_values(["jd"])
    ra = obs_df["ra"]
    dec = obs_df["dec"]
    dcmag = obs_df["dcmag"]
    band = obs_df["fid"]
    date = obs_df["jd"]

    coord = SkyCoord(ra, dec, unit=u.degree).to_string("hmsdms")
    translation_rules = {ord(i): " " for i in "hmd"}
    translation_rules[ord("s")] = ""
    coord = [el.translate(translation_rules) for el in coord]

    coord = [
        re.sub(r"(\d+)\.(\d+)", lambda matchobj: matchobj.group()[:5], s) for s in coord
    ]

    t = Time(date, format="jd")
    date = t.iso
    date = [a.split(" ") for a in date]

    date = [
        concat_date(el[0].split("-") + ["."] + [time_to_decimal(el[1].split(":"))])
        for el in date
    ]
    res = [join_string([el1] + [el2], " ") for el1, el2 in zip(date, coord)]
    res = [
        "     K21X05F  C"
        + el
        + "         "
        + str(round(mag, 1))
        + " "
        + band_to_str(b)
        + "      I41"
        for el, mag, b in zip(res, dcmag, band)
    ]

    with open(filename + ".obs", "wt") as file:
        file.write(join_string(res, "\n"))


if __name__ == "__main__":
    path = "../../data/month=03"
    df_sso = pd.read_pickle(path)

    # test1 : 2010ET42

    print(df_sso.groupby(['ssnamenr']).agg({"candid":len}).sort_values(['candid']))

    mpc_name = "2010ET42"
    mpc_name = "2936"
    mpc = df_sso[df_sso["ssnamenr"] == mpc_name]

    write_observation_file(mpc_name, mpc)
