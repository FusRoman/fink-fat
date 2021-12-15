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


def split_string(string, char_split="-"):
    return string.split(char_split)


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

half_month_letter = {
    "01": ["A", "B"],
    "02": ["C", "D"],
    "03": ["E", "F"],
    "04": ["G", "H"],
    "05": ["J", "K"],
    "06": ["L", "M"],
    "07": ["N", "O"],
    "08": ["P", "Q"],
    "09": ["R", "S"],
    "10": ["T", "U"],
    "11": ["V", "W"],
    "12": ["X", "Y"],
}

def second_letter(i, lowercase=True):
    """
    Examples
    --------
    >>> second_letter(1)
    'A'
    >>> second_letter(25)
    'Z'
    >>> second_letter(1, False)
    'a'
    >>> second_letter(25, False)
    'z'
    >>> second_letter(9)
    'J'
    >>> second_letter(9, False)
    'j'
    """
    if lowercase:
        case = 64
    else:
        case = 96

    if i <= 8:
        return chr(i + case)
    elif i <= 25:
        return chr(i + case+1)

def left_shift(number, n):
    """
    Examples
    --------
    >>> left_shift(152, 1)
    15
    >>> left_shift(14589, 3)
    14
    """
    return number // 10**n

def right_shift(number, n):
    """
    Examples
    --------
    >>> right_shift(123, 1)
    3
    >>> right_shift(0, 1)
    0
    >>> right_shift(1234, 2)
    34
    """
    return number % 10**n

def letter_cycle(cycle):
    """
    Examples
    --------
    >>> letter_cycle(10)
    'A'
    >>> letter_cycle(19)
    'J'
    >>> letter_cycle(35)
    'Z'
    >>> letter_cycle(44)
    'j'
    >>> letter_cycle(60)
    'z'
    """
    cycle -= 10
    r = (cycle-1) // 25
    if r > 0:
        cycle %= 25

        if cycle == 0:
            cycle = 25

        return second_letter(cycle, lowercase=False)
    else:
        if cycle == 0:
            cycle = 1
        elif cycle <= 8 and cycle >= 1:
            cycle += 1
        return second_letter(cycle)

def make_cycle(cycle):
    """
    Examples
    --------
    >>> make_cycle(0)
    '00'
    >>> make_cycle(10)
    '10'
    >>> make_cycle(108)
    'A8'
    >>> make_cycle(110)
    'B0'
    >>> make_cycle(127)
    'C7'
    >>> make_cycle(162)
    'G2'
    >>> make_cycle(193)
    'J3'
    >>> make_cycle(348)
    'Y8'
    >>> make_cycle(355)
    'Z5'
    >>> make_cycle(360)
    'a0'
    >>> make_cycle(418)
    'f8'
    >>> make_cycle(439)
    'h9'
    >>> make_cycle(440)
    'j0'
    """
    if cycle <= 9:
        return "0" + str(cycle)
    elif cycle <= 99:
        return str(cycle)
    else:
        digit = left_shift(cycle, 1)
        unit = right_shift(cycle, 1)
        return str(letter_cycle(digit)) + str(unit)

def make_designation(time, discovery_number):
    """
    Examples
    --------
    >>> make_designation("2021-05-22 07:33:02.111", 0)
    'K21K00A'
    >>> make_designation("2021-05-22 07:33:02.111", 24)
    'K21K00Z'
    >>> make_designation("2021-05-22 07:33:02.111", 25)
    'K21K01A'
    >>> make_designation("2021-05-22 07:33:02.111", 49)
    'K21K01Z'
    >>> make_designation("2021-05-22 07:33:02.111", 50)
    'K21K02A'
    >>> make_designation("2021-05-22 07:33:02.111", 8999)
    'K21KZ9Z'
    >>> make_designation("2021-05-22 07:33:02.111", 9000)
    'K21Ka0A'
    >>> make_designation("2021-01-01 07:33:02.111", 0)
    'K21A00A'
    >>> make_designation("2022-07-04 07:33:02.111", 0)
    'K22N00A'
    """
    time_split = time.split(" ")[0].split("-")
    year = time_split[0][-2:]

    half_month = half_month_letter[time_split[1]]
    if int(time_split[2]) <= 15:
        half_month = half_month[0]
    else:
        half_month = half_month[1]
    
    order = discovery_number % 25 + 1
    cycle = int(discovery_number / 25)
    return "K" + year + half_month + make_cycle(cycle) + second_letter(order)

def make_date(date):
    d = date.split(" ")
    return concat_date(d[0].split("-") + ["."] + [time_to_decimal(d[1].split(":"))])

def write_observation_file(obs_df):
    obs_df = obs_df.sort_values(["trajectory_id", "jd"])
    ra = obs_df["ra"]
    dec = obs_df["dec"]
    dcmag = obs_df["dcmag"]
    band = obs_df["fid"]
    date = obs_df["jd"]
    traj_id = obs_df['trajectory_id'].values[0]

    coord = SkyCoord(ra, dec, unit=u.degree).to_string("hmsdms")
    translation_rules = {ord(i): " " for i in "hmd"}
    translation_rules[ord("s")] = ""
    coord = [el.translate(translation_rules) for el in coord]

    coord = [
        re.sub(r"(\d+)\.(\d+)", lambda matchobj: matchobj.group()[:5], s) for s in coord
    ]

    t = Time(date, format="jd")
    date = t.iso
    prov_desig = make_designation(date[0], traj_id)

    date = [
        make_date(d)
        for d in date
    ]
    res = [join_string([el1] + [el2], " ") for el1, el2 in zip(date, coord)]
    res = [
        "     "
        + prov_desig
        + "  C" # how the observation was made : C means CCD
        + el
        + "         "
        + str(round(mag, 1))
        + " "
        + band_to_str(b)
        + "      I41" # ZTF observation code
        for el, mag, b in zip(res, dcmag, band)
    ]

    with open(prov_desig + ".obs", "wt") as file:
        file.write(join_string(res, "\n"))


if __name__ == "__main__":
    path = "../../data/month=03"
    df_sso = pd.read_pickle(path)

    # test1 : 2010ET42

    # gb_ssn = df_sso.groupby(['ssnamenr']).agg({"candid":len}).sort_values(['candid'])
    # mpc_name = "2010ET42"
    import time as t
    mpc_name = ["2936", "2010ET42", "19285"]
    mpc = df_sso[df_sso["ssnamenr"].isin(mpc_name)][["ra", "dec", "dcmag", "fid", "jd", "ssnamenr"]]
    all_ssnamenr = np.unique(mpc['ssnamenr'].values)
    ssnamenr_translate = {ssn : i for ssn, i in zip(all_ssnamenr, range(len(all_ssnamenr)))}
    mpc['trajectory_id'] = mpc.apply(lambda x : ssnamenr_translate[x['ssnamenr']], axis=1)
    all_traj_id = np.unique(mpc['trajectory_id'])
    for traj_id in all_traj_id:
        current_mpc = mpc[mpc['trajectory_id'] == traj_id]
        t_before = t.time()
        write_observation_file(current_mpc)
        print(t.time() - t_before)

    import doctest
    doctest.testmod()[0]
