from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np


def time_to_decimal(time):
    """
    Get the decimal part of a date.

    Parameters
    ----------
    time : array
        a time with hms format split by ':'

    Returns
    -------
    decimal_time : string
        the decimal part of a date.

    Examples
    --------
    >>> time = [20, 17, 40.088]
    >>> time_to_decimal(time)
    '73040'
    """
    return str(int(time[0]) * 3600 + int(time[1]) * 60 + int(time[0]))


def split_string(string, char_split="-"):
    """
    Split char based on char_split

    Parameters
    ----------
    string : string
        string to be split
    char_split : string
        characters used for the spliting

    Returns
    -------
    string splited : list
        a list with all splited members

    Examples
    --------
    >>> test = "a-b-c-d"
    >>> split_string(test)
    ['a', 'b', 'c', 'd']
    """
    return string.split(char_split)


def join_string(list_string, join_string):
    """
    Join string based on join_string

    Parameters
    ----------
    list_string : string list
        list of string to be join
    join_string : string
        characters used for the joining

    Returns
    -------
    string joined : string
        a string where all elements of the list string have been join with join_string between them

    Examples
    --------
    >>> test = ["a", "b", "c", "d"]
    >>> join_string(test, "/")
    'a/b/c/d'
    """
    return join_string.join(list_string)


def concat_date(list_date):
    """
    Concatenation of a date to be conform to mpc format

    Parameters
    ---------
    list_date : string list
        all elements from a date

    Returns
    -------
    string mpc format : string
        a date string in mpc format

    Examples
    --------
    >>> ld = ["1969", "07", "20", ".", "73040"]
    >>> concat_date(ld)
    '1969 07 20.73040'

    >>> ld = ["2021", "11", "03", ".", "4503"]
    >>> concat_date(ld)
    '2021 11 03.45030'

    >>> ld = ["2021", "04", "2", ".", "453"]
    >>> concat_date(ld)
    '2021 04 02.45300'
    """

    first_list = join_string(list_date[-3:], "")

    date_float = format(float(first_list), ".5f").rjust(8, "0")

    return join_string(list_date[:2] + [date_float], " ")


def band_to_str(band):
    """
    Small filter band conversion

    Parameters
    ----------
    band : integer
        the integer designation of the ZTF band filter

    Returns
    -------
    band string : string
        the band designation r or g

    Examples
    --------
    >>> band_to_str(1)
    'g'

    >>> band_to_str(2)
    'r'

    >>> band_to_str(0)
    """
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


def second_letter(i, uppercase=True):
    """
    Create the second letter of the mpc provisional designation.
    Return all alphabet letter based on i but only skip the i.

    Paramters
    ---------
    i : integer
        letter rank in the alphabet
    uppercase : boolean
        if set to true, return the letter in uppercase

    Returns
    -------
    letter : string
        the ith alphabet letter

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
    >>> second_letter(8)
    'H'
    >>> second_letter(10)
    'K'
    """
    if uppercase:
        case = 64
    else:
        case = 96

    if i <= 8:
        return chr(i + case)
    elif i <= 25:
        return chr(i + case + 1)


def left_shift(number, n):
    """
    Left shift on 10 base number.

    Parameters
    ----------
    number : integer
        the number to be shift
    n : integer
        the number of digit to shift

    Returns
    -------
    shifted number : integer
        the number left shifted by n digit

    Examples
    --------
    >>> left_shift(152, 1)
    15
    >>> left_shift(14589, 3)
    14
    """
    return number // 10 ** n


def right_shift(number, n):
    """
    Right shift on 10 base number.

    Parameters
    ----------
    number : integer
        the number to be shift
    n : integer
        the number of digit to shift

    Returns
    -------
    shifted number : integer
        the number right shifted by n digit

    Examples
    --------
    >>> right_shift(123, 1)
    3
    >>> right_shift(0, 1)
    0
    >>> right_shift(1234, 2)
    34
    """
    return number % 10 ** n


def letter_cycle(cycle):
    """
    Return the cycle letter of the provisional designation from the MPC format

    Parameters
    ----------
    cycle : integer
        the number of time that the second letter have cycle throught the alphabet

    Returns
    -------
    second letter : string
        the second letter corresponding to the cycle

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
    r = (cycle - 1) // 25
    if r > 0:
        cycle %= 25

        if cycle == 0:
            cycle = 25

        return second_letter(cycle, uppercase=False)
    else:
        if cycle == 0:
            cycle = 1
        elif cycle <= 8 and cycle >= 1:
            cycle += 1
        return second_letter(cycle)


def make_cycle(cycle):
    """
    Return the last two symbol of the provisional designation corresponding to the cycle.

    Parameters
    ----------
    cycle : integer
        the number of time that the second letter have cycle throught the alphabet

    Returns
    -------
    the cycle : string
        the last two symbol of the provisional designation

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
    Return the provisional designation from mpc standard

    Parameters
    ----------
    time : string
        the reference trajectory time
    discovery_number : integer
        the trajectory identifier

    Returns
    -------
    provisional designation : string
        the provisional designation to assign to this trajectory

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
    """
    Convert date from hmsdms to mpc format

    Parameters
    ---------
    date : string
        hmsdms date format

    Returns
    -------
    mpc date : string
        mpc date format

    Examples
    --------
    >>> date = "20-07-1969 20:17:40.088"
    >>> make_date(date)
    '20 07 1969.73040'
    """
    d = date.split(" ")
    return concat_date(d[0].split("-") + ["."] + [time_to_decimal(d[1].split(":"))])


def write_observation_file(ram_dir, obs_df):
    """
    Write an observation file to mpc standard from a dataframe containing all the observations of one trajectories

    Parameters
    ----------
    ram_dir : string
        the path where to write the file

    obs_df : dataframe
        the observation dataframe
        have to contains the following columns :
            ra, dec, dcmag, fid, jd, trajectory_id

    Returns
    -------
    prov_desig : string
        the provisional designation assign to the trajectory

    Examples
    --------
    >>> os.mkdir("mpcobs")
    >>> test_obs = pd.DataFrame({
    ... "ra" : [0, 1],
    ... "dec": [0, 1],
    ... "dcmag" : [17.4, 17.6],
    ... "fid": [1, 2],
    ... "jd" : [2440423.34352, 2440423.34387],
    ... "trajectory_id" : [0, 0]
    ... })

    >>> write_observation_file("", test_obs)
    'K69O00A'

    >>> filecmp.cmp("mpcobs/K69O00A.obs", "fink_fat/test/K69O00A_test.obs")
    True

    >>> shutil.rmtree("mpcobs")
    """
    obs_df = obs_df.sort_values("jd")
    ra = obs_df["ra"]
    dec = obs_df["dec"]
    dcmag = obs_df["dcmag"]
    band = obs_df["fid"]
    date = obs_df["jd"]
    traj_id = obs_df["trajectory_id"].values[0]

    coord = SkyCoord(ra, dec, unit=u.degree).to_string("hmsdms", precision=2, pad=True)
    translation_rules = {ord(i): " " for i in "hmd"}
    translation_rules[ord("s")] = ""
    coord = [el.translate(translation_rules) for el in coord]

    t = Time(date.astype(np.double), format="jd")
    date = t.iso
    prov_desig = make_designation(date[0], traj_id)

    date = [make_date(d) for d in date]
    res = [join_string([el1] + [el2], " ") for el1, el2 in zip(date, coord)]

    res = [
        "     "
        + prov_desig
        + "  C"  # how the observation was made : C means CCD
        + el
        + "         "
        + str(round(mag, 1))
        + " "
        + band_to_str(b)
        + "      I41"  # ZTF observation code
        for el, mag, b in zip(res, dcmag, band)
    ]

    res[0] = res[0][:12] + "*" + res[0][13:]

    dir_path = ram_dir + "mpcobs/"
    with open(dir_path + prov_desig + ".obs", "wt") as file:
        file.write(join_string(res, "\n"))

    return prov_desig


if __name__ == "__main__":  # pragma: no cover
    import sys
    import doctest
    from pandas.testing import assert_frame_equal  # noqa: F401
    import fink_fat.test.test_sample as ts  # noqa: F401
    from unittest import TestCase  # noqa: F401
    import shutil  # noqa: F401
    import filecmp  # noqa: F401
    import stat  # noqa: F401
    import os  # noqa: F401
    import pandas as pd  # noqa: F401

    if "unittest.util" in __import__("sys").modules:
        # Show full diff in self.assertEqual.
        __import__("sys").modules["unittest.util"]._MAX_LENGTH = 999999999

    sys.exit(doctest.testmod()[0])
