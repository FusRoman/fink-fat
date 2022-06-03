from glob import glob
import signal
import numpy as np
import pandas as pd
from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as u
from shutil import copyfile, rmtree
import re
import subprocess
import os
import multiprocessing as mp
from fink_fat import __file__

import traceback
import logging


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

    >>> shutil.rmtree("mpcobs/")
    """
    obs_df = obs_df.sort_values(["trajectory_id", "jd"])
    ra = obs_df["ra"]
    dec = obs_df["dec"]
    dcmag = obs_df["dcmag"]
    band = obs_df["fid"]
    date = obs_df["jd"]
    traj_id = obs_df["trajectory_id"].values[0]

    coord = SkyCoord(ra, dec, unit=u.degree).to_string("hmsdms")
    translation_rules = {ord(i): " " for i in "hmd"}
    translation_rules[ord("s")] = ""
    coord = [el.translate(translation_rules) for el in coord]

    coord = [
        re.sub(r"(\d+)\.(\d+)", lambda matchobj: matchobj.group()[:5], s) for s in coord
    ]

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

    dir_path = ram_dir + "mpcobs/"
    with open(dir_path + prov_desig + ".obs", "wt") as file:
        file.write(join_string(res, "\n"))

    return prov_desig


def write_inp(ram_dir, provisional_designation):
    """
    Write the input file of orbit

    Parameters
    ----------
    ram_dir : string
        path where to write the file

    provisional_designation : string
        the provisional designation of the trajectory

    Returns
    -------
    None

    Examples
    --------
    >>> write_inp("", 'K69O00A')

    >>> with open("K69O00A.inp", "r") as file:
    ...     file.read()
    'K69O00A'

    >>> os.remove("K69O00A.inp")
    """
    with open(ram_dir + provisional_designation + ".inp", "wt") as file:
        file.write(ram_dir + provisional_designation)


def write_oop(ram_dir, provisional_designation):
    """
    Write the option configuration file for the OrbFit computation

    Parameters
    ----------
    ram_dir : string
        path where to write the file
    provisional_designation : string
        the provisional designation of the trajectory

    Returns
    -------
    None

    Examples
    --------
    >>> write_oop("", "K69O00A")

    >>> filecmp.cmp("K69O00A.oop", "fink_fat/test/K69O00A_test.oop")
    True

    >>> os.remove("K69O00A.oop")
    """
    fink_fat_path = os.path.dirname(__file__)
    oop_template = os.path.join(fink_fat_path, "orbit_fitting", "template.oop")

    copyfile(oop_template, ram_dir + provisional_designation + ".oop")
    with open(ram_dir + provisional_designation + ".oop", "a") as file:
        file.write(".filbe=" + ram_dir + "AST17")
        file.write("\noutput_files.\n")
        file.write("\t.elem = " + ram_dir + provisional_designation + ".oel\n")
        file.write("object1.\n")
        file.write("\t.obs_dir = " + ram_dir + "mpcobs\n")
        file.write("\t.obs_fname = " + provisional_designation)


def prep_orbitfit(ram_dir):
    """
    Preparation for OrbFit computation

    Copy the AST17 ephemeris files needed for the orbfit computation to the correct location.
    Set their permissions to be read by OrbFit.

    Parameters
    ----------
    ram_dir : string
        path where to write file

    Returns
    -------

    Examples
    --------

    >>> prep_orbitfit("")

    >>> st = os.stat("AST17.bai")
    >>> stat.filemode(st.st_mode)
    '-rwxrwxrwx'

    >>> st = os.stat("AST17.bep")
    >>> stat.filemode(st.st_mode)
    '-rwxrwxrwx'

    >>> shutil.rmtree("mpcobs")
    >>> os.remove("AST17.bai")
    >>> os.remove("AST17.bep")
    """

    try:
        fink_fat_path = os.path.dirname(__file__)
        orbfit_path = os.path.join(fink_fat_path, "orbit_fitting")
        dir_path = ram_dir + "mpcobs/"

        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)

        copyfile(os.path.join(orbfit_path, "AST17.bai_431_fcct"), ram_dir + "AST17.bai")
        os.chmod(ram_dir + "AST17.bai", 0o777)

        copyfile(os.path.join(orbfit_path, "AST17.bep_431_fcct"), ram_dir + "AST17.bep")
        os.chmod(ram_dir + "AST17.bep", 0o777)
    except Exception:
        logging.error(traceback.format_exc())


def call_orbitfit(ram_dir, provisional_designation):
    """
    Call the OrbFit software in a subprocess. Kill it after 2 second if OrbFit are blocked.

    Parameters
    ----------
    ram_dir : string
        path where to write the file
    provisional_designation : string
        the provisional designation of the trajectory

    Returns
    -------
    output : integer
        return status of the orbfit process

    Examples
    --------

    >>> call_orbitfit("fink_fat/test/call_orbfit/", "K21E00A")

    >>> os.path.exists("fink_fat/test/call_orbfit/K21E00A.oel")
    True

    >>> os.remove("fink_fat/test/call_orbfit/K21E00A.odc")
    >>> os.remove("fink_fat/test/call_orbfit/K21E00A.olg")
    >>> os.remove("fink_fat/test/call_orbfit/K21E00A.pro")
    >>> os.remove("fink_fat/test/call_orbfit/mpcobs/K21E00A.rwo")
    """
    orbitfit_path = os.path.join("~", "OrbitFit", "bin", "")
    command = (
        orbitfit_path
        + "orbfit.x < "
        + ram_dir
        + provisional_designation
        + ".inp "
        + ">/dev/null 2>&1"
    )

    with subprocess.Popen(
        command, shell=True, stdout=subprocess.DEVNULL, preexec_fn=os.setsid
    ) as process:
        try:
            output = process.communicate(timeout=5)[0]
            return output
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGINT)  # send signal to the process group
            output = process.communicate()[0]
            return output


def rm_files(files):
    """
    Remove all files contains in the files parameters

    Parameters
    files : string list
        A list of files path (typically return by the glob library)

    Return
    ------
    None
    """
    for path_f in files:
        os.remove(path_f)


def obs_clean(ram_dir, prov_desig):
    """
    Remove all the temporary file named as prov_desig created during the OrbFit process.

    Parameters
    ----------
    ram_dir : string
        Path where files are located
    prov_desig : string
        the provisional designation of the trajectory that triggered the OrbFit process.

    Returns
    -------
    None

    Examples
    --------

    >>> prov_desig = "A000001"
    >>> open(prov_desig + ".oel", 'a').close()
    >>> open(prov_desig + ".err", 'a').close()

    >>> os.makedirs("mpcobs")
    >>> open("mpcobs/" + prov_desig + ".obs", 'a').close()
    >>> open("mpcobs/" + prov_desig + ".rwo", 'a').close()

    >>> obs_clean("", prov_desig)

    >>> os.rmdir("mpcobs")
    """

    rm_files(glob(ram_dir + prov_desig + ".*"))
    rm_files(glob(ram_dir + "mpcobs/" + prov_desig + ".*"))


def final_clean(ram_dir):
    """
    Remove the residuals files used by OrbFit

    Parameters
    ----------
    ram_dir : string
        Path where files are located

    Returns
    -------
    None

    Examples
    --------
    >>> prep_orbitfit("")

    >>> os.path.exists("AST17.bai")
    True
    >>> os.path.exists("AST17.bep")
    True
    >>> os.path.exists("mpcobs")
    True

    >>> final_clean("")

    >>> os.path.exists("AST17.bai")
    False
    >>> os.path.exists("AST17.bep")
    False
    >>> os.path.exists("mpcobs")
    False
    """
    rm_files(glob(ram_dir + "*.bai"))
    rm_files(glob(ram_dir + "*.bep"))
    rm_files(glob(ram_dir + "*.log"))

    os.rmdir(ram_dir + "mpcobs")


def read_oel(ram_dir, prov_desig):
    """
    Read the .oel file return by orbfit. This file contains the orbital elements, the reference epoch of the orbit computation and
    the rms of the orbital elements

    Parameters
    ----------
    ram_dir : string
        Path where files are located
    prov_desig : string
        the provisional designation of the trajectory that triggered the OrbFit process.

    Returns
    -------
    orb_elem : integer list
        A list with the reference epoch first then the orbital elements and finally the rms.

    Examples
    --------
    >>> read_oel("fink_fat/test/call_orbfit/", "K21E00A")
    [2459274.810893373, '1.5833993623527698E+00', '0.613559993695898', '5.9440877456670', '343.7960539272898', '270.1931234374459', '333.9557366497585', -1, -1, -1, -1, -1, -1]

    >>> read_oel("", "")
    [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]

    >>> read_oel("fink_fat/test/call_orbfit/", "K21H00A")
    [2459345.797868819, '3.1514694062448680E+00', '0.113946062348132', '1.6879159876457', '38.1016474068882', '136.1915246941109', '46.5628893357021', '7.94527E-03', '1.83696E-02', '4.77846E-02', '3.17863E-01', '1.34503E+01', '9.82298E+00']
    """
    try:
        with open(ram_dir + prov_desig + ".oel") as file:
            lines = file.readlines()

            ref_mjd = float(lines[8].strip().split()[1])
            # conversion from modified julian date to julian date
            ref_jd = ref_mjd + 2400000.5

            orb_params = " ".join(lines[7].strip().split()).split(" ")
            if len(lines) > 12:
                rms = " ".join(lines[12].strip().split()).split(" ")
            else:
                rms = [-1, -1, -1, -1, -1, -1, -1, -1]
            return [ref_jd] + orb_params[1:] + rms[2:]
    except FileNotFoundError:
        return list(np.ones(13, dtype=np.float64) * -1)
    except Exception as e:
        print("----")
        print(e)
        print()
        print("ERROR READ OEL FILE: {}".format(prov_desig))
        print()
        print(lines)
        print()
        print()
        logging.error(traceback.format_exc())
        print("----")
        return list(np.ones(13, dtype=np.float64) * -1)


def read_rwo(ram_dir, prov_desig, nb_obs):
    """
    Read the .rwo file return by orbfit. This file contains the observations of the trajectories and the goodness of the fit computed by OrbFit.
    Return the chi values for each observations.

    Parameters
    ----------
    ram_dir : string
        Path where files are located
    prov_desig : string
        the provisional designation of the trajectory that triggered the OrbFit process.

    Returns
    -------
    chi : integer list
        The list of all chi values of each observations.

    Examples
    --------
    """
    try:
        with open(ram_dir + "mpcobs/" + prov_desig + ".rwo") as file:
            lines = file.readlines()

            chi_obs = [obs_l.strip().split(" ")[-3] for obs_l in lines[7:]]

            return np.array(chi_obs).astype(np.float32)
    except FileNotFoundError:
        return list(np.ones(nb_obs, dtype=np.float64) * -1)
    except ValueError:
        return list(np.ones(nb_obs, dtype=np.float64) * -1)
    except Exception as e:
        print("----")
        print(e)
        print()
        print("ERROR READ RWO FILE: {}".format(prov_desig))
        print()
        print(lines)
        print()
        print()
        logging.error(traceback.format_exc())
        print("----")
        return list(np.ones(nb_obs, dtype=np.float64) * -1)


def get_orbit_param(ram_dir, df):
    """
    Compute the orbital elements of one trajectory.

    Parameters
    ----------
    ram_dir : string
        Path where files are located
    df : dataframe
        All the observation of the trajectory. An observation file will be write in the MPC format based on the observations contains in this dataframe.

    Returns
    -------
    results : list
        The list contains in this order : the trajectory identifier, the provisional designation, the reference epoch, the 6 orbital elements and finally
        the rms of the orbital elements.

    Examples
    --------
    >>> df = pd.DataFrame({
    ... 'ra': [169.8604675, 169.8568848, 169.8336664, 169.8297121, 169.8296555],
    ... 'dec': [15.2063604, 15.2103091, 15.2360481, 15.2403893, 15.24049],
    ... 'dcmag': [16.438142098160576, 16.47854604642893, 15.767506616421468, 15.781593431530103, 15.764373749886605],
    ... 'fid': [1, 1, 2, 2, 2],
    ... 'jd': [2459274.7206481, 2459274.7391435, 2459274.8594444, 2459274.8799074, 2459274.8803819],
    ... 'trajectory_id': [0, 0, 0, 0, 0]
    ... })

    >>> prep_orbitfit("")
    >>> get_orbit_param("", df)
    [[0, 'K21E00A', 2459274.810893373, '1.5833993623527698E+00', '0.613559993695898', '5.9440877456670', '343.7960539272898', '270.1931234374459', '333.9557366497585', -1, -1, -1, -1, -1, -1, -1.0]]
    >>> final_clean("")
    """

    all_traj_id = np.unique(df["trajectory_id"].values)

    results = []
    for traj_id in all_traj_id:
        df_one_traj = df[df["trajectory_id"] == traj_id]
        prov_desig = write_observation_file(ram_dir, df_one_traj)
        write_inp(ram_dir, prov_desig)
        write_oop(ram_dir, prov_desig)

        try:
            call_orbitfit(ram_dir, prov_desig)
        except Exception as e:
            print(e)
            print("ERROR CALLING ORBFIT: {}".format(prov_desig))
            print()
            logging.error(traceback.format_exc())
            print()
            print(prov_desig)
            print()
            print()
            print(df_one_traj)

        chi_values = read_rwo(ram_dir, prov_desig, len(df_one_traj))

        # reduced the chi values
        chi_reduced = np.sum(np.array(chi_values)) / len(df_one_traj)

        results.append(
            [traj_id, prov_desig] + read_oel(ram_dir, prov_desig) + [chi_reduced]
        )

        try:
            obs_clean(ram_dir, prov_desig)
        except FileNotFoundError:
            print("ERROR CLEANING ORBFIT: {}".format(prov_desig))
            print(prov_desig)
            print()
            print()
            print(df_one_traj)

    return results


def orbit_elem_dataframe(orbit_elem):
    """
    Convert the list return by get_orbit_param into a dataframe.

    Parameters
    ----------
    orbit_elem : list
        The return of get_orbit_param

    Returns
    -------
    df_orb_elem : dataframe
        the input list convert into a dataframe

    Examples
    --------
    >>> orb_list = [[0, 'K21E00A', 2459274.810893373, '1.5834346988159376E+00', '0.613572037782866', '5.9442185803697', '343.7959802838470', '270.1932521117385', '333.9568546371023', -1, -1, -1, -1, -1, -1, 2.4]]

    >>> orb_df = orbit_elem_dataframe(orb_list)

    >>> assert_frame_equal(orb_df, ts.orb_elem_output)
    """

    column_name = [
        "trajectory_id",
        "provisional designation",
        "ref_epoch",
        "a",
        "e",
        "i",
        "long. node",
        "arg. peric",
        "mean anomaly",
        "rms_a",
        "rms_e",
        "rms_i",
        "rms_long. node",
        "rms_arg. peric",
        "rms_mean anomaly",
        "chi_reduced",
    ]

    df_orb_elem = pd.DataFrame(orbit_elem, columns=column_name,)

    for col_name in set(column_name).difference(set(["provisional designation"])):
        df_orb_elem[col_name] = pd.to_numeric(df_orb_elem[col_name])

    return df_orb_elem


def compute_df_orbit_param(trajectory_df, cpu_count, ram_dir):
    """
    Compute the orbital elements of a set of trajectories. Computation are done in parallel.

    Parameters
    ----------
    trajectory_df : dataframe
        the set of trajectories, the following columns are required : "ra", "dec", "dcmag", "fid", "jd", "trajectory_id"
    cpu_count : integer
        the number of core for the parallel computation
    ram_dir : string
        Path where files are located

    Returns
    -------
    orbit_elem : dataframe
        the orbital elements computed by OrbFit for each inputs trajectories.


    Examples
    --------

    >>> orb_elem = compute_df_orbit_param(ts.orbfit_samples, 2, "")

    >>> assert_frame_equal(orb_elem, ts.orbfit_output)
    """

    prep_orbitfit(ram_dir)

    all_traj_id = np.unique(trajectory_df["trajectory_id"].values)

    trajectory_id_chunks = np.array_split(all_traj_id, cpu_count)

    chunk_ramdir = [
        os.path.join(ram_dir, "chunkid_{}".format(chunk_id), "")
        for chunk_id in np.arange(len(trajectory_id_chunks))
    ]

    for chunk_dir in chunk_ramdir:
        os.mkdir(chunk_dir)
        prep_orbitfit(chunk_dir)

    chunks = [
        (chunk_dir, trajectory_df[trajectory_df["trajectory_id"].isin(tr_chunk)])
        for tr_chunk, chunk_dir in zip(trajectory_id_chunks, chunk_ramdir)
        if len(tr_chunk) > 0
    ]

    pool = mp.Pool(cpu_count)

    results = pool.starmap(get_orbit_param, chunks)
    results = [el2 for el1 in results for el2 in el1]

    pool.close()

    for chunk_dir in chunk_ramdir:
        rmtree(chunk_dir)

    final_clean(ram_dir)

    if len(results) > 0:
        return orbit_elem_dataframe(np.array(results))
    else:
        return pd.DataFrame(
            columns=[
                "trajectory_id",
                "provisional designation",
                "ref_epoch",
                "a",
                "e",
                "i",
                "long. node",
                "arg. peric",
                "mean anomaly",
                "rms_a",
                "rms_e",
                "rms_i",
                "rms_long. node",
                "rms_arg. peric",
                "rms_mean anomaly",
                "chi_reduced",
            ]
        )


if __name__ == "__main__":  # pragma: no cover
    import sys
    import doctest
    from pandas.testing import assert_frame_equal  # noqa: F401
    import fink_fat.test.test_sample as ts  # noqa: F401
    from unittest import TestCase  # noqa: F401
    import shutil  # noqa: F401
    import filecmp  # noqa: F401
    import stat  # noqa: F401

    if "unittest.util" in __import__("sys").modules:
        # Show full diff in self.assertEqual.
        __import__("sys").modules["unittest.util"]._MAX_LENGTH = 999999999

    sys.exit(doctest.testmod()[0])
