import logging
import os
import signal
import sys
import subprocess
import traceback
import numpy as np

from glob import glob
import pandas as pd
from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as u
from shutil import copyfile
import re

from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import *  # noqa: F403
from pyspark.sql import functions as F  # noqa: F401
from pyspark.sql import SparkSession  # noqa: F401


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

    Paramters
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


def prep_orbfit(ram_dir):
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

    orbfit_path = os.path.join("/opt", "OrbitFit", "tests", "bineph", "testout")
    dir_path = ram_dir + "mpcobs/"
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)

    copyfile(os.path.join(orbfit_path, "AST17.bai_431_fcct"), ram_dir + "AST17.bai")
    os.chmod(ram_dir + "AST17.bai", 0o777)

    copyfile(os.path.join(orbfit_path, "AST17.bep_431_fcct"), ram_dir + "AST17.bep")
    os.chmod(ram_dir + "AST17.bep", 0o777)


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
    """
    rm_files(glob(ram_dir + "*.bai"))
    rm_files(glob(ram_dir + "*.bep"))
    rm_files(glob(ram_dir + "*.log"))

    os.rmdir(ram_dir + "mpcobs")


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
    """

    rm_files(glob(ram_dir + prov_desig + ".*"))
    rm_files(glob(ram_dir + "mpcobs/" + prov_desig + ".*"))


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
    """

    with open(ram_dir + provisional_designation + ".inp", "wt") as file:
        file.write(ram_dir + provisional_designation)


def write_observation_file(ram_dir, ra, dec, dcmag, band, date, traj_id):
    """
    Write an observation file to mpc standard from a dataframe containing all the observations of one trajectories
    This version of the function is used by the cluster mode.

    Parameters
    ----------
    ram_dir : string
        the path where to write the file

    ra : spark dataframe columns
        Right ascension columns of the observations
    dec : spark dataframe columns
        Declination columns of the observations
    dcmag : spark dataframe columns
        The apparent magnitude computed with the dc_mag function in fink_science.conversion
    band : spark dataframe columns
        The filter used during the observations
    date : spark dataframe columns
        The observations date in jd
    traj_id : spark dataframe columns
        The trajectory_id of each trajectories

    Returns
    -------
    prov_desig : string
        the provisional designation assign to the trajectory
    """

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


def write_oop(ram_dir, provisional_designation):
    """
    Write the option configuration file for the OrbFit computation in cluster mode.

    Parameters
    ----------
    ram_dir : string
        path where to write the file
    provisional_designation : string
        the provisional designation of the trajectory

    Returns
    -------
    None
    """

    with open(ram_dir + provisional_designation + ".oop", "w") as file:
        # write output options
        file.write("output.\n")
        file.write("\t.elements = 'KEP'\n")

        # write operations options
        file.write("operations.\n")
        file.write("\t.init_orbdet = 2\n")
        file.write("\t.diffcor = 2\n")
        file.write("\t.ident = 0\n")
        file.write("\t.ephem = 0\n")

        # write error model options
        file.write("error_model.\n")
        file.write("\t.name='fcct14'\n")

        # write additional options
        file.write("IERS.\n")
        file.write("\t.extrapolation = .T.\n")

        # write reject options
        file.write("reject.\n")
        file.write("\t.rejopp = .FALSE.\n")

        # write propagation options
        file.write("propag.\n")
        file.write("\t.iast = 17\n")
        file.write("\t.npoint = 600\n")
        file.write("\t.dmea = 0.2d0\n")
        file.write("\t.dter = 0.05d0\n")

        # write location files options
        file.write(".filbe=" + ram_dir + "AST17\n")
        file.write("\noutput_files.\n")
        file.write("\t.elem = " + ram_dir + provisional_designation + ".oel\n")
        file.write("object1.\n")
        file.write("\t.obs_dir = " + ram_dir + "mpcobs\n")
        file.write("\t.obs_fname = " + provisional_designation)


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
    """

    orbitfit_path = os.path.join("/opt", "OrbitFit", "bin/")

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
            return [float(i) for i in [ref_jd] + orb_params[1:] + rms[2:]]
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
        dir_path = ram_dir + "mpcobs/"
        with open(dir_path + prov_desig + ".rwo", "r") as file:
            lines = file.readlines()

            chi_obs = [obs_l.strip().split(" ")[-3] for obs_l in lines[7:]]

            return np.array(chi_obs).astype(np.float32)
    except FileNotFoundError:
        return np.ones(nb_obs, dtype=np.float64) * -1
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
        return np.ones(nb_obs, dtype=np.float64) * -1


def orbit_wrapper(ra, dec, dcmag, band, date, traj_id, ram_dir):
    """
    Computation of OrbFit in cluster mode.

    Parameters
    ----------
    ra : spark dataframe columns
        Right ascension columns of the observations
    dec : spark dataframe columns
        Declination columns of the observations
    dcmag : spark dataframe columns
        The apparent magnitude computed with the dc_mag function in fink_science.conversion
    band : spark dataframe columns
        The filter used during the observations
    date : spark dataframe columns
        The observations date in jd
    traj_id : spark dataframe columns
        The trajectory_id of each trajectories
    ram_dir : string
        the path where to write the file

    Return
    ------
    res : Series
        The orbital elements and their RMS computed by orbfit for each trajectories.
    """

    @pandas_udf(ArrayType(DoubleType()))  # noqa: F405
    def get_orbit_element(ra, dec, dcmag, band, date, traj_id):
        _pid = os.getpid()
        current_ram_path = os.path.join(ram_dir, str(_pid), "")
        if not os.path.isdir(current_ram_path):
            os.mkdir(current_ram_path)

        prep_orbfit(current_ram_path)

        res = []
        for c_ra, c_dec, c_dcmag, c_band, c_date, c_traj_id in zip(
            ra, dec, dcmag, band, date, traj_id
        ):
            prov_desig = write_observation_file(
                current_ram_path, c_ra, c_dec, c_dcmag, c_band, c_date, c_traj_id
            )
            write_inp(current_ram_path, prov_desig)
            write_oop(current_ram_path, prov_desig)

            call_orbitfit(current_ram_path, prov_desig)
            orb_elem = read_oel(current_ram_path, prov_desig)

            chi_values = read_rwo(current_ram_path, prov_desig, len(c_ra))
            # reduced the chi values
            chi_reduced = np.sum(np.array(chi_values)) / len(c_ra)

            res.append(orb_elem + [chi_reduced])

            obs_clean(current_ram_path, prov_desig)

        final_clean(current_ram_path)
        os.rmdir(current_ram_path)
        return pd.Series(res)

    return get_orbit_element(ra, dec, dcmag, band, date, traj_id)


if __name__ == "__main__":
    ram_dir = "/tmp/ramdisk/"
    master_adress = sys.argv[1]

    spark = spark = (
        SparkSession.builder.master(master_adress)
        .appName("Fink-FAT_solve_orbit")
        .getOrCreate()
    )

    # read the input from local parquet file
    traj_df = pd.read_parquet("tmp_traj.parquet")
    # transform the local pandas dataframe into a spark dataframe
    sparkDF = spark.createDataFrame(traj_df)

    spark_gb = (
        sparkDF.groupby("trajectory_id")
        .agg(
            F.sort_array(
                F.collect_list(F.struct("jd", "ra", "dec", "fid", "dcmag"))
            ).alias("collected_list")
        )
        .withColumn("ra", F.col("collected_list.ra"))
        .withColumn("dec", F.col("collected_list.dec"))
        .withColumn("fid", F.col("collected_list.fid"))
        .withColumn("dcmag", F.col("collected_list.dcmag"))
        .withColumn("jd", F.col("collected_list.jd"))
        .drop("collected_list")
    )

    max_core = int(dict(spark.sparkContext.getConf().getAll())["spark.cores.max"])
    spark_gb = spark_gb.repartition(max_core * 100)

    print("begin compute orbital elem on spark")
    spark_column = spark_gb.withColumn(
        "orbital_elements",
        orbit_wrapper(
            spark_gb.ra,
            spark_gb.dec,
            spark_gb.dcmag,
            spark_gb.fid,
            spark_gb.jd,
            spark_gb.trajectory_id,
            ram_dir,
        ),
    )

    orb_pdf = spark_column.toPandas()
    orb_pdf.to_parquet("res_orb.parquet")
