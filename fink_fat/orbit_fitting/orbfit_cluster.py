import os
import signal
import subprocess
import numpy as np

import pandas as pd
from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as u
from shutil import copyfile
import re
from fink_fat.orbit_fitting.orbfit_local import (
    band_to_str,
    final_clean,
    join_string,
    make_date,
    make_designation,
    obs_clean,
    read_oel,
    write_inp,
)

from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import *  # noqa: F403
from pyspark.sql import functions as F  # noqa: F401
from pyspark.sql import SparkSession  # noqa: F401


def prep_orbfit(ram_dir):
    orbfit_path = os.path.join("/opt", "OrbitFit", "tests", "bineph", "testout")
    dir_path = ram_dir + "mpcobs/"
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)

    copyfile(os.path.join(orbfit_path, "AST17.bai_431_fcct"), ram_dir + "AST17.bai")
    os.chmod(ram_dir + "AST17.bai", 0o777)

    copyfile(os.path.join(orbfit_path, "AST17.bep_431_fcct"), ram_dir + "AST17.bep")
    os.chmod(ram_dir + "AST17.bep", 0o777)


def write_observation_file(ram_dir, ra, dec, dcmag, band, date, traj_id):
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


def orbit_wrapper(ra, dec, dcmag, band, date, traj_id, ram_dir):
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

            res.append(orb_elem)

            obs_clean(current_ram_path, prov_desig)

        final_clean(current_ram_path)
        os.rmdir(current_ram_path)
        return pd.Series(res)

    return get_orbit_element(ra, dec, dcmag, band, date, traj_id)
