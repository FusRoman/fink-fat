import pandas as pd
import numpy as np
import os
from sklearn.neighbors import NearestNeighbors
from fink_fat.orbit_fitting.orbfit_local import prep_orbitfit
from fink_fat.orbit_fitting.orbfit_local import write_observation_file
from fink_fat.orbit_fitting.orbfit_local import rm_files


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

import fink_fat.orbit_fitting.orbfit_local as ol
import glob

import time as t

def write_inp(ram_dir, first_designation, second_designation):
    with open(ram_dir + first_designation + "_" + second_designation + ".inp", "wt") as file:
        file.write(ram_dir + first_designation + "_" + second_designation)

def write_oop(ram_dir, first_designation, second_designation):
    with open(ram_dir + first_designation + "_" + second_designation + ".oop", "w") as file:
        # write output options
        file.write("output.\n")
        file.write("\t.elements = 'KEP'\n")

        # write operations options
        file.write("operations.\n")
        file.write("\t.init_orbdet = 2\n")
        file.write("\t.diffcor = 2\n")
        file.write("\t.ident = 2\n")
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
        file.write("\t.elem = " + ram_dir + first_designation + "_" + second_designation + ".oel\n")
        file.write("object1.\n")
        file.write("\t.obs_dir = " + ram_dir + "mpcobs\n")
        file.write("\t.name = " + first_designation)

        # write second object location
        file.write("\nobject2.\n")
        file.write("\t.obs_dir = " + ram_dir + "mpcobs\n")
        file.write("\t.name = " + second_designation)


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
    prov_desig = ol.make_designation(date[0], traj_id)

    date = [ol.make_date(d) for d in date]
    res = [ol.join_string([el1] + [el2], " ") for el1, el2 in zip(date, coord)]
    res = [
        "     "
        + prov_desig
        + "  C"  # how the observation was made : C means CCD
        + el
        + "         "
        + str(round(mag, 1))
        + " "
        + ol.band_to_str(b)
        + "      I41"  # ZTF observation code
        for el, mag, b in zip(res, dcmag, band)
    ]

    res[0] = res[0][:12] + '*' + res[0][13:]

    dir_path = ram_dir + "mpcobs/"
    with open(dir_path + prov_desig + ".obs", "wt") as file:
        file.write(ol.join_string(res, "\n"))

    return prov_desig


def call_orbitfit(ram_dir, first_designation, second_designation):
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
        + first_designation + "_" + second_designation
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


def read_oel(ram_dir, first_desig, second_desig):
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
        with open(ram_dir + first_desig + "_" + second_desig + ".oel") as file:
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
        print("ERROR READ OEL FILE: {}".format(first_desig + "_" + second_desig))
        print()
        print(lines)
        print()
        print()
        logging.error(traceback.format_exc())
        print("----")
        return list(np.ones(13, dtype=np.float64) * -1)


def detect_ident(ram_dir, first_desig, second_desig):
    
    try:
        with open(ram_dir + first_desig + "_" + second_desig + ".olg") as file:
            lines = file.readlines()

            try:
                for i in range(len(lines)):
                    if first_desig + "=" + second_desig in lines[i] and "Differential correction" in lines[i] and lines[i+1].strip() != "FAILED":

                        orb_res = []

                        for j in range(i+2, i+8):

                            numeric_const_pattern = r"""
                            [-+]? # optional sign
                            (?:
                                (?: \d* \. \d+ ) # .1 .12 .123 etc 9.1 etc 98.1 etc
                                |
                                (?: \d+ \.? ) # 1. 12. 123. etc 1 12 123 etc
                            )
                            # followed by optional exponent part if desired
                            (?: [Ee] [+-]? \d+ ) ?
                            """

                            rx = re.compile(numeric_const_pattern, re.VERBOSE)

                            get_orb_str = rx.findall(lines[j])[0]
                            orb_res.append(float(get_orb_str))

                        ref_mjd = float(rx.findall(lines[i+8])[0])
                        # conversion from modified julian date to julian date
                        ref_jd = ref_mjd + 2400000.5

                        orb_res.append(ref_jd)

                        return orb_res
            except Exception:
                return list(np.ones(7, dtype=np.float64) * -1)
            
            return list(np.ones(7, dtype=np.float64) * -1)
    
    except FileNotFoundError:
        return list(np.ones(7, dtype=np.float64) * -1)



def parallel_merger(ram_dir, trajectory_df, orb_cand, indices):

    res_orb = []

    for i in range(len(indices)):
        neighbor_traj = list(orb_cand.iloc[indices[i]]["trajectory_id"])

        prep_orbitfit(ram_dir)

        first_traj = trajectory_df[trajectory_df["trajectory_id"] == neighbor_traj[0]]

        first_obj = write_observation_file(ram_dir, first_traj)

        for other_traj in neighbor_traj[1:]:

            second_traj = trajectory_df[trajectory_df["trajectory_id"] == other_traj]

            second_obj = write_observation_file(ram_dir, second_traj)

            write_inp(ram_dir, first_obj, second_obj)
            write_oop(ram_dir, first_obj, second_obj)

            call_orbitfit(ram_dir, first_obj, second_obj)

            res_orb.append([neighbor_traj[0], other_traj] + detect_ident(ram_dir, first_obj, second_obj))

        rm_files(glob.glob(os.path.join(ram_dir, "*.err")))
        rm_files(glob.glob(os.path.join(ram_dir, "*.inp")))
        rm_files(glob.glob(os.path.join(ram_dir, "*.oop")))
        rm_files(glob.glob(os.path.join(ram_dir, "*.pro")))
        rm_files(glob.glob(os.path.join(ram_dir, "*.odc")))
    
    return res_orb


def merge_orbit(ram_dir, orbit_candidate, observations, nb_neighbors, cpu_count):
    orb_features = np.array(orbit_candidate[["a", "e", "i"]])

    nbrs = NearestNeighbors(n_neighbors=nb_neighbors, algorithm='ball_tree').fit(orb_features)

    _, indices = nbrs.kneighbors(orb_features[:100])

    print("K nearest neighbor ...")

    trajectory_id_chunks = np.array_split(indices, cpu_count)

    chunk_ramdir = [
        os.path.join(ram_dir, "chunkid_{}".format(chunk_id), "")
        for chunk_id in np.arange(len(trajectory_id_chunks))
    ]

    for chunk_dir in chunk_ramdir:
        os.mkdir(chunk_dir)
        prep_orbitfit(chunk_dir)

    chunks = [
        (chunk_dir, observations, orbit_candidate, tr_chunk)
        for tr_chunk, chunk_dir in zip(trajectory_id_chunks, chunk_ramdir)
        if len(tr_chunk) > 0
    ]

    print("orbit trajectories merging ...")

    pool = mp.Pool(cpu_count)

    results = pool.starmap(parallel_merger, chunks)

    print("orbit parallel merging ended ...")

    for chunk_dir in chunk_ramdir:
        rmtree(chunk_dir)

    results = [el2 for el1 in results for el2 in el1]
    
    column_name = [
        "trajectory_id_1",
        "trajectory_id_2",
        "a",
        "e",
        "i",
        "long. node",
        "arg. peric",
        "mean anomaly",
        "ref_epoch"
    ]

    print("creation of the dataframe")

    df_orb_elem = pd.DataFrame(results, columns=column_name,)

    print("end of the orbit merging")

    return df_orb_elem[df_orb_elem["a"] != -1.0]


def remove_mirror(pdf):
    return pdf.loc[pd.DataFrame(np.sort(pdf[['trajectory_id_1','trajectory_id_2']],1),index=pdf.index).drop_duplicates(keep='first').index].sort_values(["trajectory_id_2"])


def remove_transitive(pdf):
    transitive_left = pdf["trajectory_id_1"].isin(pdf["trajectory_id_2"])

    transitive_id = pdf[pdf["trajectory_id_2"].isin(pdf["trajectory_id_1"])]["trajectory_id_1"].values

    pdf.loc[transitive_left, "trajectory_id_1"] = transitive_id

    return pdf

def merge_obs_id(pdf_obs, pdf_traj_merge):
    second_traj = pdf_obs[pdf_obs["trajectory_id"].isin(pdf_traj_merge["trajectory_id_2"])]

    second_traj_size = second_traj.groupby(["trajectory_id"]).count()["ra"].to_numpy()

    tr_id_repeat = np.repeat(pdf_traj_merge["trajectory_id_1"].to_numpy(), second_traj_size)

    pdf_obs.loc[pdf_obs["trajectory_id"].isin(pdf_traj_merge["trajectory_id_2"]), "trajectory_id"] = tr_id_repeat

    return pdf_obs

def merge_orb_id(pdf_orb, pdf_traj_merge):

    print(pdf_orb)

    print(len(np.unique(pdf_orb["trajectory_id"])))

    print(pdf_traj_merge)

    merge_orb_id = pdf_orb["trajectory_id"].isin(pdf_traj_merge["trajectory_id_2"])

    pdf_orb.loc[merge_orb_id, "trajectory_id"] = pdf_traj_merge["trajectory_id_1"].to_numpy()
    pdf_orb.loc[merge_orb_id, "ref_epoch"] = pdf_traj_merge["ref_epoch"].to_numpy()

    pdf_orb.loc[merge_orb_id, ["a", "e", "i", "long. node", "arg. peric", "mean anomaly"]] = pdf_traj_merge[["a", "e", "i", "long. node", "arg. peric", "mean anomaly"]].to_numpy()

    print(len(np.unique(pdf_orb["trajectory_id"])))

    print(pdf_orb.drop_duplicates(["trajectory_id"]))

if __name__ == "__main__":

    ram_dir = "/tmp/ramdisk/"

    path_data = "~/Documents/Doctorat/Asteroids/asteroids_candidates_resultats/candidates"

    obs_cand = pd.read_parquet(os.path.join(path_data, "trajectory_orb.parquet")).sort_values(["trajectory_id"])

    orbit_candidate = pd.read_parquet(os.path.join(path_data, "orbital.parquet")).reset_index(drop=True)

    # # traj id : 12, 31571

    t_before = t.time()
    merge_results = merge_orbit(ram_dir, orbit_candidate, obs_cand, 5, 12)
    print(t.time() - t_before)

    merge_results.to_parquet("merge_traj.parquet")

    # merge_traj = pd.read_parquet("merge_traj.parquet").reset_index(drop=True)

    # print(merge_traj)

    # merge_traj = remove_mirror(merge_traj)

    # print(merge_traj)

    # merge_traj = remove_transitive(merge_traj)

    # obs_cand = merge_obs_id(obs_cand, merge_traj)

    # merge_orb_id(orbit_candidate, merge_traj)

    exit()

    print()

    remove_mirror = merge_traj.loc[pd.DataFrame(np.sort(merge_traj[['trajectory_id_1','trajectory_id_2']],1),index=merge_traj.index).drop_duplicates(keep='first').index].sort_values(["trajectory_id_2"])

    print(remove_mirror)

    print()

    transitive_boolean = remove_mirror["trajectory_id_1"].isin(remove_mirror["trajectory_id_2"])

    print(remove_mirror[transitive_boolean])

    print()

    transitive_id = remove_mirror[remove_mirror["trajectory_id_2"].isin(remove_mirror["trajectory_id_1"])]["trajectory_id_1"].values

    print(transitive_id)

    remove_mirror.loc[transitive_boolean, "trajectory_id_1"] = transitive_id

    print()

    print(remove_mirror)

    print()
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(obs_cand.loc[obs_cand["trajectory_id"].isin(remove_mirror["trajectory_id_2"])]["trajectory_id"])

    second_traj = obs_cand[obs_cand["trajectory_id"].isin(remove_mirror["trajectory_id_2"])]

    second_traj_size = second_traj.groupby(["trajectory_id"]).count()["ra"].to_numpy()

    tr_id_repeat = np.repeat(remove_mirror["trajectory_id_1"].to_numpy(), second_traj_size)

    print()
    print(tr_id_repeat)

    # obs_cand have to be sorted by trajectory_id and remove_mirror have to be sorted by trajectory_id_2

    print(len(obs_cand.loc[obs_cand["trajectory_id"].isin(remove_mirror["trajectory_id_2"])]))

    print(len(tr_id_repeat))

    obs_cand.loc[obs_cand["trajectory_id"].isin(remove_mirror["trajectory_id_2"]), "trajectory_id"] = tr_id_repeat

    print()

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):

        print()
        print()

        print(obs_cand.groupby(["trajectory_id"]).count().sort_values(["ra"]))

    exit()

    # print()
    # print()
    # i = 2
    # print(distances[i])
    # print(indices[i])
    # print(orb_cand.iloc[indices[i]])

    for i in range(len(indices)):
        neighbor_traj = list(orb_cand.iloc[indices[i]]["trajectory_id"])

        prep_orbitfit(ram_dir)

        first_traj = obs_cand[obs_cand["trajectory_id"] == neighbor_traj[0]]

        first_obj = write_observation_file(ram_dir, first_traj)

        print("compute orbit det for: {}".format(first_obj))

        for other_traj in neighbor_traj[1:]:
            print("----------")

            second_traj = obs_cand[obs_cand["trajectory_id"] == other_traj]

            second_obj = write_observation_file(ram_dir, second_traj)

            print("current second obj: {}".format(second_obj))

            f_orb = orb_cand[orb_cand["trajectory_id"] == neighbor_traj[0]]
            s_orb = orb_cand[orb_cand["trajectory_id"] == other_traj]

            print("initial orbit: \nfirst_obj:\n\ta: {}\n\te: {}\n\ti:{}\nsecond_obj:\n\ta: {}\n\te: {}\n\ti: {}".format(
                f_orb["a"].values, f_orb["e"].values, f_orb["i"].values,
                s_orb["a"].values, s_orb["e"].values, s_orb["i"].values
            ))

            write_inp(ram_dir, first_obj, second_obj)
            write_oop(ram_dir, first_obj, second_obj)

            call_orbitfit(ram_dir, first_obj, second_obj)

            print(detect_ident(ram_dir, first_obj, second_obj))

        print("#########################")
        print()

        rm_files(glob.glob(os.path.join(ram_dir, "*.err")))
        rm_files(glob.glob(os.path.join(ram_dir, "*.inp")))
        rm_files(glob.glob(os.path.join(ram_dir, "*.oop")))
        rm_files(glob.glob(os.path.join(ram_dir, "*.pro")))
        rm_files(glob.glob(os.path.join(ram_dir, "*.odc")))

    



