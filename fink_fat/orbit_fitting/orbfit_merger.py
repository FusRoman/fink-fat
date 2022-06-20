import pandas as pd
import numpy as np
import os
from sklearn.neighbors import NearestNeighbors


from glob import glob
from shutil import rmtree
import re
import multiprocessing as mp

import traceback
import logging

import fink_fat.orbit_fitting.orbfit_local as ol
import fink_fat.orbit_fitting.orbfit_files as of
import fink_fat.orbit_fitting.mpcobs_files as mf


def detect_ident(ram_dir, first_desig, second_desig):
    """
    Read the .olg files return by orbfit. The .olg files is the log files of OrbFit. It contains the information if the orbit identification
    have been a success and the keplerian orbital elements of the merged trajectories.

    Parameters
    ----------
    ram_dir : string
        Path where the temporary files of Orbfit are written.
    first_desig : string
        the provisional designation of the first arc.
    second_desig : string
        the provisional designation of the second arc.
    """
    try:
        with open(ram_dir + first_desig + "_" + second_desig + ".olg") as file:
            lines = file.readlines()

            try:
                for i in range(len(lines)):
                    if (
                        first_desig + "=" + second_desig in lines[i]
                        and "Differential correction" in lines[i]
                        and lines[i + 1].strip() != "FAILED"
                    ):

                        orb_res = []

                        for j in range(i + 2, i + 8):

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

                        ref_mjd = float(rx.findall(lines[i + 8])[0])
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
    """
    Function used for the parallel computation of Orbfit in orbit identification mode.
    Try all the combination of trajectories return by the nearest neighbor algorithm.

    Paramters
    ---------
    ram_dir : string
        Path where the temporary files of Orbfit are written.
    trajectory_df : dataframe
        The set of observations of each trajectories
    orb_cand : dataframe
        The set of orbital elements of each trajectories
    indices : integer
        Indices of the orb_cand parameters used to recover the neighborhood of a trajectories

    Return
    ------
    res_orb : dataframe
        The results of the orbit identification. The dataframe contains the trajectories of the both arcs involved in the merge and
        the orbitals elements of the new merged trajectories.
    """
    res_orb = []

    # for each trajectories in the current chunk
    for i in range(len(indices)):
        neighbor_traj = list(orb_cand.iloc[indices[i]]["trajectory_id"])

        of.prep_orbitfit(ram_dir)

        first_traj = trajectory_df[trajectory_df["trajectory_id"] == neighbor_traj[0]]

        first_obj = mf.write_observation_file(ram_dir, first_traj)

        # for each trajectories in the neighborhood of the ith trajectories
        for other_traj in neighbor_traj[1:]:

            second_traj = trajectory_df[trajectory_df["trajectory_id"] == other_traj]

            second_obj = mf.write_observation_file(ram_dir, second_traj)

            of.write_inp(ram_dir, first_obj, second_obj)
            of.write_oop(ram_dir, first_obj, second_obj)

            ol.call_orbitfit(ram_dir, first_obj, second_obj)

            res_orb.append(
                [neighbor_traj[0], other_traj]
                + detect_ident(ram_dir, first_obj, second_obj)
            )

        of.rm_files(glob.glob(os.path.join(ram_dir, "*.err")))
        of.rm_files(glob.glob(os.path.join(ram_dir, "*.inp")))
        of.rm_files(glob.glob(os.path.join(ram_dir, "*.oop")))
        of.rm_files(glob.glob(os.path.join(ram_dir, "*.pro")))
        of.rm_files(glob.glob(os.path.join(ram_dir, "*.odc")))

    return res_orb


def merge_orbit(observations, orbit_candidate, ram_dir, nb_neighbors, cpu_count):
    """
    Call OrbFit with the orbit identification mode activated.
    OrbFit is call with an observations files containing two sets of observations, one for the first orbital arcs and one for the second.
    Orbfit compute then an orbit for the two arcs and try to match a single orbit for the two arcs.
    If it succeed, this function return the orbital elements of the orbit belonging to both arcs.

    To merge all the trajectories, we should try all the trajectories combination. It means O(len(observations))^2. To reduce the computation time,
    a nearest neighbor algorithm is applied with the first three orbital elements (a, e, i) as features. Only the nearest neighborhood of a trajectories
    are tested with OrbFit.

    Parameters
    ----------
    observations : dataframe
        The observations of each trajectories
    orbit_candidate : dataframe
        The set of orbital elements of each trajectories
    ram_dir : string
        Path where the temporary files produced by orbfit are written.
    nb_neighbors : integer
        The number of trajectories in the nearest neighborhood of all trajectories. Increase this parameters will increase the computation time.
    cpu_count : integer
        The number of cpu core used for the parallel computation

    Return
    ------
    df_orb_elem : dataframe
        The set of orbital parameters of the merged trajectories. It contains also the trajectory_id of both merged trajectories.
    """
    orb_features = np.array(orbit_candidate[["a", "e", "i"]])

    nbrs = NearestNeighbors(n_neighbors=nb_neighbors, algorithm="ball_tree").fit(
        orb_features
    )

    _, indices = nbrs.kneighbors(orb_features)

    trajectory_id_chunks = np.array_split(indices, cpu_count)

    chunk_ramdir = [
        os.path.join(ram_dir, "chunkid_{}".format(chunk_id), "")
        for chunk_id in np.arange(len(trajectory_id_chunks))
    ]

    for chunk_dir in chunk_ramdir:
        os.mkdir(chunk_dir)
        of.prep_orbitfit(chunk_dir)

    chunks = [
        (chunk_dir, observations, orbit_candidate, tr_chunk)
        for tr_chunk, chunk_dir in zip(trajectory_id_chunks, chunk_ramdir)
        if len(tr_chunk) > 0
    ]

    pool = mp.Pool(cpu_count)

    results = pool.starmap(parallel_merger, chunks)

    for chunk_dir in chunk_ramdir:
        rmtree(chunk_dir)

    of.rm_files(glob.glob("*.rwo"))

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
        "ref_epoch",
    ]

    df_orb_elem = pd.DataFrame(results, columns=column_name,)

    return df_orb_elem[df_orb_elem["a"] != -1.0]


def remove_mirror(pdf):
    """
    Remove the mirror associations that can occurs in the dataframe return by the orbit identification
    - Mirrors means (A <-> B and B <-> A)

    Parameters
    ----------
    pdf : dataframe
        The dataframe return by the orbit identification where trajectory_id_1 are for the first arcs and trajectory_id_2 are for the second arcs.

    Return
    ------
    pdf : dataframe
        The mirrors associations have been removed, keep only one of them.
    """
    return pdf.loc[
        pd.DataFrame(
            np.sort(pdf[["trajectory_id_1", "trajectory_id_2"]], 1), index=pdf.index
        )
        .drop_duplicates(keep="first")
        .index
    ].sort_values(["trajectory_id_2"])


def remove_transitive(pdf):
    """
    Remove the transitive associations that can occurs in the dataframe return by the orbit identification
    - Transitive means (A <-> B and B <-> C => A <-> C)
    Parameters
    ----------
    pdf : dataframe
        The dataframe return by the orbit identification where trajectory_id_1 are for the first arcs and trajectory_id_2 are for the second arcs.

    Return
    ------
    pdf : dataframe
        The trajectory_id_2 replaced by the trajectory_id_1 of the first arcs involved in the transitive associations
    """
    transitive_left = pdf["trajectory_id_1"].isin(pdf["trajectory_id_2"])

    transitive_id = pdf[pdf["trajectory_id_2"].isin(pdf["trajectory_id_1"])][
        "trajectory_id_1"
    ].values

    pdf.loc[transitive_left, "trajectory_id_1"] = transitive_id

    return pdf


def merge_obs_id(pdf_obs, pdf_traj_merge):
    """
    Merge the observations of the both arcs to return only one single trajectories for each merged trajectories.

    Parameters
    ----------
    pdf_obs : dataframe
        The original observations dataframe
    pdf_traj_merge : dataframe
        The dataframe return by the orbit identification. It contains mainly the trajectory id of the both arcs

    Return
    ------
    tmp_pdf_obs : dataframe
        A copy of the original observations dataframe. The trajectory_id of the second arcs have been replaced by the one of the first arcs.
    """
    second_traj = pdf_obs[
        pdf_obs["trajectory_id"].isin(pdf_traj_merge["trajectory_id_2"])
    ]

    second_traj_size = second_traj.groupby(["trajectory_id"]).count()["ra"].to_numpy()

    tr_id_repeat = np.repeat(
        pdf_traj_merge["trajectory_id_1"].to_numpy(), second_traj_size
    )

    tmp_pdf_obs = pdf_obs.copy()

    tmp_pdf_obs.loc[
        tmp_pdf_obs["trajectory_id"].isin(pdf_traj_merge["trajectory_id_2"]),
        "trajectory_id",
    ] = tr_id_repeat

    return tmp_pdf_obs


def merge_orb_id(orb_cand, confirmed_merger, pdf_traj_merge):
    """
    Modification of the orbital elements dataframe to take into account of the merging.
    Remove the orbital elements of the second arcs and replaces the orbital elements of the first arcs
    by those of the merger.

    Parameters
    ----------
    orb_cand : dataframe
        the orginal orbital elements dataframe
    confirmed_merger : dataframe
        the orbital elements of the merged trajectories confirmed return by the second call to orbfit
    pdf_traj_merge : dataframe
        the trajectory_id of the both arcs with their orbitals elements return by the orbit identification

    Return
    ------
    orb_cand : dataframe
        The orbit elements dataframe where the second arcs of the confirmed merger have been removed and the orbital elements
        of the first arcs have been replaced by those of the merger.
    """

    # get the tmp merger view of the confirmed merger
    tmp_merger = pdf_traj_merge[
        pdf_traj_merge["trajectory_id_1"].isin(confirmed_merger["trajectory_id"])
    ]

    # remove the second orbital elements and observation arcs of the merger
    orb_cand = orb_cand[~orb_cand["trajectory_id"].isin(tmp_merger["trajectory_id_2"])]

    # get the orbital elements of the first confirmed arcs
    merged_traj_id = orb_cand["trajectory_id"].isin(tmp_merger["trajectory_id_1"])

    column_name = [
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

    # replaces the orbital elements by the merger ones
    orb_cand.loc[merged_traj_id, column_name] = confirmed_merger[column_name].to_numpy()

    return orb_cand


def orbit_identification(obs_cand, orbit_elem_cand, ram_dir, nb_neighbor, cpu_count):
    """
    Call orbfit to merge two trajectories candidates. Create two observations files and orbfit try to match a single orbit
    based on the two orbitals arcs (the two trajectories). Orbfit is able to return orbital elements for the merged trajectories
    but, due to a lack of documentation from Orbfit, a second call to orbfit is performed to return a better set of orbital elements for the
    merged trajectories.

    Parameters
    ----------
    obs_cand : dataframe
        The set of observations of each trajectories candidates.
    orbit_elem_cand : dataframe
        The set of orbital elements of each trajectories candidates
    ram_dir : string
        path where to write the temporary files generated by Orbfit
    nb_neighbor : integer
        The number of neighbor used to associates the trajectories
    cpu_count : integer
        The number of cpu core used by the parallel orbfit computation

    Return
    ------
    new_obs_cand : dataframe
        The observations dataframe, the trajectories that may belong to the same solar system objects have been merged together
    new_orbit_cand : dataframe
        The dataframe with the orbital elements, the orbital elements of the second arcs have been discarded and the orbital elements
        of the first arcs have been replaced by those of the merger.
    """

    # call orbfit to merge two orbitals arcs
    merge_results = merge_orbit(
        obs_cand, orbit_elem_cand, ram_dir, nb_neighbor, cpu_count
    )

    # (A <-> B means A associated with B)
    # remove the mirror (A <-> B and B <-> A) and transitive (A <-> B and B <-> C => A <-> C)
    merge_traj = remove_mirror(merge_results)
    merge_traj = remove_transitive(merge_traj)

    # the trajectory_id of the second arcs are replaced with the trajectory_id of the first arcs
    merged_obs_cand = merge_obs_id(obs_cand, merge_traj)

    # get the new merged trajectories
    new_traj = merged_obs_cand[
        merged_obs_cand["trajectory_id"].isin(merge_traj["trajectory_id_1"])
    ]

    # call orbfit to get the new orbital paramters of the merged trajectories even if the orbit identification return orbital elements.
    new_orb = ol.compute_df_orbit_param(new_traj, cpu_count, ram_dir)
    confirmed_merger = new_orb[(new_orb["a"] != -1.0) & (new_orb["rms_a"] != -1.0)]

    new_orbit_cand = merge_orb_id(orbit_elem_cand, confirmed_merger, merge_traj)

    # get the both trajectory_id merged arcs
    tmp_confirmed_merger = merge_traj[
        merge_traj["trajectory_id_1"].isin(confirmed_merger["trajectory_id"])
    ]

    # remove the both merged arcs from the observations dataframe
    obs_cand = obs_cand[
        ~obs_cand["trajectory_id"].isin(tmp_confirmed_merger["trajectory_id_1"])
    ]
    obs_cand = obs_cand[
        ~obs_cand["trajectory_id"].isin(tmp_confirmed_merger["trajectory_id_2"])
    ]

    # get the observations of the confirmed merger
    confirmed_obs = new_traj[
        new_traj["trajectory_id"].isin(confirmed_merger["trajectory_id"])
    ]
    # concat the old observations with the observations of the merged observations
    new_obs_cand = pd.concat([obs_cand, confirmed_obs])

    return new_obs_cand, new_orbit_cand


# if __name__ == "__main__":

#     ram_dir = "/media/virtuelram/" # "/tmp/ramdisk/"

#     path_data = "~/Documents/Doctorat/Asteroids/test_asteroids_candidates/ZTF/hope_without_bug/asteroids_candidates_resultats/candidates"

#     obs_cand = pd.read_parquet(os.path.join(path_data, "trajectory_orb.parquet")).sort_values(["trajectory_id"])

#     orbit_candidate = pd.read_parquet(os.path.join(path_data, "orbital.parquet")).reset_index(drop=True)


#     orbit_candidate["chi_reduced"] = np.ones(len(orbit_candidate)) * -1.0

#     obs_cand_with_merger, orb_cand_with_merger = orbit_identification(obs_cand, orbit_candidate, "/media/virtuelram/", 5, 10)

#     print(orb_cand_with_merger.sort_values(["chi_reduced"]))
