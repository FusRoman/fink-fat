import pandas as pd
import numpy as np
import os
from sklearn.neighbors import NearestNeighbors


from glob import glob
from shutil import rmtree
import multiprocessing as mp

import fink_fat.orbit_fitting.orbfit_local as ol
import fink_fat.orbit_fitting.orbfit_files as of
import fink_fat.orbit_fitting.mpcobs_files as mf


def parallel_merger(ram_dir, trajectory_df, orb_cand, indices, prop_epoch):
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
        Indices of the orb_cand parameters used to recover the neighborhood of a trajectories.
    prop_epoch : float
        Epoch at which output orbital elements in JD.

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
            of.write_oop(
                ram_dir,
                first_obj,
                second_obj,
                prop_epoch="JD  {} UTC".format(prop_epoch),
            )

            ol.call_orbitfit(ram_dir, first_obj, second_obj)

            readed_orb = of.read_oel(ram_dir, first_obj, second_obj)
            readed_orb[1] = neighbor_traj[0]
            readed_orb[14] = other_traj
            readed_orb[26] = first_obj + "_" + second_obj
            res_orb.append(readed_orb)

        of.rm_files(glob(os.path.join(ram_dir, "*.err")))
        of.rm_files(glob(os.path.join(ram_dir, "*.inp")))
        of.rm_files(glob(os.path.join(ram_dir, "*.oop")))
        of.rm_files(glob(os.path.join(ram_dir, "*.pro")))
        of.rm_files(glob(os.path.join(ram_dir, "*.odc")))

    return res_orb


def merge_orbit(
    observations, orbit_candidate, ram_dir, nb_neighbors, cpu_count, prop_epoch
):
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
    prop_epoch : float
        Epoch at which output orbital elements in JD.

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
        (chunk_dir, observations, orbit_candidate, tr_chunk, prop_epoch)
        for tr_chunk, chunk_dir in zip(trajectory_id_chunks, chunk_ramdir)
        if len(tr_chunk) > 0
    ]

    pool = mp.Pool(cpu_count)

    results = pool.starmap(parallel_merger, chunks)

    for chunk_dir in chunk_ramdir:
        rmtree(chunk_dir)

    of.rm_files(glob("*.rwo"))

    results = [el2 for el1 in results for el2 in el1]

    column_name = [
        "ref_epoch",
        "trajectory_id_1",
        "a1",
        "e1",
        "i1",
        "long. node1",
        "arg. peric1",
        "mean anomaly1",
        "rms_a1",
        "rms_e1",
        "rms_i1",
        "rms_long. node1",
        "rms_arg. peric1",
        "rms_mean anomaly1",
        "trajectory_id_2",
        "a2",
        "e2",
        "i2",
        "long. node2",
        "arg. peric2",
        "mean anomaly2",
        "rms_a2",
        "rms_e2",
        "rms_i2",
        "rms_long. node2",
        "rms_arg. peric2",
        "rms_mean anomaly2",
        "traj_merge",
        "a_merge",
        "e_merge",
        "i_merge",
        "long. node_merge",
        "arg. peric_merge",
        "mean anomaly_merge",
        "rms_a_merge",
        "rms_e_merge",
        "rms_i_merge",
        "rms_long. node_merge",
        "rms_arg. peric_merge",
        "rms_mean anomaly_merge",
    ]

    df_orb_elem = pd.DataFrame(results, columns=column_name,)

    return df_orb_elem[df_orb_elem["a1"] != -1.0]
