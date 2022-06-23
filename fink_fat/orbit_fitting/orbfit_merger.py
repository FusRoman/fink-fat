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

    Examples
    --------
    >>> merge_file = open('fink_fat/test/merge_test/merge_data.pickle', 'rb')
    >>> merge_data = pickle.load(merge_file)
    >>> os.makedirs("chunkid_0/mpcobs/")

    >>> parallel_merger(*merge_data)
    [[2459752.003580741, 0, '2.4238086390916203E+00', '0.269164770785356', '25.6894257030655', '335.5592066316808', '251.8641715562610', '287.2148986522243', '6.42137E-05', '2.01103E-05', '7.82872E-04', '1.06443E-03', '8.46007E-03', '1.96242E-02', 30, '2.4184111483160988E+00', '0.267570739798363', '25.6575323148178', '335.5804344016931', '251.8205861139055', '287.9142383344984', '1.00962E-04', '4.12691E-05', '7.29526E-05', '7.69820E-04', '2.10740E-03', 'K20M00A_K21A01F', 'K20M00A=K21A01F', '2.4193869692732242E+00', '0.267263915046098', '25.6520543897533', '335.5009168460746', '251.7607054960978', '288.0667606295705', '2.82103E-04', '9.26884E-05', '2.51249E-04', '4.43638E-04', '4.76105E-03', '2.58625E-02'], [2459752.003580741, 29, '3.0452145924429099E+00', '0.065606648677047', '23.3374231190184', '267.4524661689092', '128.7655075980927', '101.3033609159210', '6.18848E-04', '8.87126E-05', '7.53815E-04', '7.18789E-03', '1.63296E-01', '1.08993E-01', 59, '3.0509605762796492E+00', '0.068572028897144', '23.3167627278706', '267.5385575735852', '130.8180167064951', '99.1979965927759', '4.29550E-04', '1.04754E-04', '2.98456E-04', '8.04180E-04', '4.46305E-02', 'K19V01E_K20V02K', 'K19V01E=K20V02K', '3.0425223319646371E+00', '0.066444480517143', '23.3221026518435', '267.5088738794881', '130.2270472077813', '100.1547290834541', '2.75905E-04', '7.90609E-05', '7.69444E-04', '3.31889E-03', '3.51691E-02', '3.21688E-02'], [2459752.003580741, 30, '2.4184111431953395E+00', '0.267570741616777', '25.6575323561144', '335.5804344791498', '251.8205858956849', '287.9142389123562', '1.00957E-04', '4.12670E-05', '7.29480E-05', '7.69737E-04', '2.10729E-03', '6.20331E-03', 0, '2.4238086412407203E+00', '0.269164770589233', '25.6894256560389', '335.5592065429524', '251.8641720787436', '287.2148978126605', '6.42137E-05', '2.01101E-05', '7.82866E-04', '1.06442E-03', '8.46004E-03', 'K21A01F_K20M00A', 'K21A01F=K20M00A', '2.4193869694207026E+00', '0.267263914985993', '25.6520544000150', '335.5009168774209', '251.7607054287975', '288.0667606790379', '2.82090E-04', '9.26841E-05', '2.51233E-04', '4.43628E-04', '4.76081E-03', '2.58613E-02'], [2459752.003580741, 59, '3.0509604188329651E+00', '0.068571983469542', '23.3167628061171', '267.5385569049111', '130.8180092135307', '99.1980116026713', '4.29555E-04', '1.04755E-04', '2.98465E-04', '8.04201E-04', '4.46309E-02', '5.91528E-02', 29, '3.0452146848792898E+00', '0.065606644259685', '23.3374225384160', '267.4524729115861', '128.7654708169027', '101.3033825001779', '6.18835E-04', '8.87085E-05', '7.53780E-04', '7.18783E-03', '1.63293E-01', 'K20V02K_K19V01E', 'K20V02K=K19V01E', '3.0425223182949446E+00', '0.066444467499287', '23.3221026916831', '267.5088744082560', '130.2270556368874', '100.1547219079311', '2.75901E-04', '7.90590E-05', '7.69436E-04', '3.31884E-03', '3.51693E-02', '3.21699E-02']]

    >>> rmtree("chunkid_0")
    >>> of.rm_files(glob("*.rwo"))
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

    Examples
    --------

    >>> merge_test_path = "fink_fat/test/merge_test/"
    >>> merge_res = merge_orbit(pd.read_parquet(merge_test_path + "obs_merge.parquet"), pd.read_parquet(merge_test_path + "orb_merge.parquet"), "", 2, 1, 2459752.00278)

    >>> assert_frame_equal(merge_res, pd.read_parquet(merge_test_path + "merge_test_results.parquet"))
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

    of.rm_files(glob(ram_dir + "*.rwo"))

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


if __name__ == "__main__":  # pragma: no cover
    import sys
    import doctest
    from pandas.testing import assert_frame_equal  # noqa: F401
    import fink_fat.test.test_sample as ts  # noqa: F401
    from unittest import TestCase  # noqa: F401
    import shutil  # noqa: F401
    import filecmp  # noqa: F401
    import stat  # noqa: F401

    # import pickle  # noqa: F401
    import pickle5 as pickle  # noqa: F401

    if "unittest.util" in __import__("sys").modules:
        # Show full diff in self.assertEqual.
        __import__("sys").modules["unittest.util"]._MAX_LENGTH = 999999999

    sys.exit(doctest.testmod()[0])
