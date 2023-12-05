import numpy as np
import pandas as pd
from shutil import rmtree
import subprocess
import os
import multiprocessing as mp

import fink_fat.orbit_fitting.orbfit_files as of
import fink_fat.orbit_fitting.mpcobs_files as mf

import traceback
import logging
from typing import Union

orbfit_column_name = [
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


def call_orbitfit(ram_dir, first_designation, second_designation=None, verbose=False):
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

    >>> call_orbitfit("fink_fat/test/call_orbfit/", "K19V01E", "K20V02K")

    >>> os.path.exists("fink_fat/test/call_orbfit/K19V01E_K20V02K.oel")
    True

    >>> call_orbitfit("fink_fat/test/call_orbfit/", "K23Nb8K")

    >>> call_orbitfit("fink_fat/test/call_orbfit/", "K21E00A", verbose=True)
    >>> call_orbitfit("fink_fat/test/call_orbfit/", "K19V01E", "K20V02K", verbose=True)

    >>> os.path.exists("fink_fat/test/call_orbfit/K23Nb8K.log")
    True
    >>> os.path.exists("fink_fat/test/call_orbfit/K21E00A.log")
    True
    >>> os.path.exists("fink_fat/test/call_orbfit/K19V01E_K20V02K.log")
    True

    >>> os.remove("fink_fat/test/call_orbfit/K19V01E_K20V02K.odc")
    >>> os.remove("fink_fat/test/call_orbfit/K19V01E_K20V02K.olg")
    >>> os.remove("fink_fat/test/call_orbfit/K19V01E_K20V02K.pro")
    >>> os.remove("fink_fat/test/call_orbfit/mpcobs/K19V01E.rwo")
    >>> os.remove("fink_fat/test/call_orbfit/mpcobs/K20V02K.rwo")
    >>> os.remove("fink_fat/test/call_orbfit/K19V01E_K20V02K.oel")
    >>> os.remove("fink_fat/test/call_orbfit/K19V01E_K20V02K.err")

    >>> os.remove("fink_fat/test/call_orbfit/K21E00A.odc")
    >>> os.remove("fink_fat/test/call_orbfit/K21E00A.olg")
    >>> os.remove("fink_fat/test/call_orbfit/K21E00A.pro")
    >>> os.remove("fink_fat/test/call_orbfit/K21E00A.clo")
    >>> os.remove("fink_fat/test/call_orbfit/K21E00A.err")
    >>> os.remove("fink_fat/test/call_orbfit/mpcobs/K21E00A.rwo")
    >>> os.remove("fink_fat/test/call_orbfit/K21E00A.oel")

    >>> os.remove("fink_fat/test/call_orbfit/K23Nb8K.odc")
    >>> os.remove("fink_fat/test/call_orbfit/K23Nb8K.olg")
    >>> os.remove("fink_fat/test/call_orbfit/K23Nb8K.pro")
    >>> os.remove("fink_fat/test/call_orbfit/K23Nb8K.clo")
    >>> os.remove("fink_fat/test/call_orbfit/K23Nb8K.err")
    >>> os.remove("fink_fat/test/call_orbfit/mpcobs/K23Nb8K.rwo")
    >>> os.remove("fink_fat/test/call_orbfit/K23Nb8K.oel")

    >>> os.remove("fink_fat/test/call_orbfit/K23Nb8K.log")
    >>> os.remove("fink_fat/test/call_orbfit/K21E00A.log")
    >>> os.remove("fink_fat/test/call_orbfit/K19V01E_K20V02K.log")
    """
    orbitfit_path = os.path.join(os.environ["ORBFIT_HOME"], "bin", "")

    if second_designation is None:
        command = orbitfit_path + "orbfit.x < " + ram_dir + first_designation + ".inp "
    else:
        command = (
            orbitfit_path
            + "orbfit.x < "
            + ram_dir
            + first_designation
            + "_"
            + second_designation
            + ".inp "
        )

    def generate_logs(
        run_exception: Union[subprocess.CalledProcessError, subprocess.TimeoutExpired]
    ):
        err_type = (
            f"timeout exceeded: {run_exception.timeout} seconds"
            if isinstance(run_exception, subprocess.TimeoutExpired)
            else f"return code: {run_exception.returncode}"
        )
        str_err = f"""
--- ORBFIT ERROR ---
command: {run_exception.cmd}
{err_type}

stdout:
{run_exception.stdout.decode("utf-8")}



stderr:
{run_exception.stderr.decode("utf-8")}



subprocess output:
{run_exception.output.decode("utf-8")}
-------------------------------------
"""
        return str_err

    def write_logs(str_log, first_designation, second_designation):
        log_file_name = first_designation if second_designation is None else first_designation + "_" + second_designation
        with open(os.path.join(ram_dir, f"{log_file_name}.log"), "w") as f:
            f.write(str_log)

    try:
        completed_process = subprocess.run(
            command, shell=True, capture_output=True, timeout=5
        )
        try:
            completed_process.check_returncode()
            if verbose:
                str_out = f"""
--- ORBFIT OUTPUT ---
args: {completed_process.args}
returncode: {completed_process.returncode}

stdout:
{completed_process.stdout.decode("utf-8")}



stderr:
{completed_process.stderr.decode("utf-8")}
-------------------------------------
"""
                write_logs(str_out, first_designation, second_designation)

        except subprocess.CalledProcessError as e:
            write_logs(generate_logs(e), first_designation, second_designation)
    except subprocess.TimeoutExpired as te:
        write_logs(generate_logs(te), first_designation, second_designation)


def get_orbit_param(ram_dir, df, n_triplets, noise_ntrials, prop_epoch=None, verbose_orbfit=1, verbose=None):
    """
    Compute the orbital elements of one trajectory.

    Parameters
    ----------
    ram_dir : string
        Path where files are located
    df : dataframe
        All the observation of the trajectory. An observation file will be write in the MPC format based on the observations contains in this dataframe.
    n_triplets : integer
        max number of triplets of observations to be tried for the initial orbit determination
    noise_ntrials : integer
        number of trials for each triplet for the initial orbit determination
    prop_epoch : float
        Epoch at which output orbital elements in JD.
    verbose : integer
        Verbosity levels of Orbfit
        1 = summary information on the solution found
        2 = summary information on all trials
        3 = debug

    Returns
    -------
    results : list
        The list contains in this order : the trajectory identifier, the provisional designation, the reference epoch, the 6 orbital elements and finally
        the rms of the orbital elements.

    Examples
    --------
    >>> import fink_fat.orbit_fitting.orbfit_files as of
    >>> df = pd.DataFrame({
    ... 'ra': [169.8604675, 169.8568848, 169.8336664, 169.8297121, 169.8296555],
    ... 'dec': [15.2063604, 15.2103091, 15.2360481, 15.2403893, 15.24049],
    ... 'magpsf': [16.438142098160576, 16.47854604642893, 15.767506616421468, 15.781593431530103, 15.764373749886605],
    ... 'fid': [1, 1, 2, 2, 2],
    ... 'jd': [2459274.7206481, 2459274.7391435, 2459274.8594444, 2459274.8799074, 2459274.8803819],
    ... 'trajectory_id': [0, 0, 0, 0, 0]
    ... })

    >>> of.prep_orbitfit("")
    >>> res = get_orbit_param("", df, 15, 15)
    >>> res[0][:9]
    [0, 'K21E00A', 2459274.881182641, '1.2731835539687217E+00', '0.206413338170746', '2.7535934287416', '137.2985320438607', '321.1224023053072', '43.2327613057881']

    >>> of.final_clean("")

    >>> of.prep_orbitfit("")
    >>> res = get_orbit_param("", df, 15, 15, prop_epoch=2459752.16319)
    >>> res[0][:9]
    [0, 'K21E00A', 2459752.163990741, '1.2731741718578882E+00', '0.206162998389323', '2.7531115133717', '137.2874441097704', '321.1518212593283', '10.6664371694522']

    >>> of.final_clean("")
    """

    all_traj_id = np.unique(df["trajectory_id"].values)

    results = []
    for traj_id in all_traj_id:
        df_one_traj = df[df["trajectory_id"] == traj_id]
        prov_desig = mf.write_observation_file(ram_dir, df_one_traj)
        of.write_inp(ram_dir, prov_desig)

        if prop_epoch is None:
            of.write_oop(
                ram_dir,
                prov_desig,
                prop_epoch="JD  {} UTC".format(df_one_traj["jd"].values[-1]),
                n_triplets=n_triplets,
                noise_ntrials=noise_ntrials,
                verbose=verbose_orbfit,
            )
        else:
            of.write_oop(
                ram_dir,
                prov_desig,
                prop_epoch="JD  {} UTC".format(prop_epoch),
                n_triplets=n_triplets,
                noise_ntrials=noise_ntrials,
                verbose=verbose_orbfit,
            )

        try:
            call_orbitfit(ram_dir, prov_desig, verbose=verbose)
        except Exception as e:  # pragma: no cover
            print(e)
            print("ERROR CALLING ORBFIT: {}".format(prov_desig))
            print()
            logging.error(traceback.format_exc())
            print()
            print(prov_desig)
            print()
            print()
            print(df_one_traj)

        chi_values = of.read_rwo(ram_dir, prov_desig, len(df_one_traj))

        # reduced the chi values
        chi_reduced = np.sum(np.array(chi_values)) / len(df_one_traj)

        results.append(
            [traj_id, prov_desig] + of.read_oel(ram_dir, prov_desig) + [chi_reduced]
        )

        try:
            of.obs_clean(ram_dir, prov_desig)
        except FileNotFoundError:  # pragma: no cover
            print("ERROR CLEANING ORBFIT: {}".format(prov_desig))
            print(prov_desig)
            print()
            print()
            print(df_one_traj)

    return results


def orbit_elem_dataframe(orbit_elem, column_name):
    """
    Convert the list return by get_orbit_param into a dataframe.

    Parameters
    ----------
    orbit_elem : list
        The return of get_orbit_param
    column_name : string list
        Columns name of the result dataframe

    Returns
    -------
    df_orb_elem : dataframe
        the input list convert into a dataframe

    Examples
    --------
    >>> orb_list = [[0, 'K21E00A', 2459274.810893373, '1.5834346988159376E+00', '0.613572037782866', '5.9442185803697', '343.7959802838470', '270.1932521117385', '333.9568546371023', -1, -1, -1, -1, -1, -1, 2.4]]

    >>> orb_df = orbit_elem_dataframe(orb_list, orbfit_column_name)

    >>> assert_frame_equal(orb_df, ts.orb_elem_output)

    >>> os.rmdir("mpcobs")
    """

    df_orb_elem = pd.DataFrame(
        orbit_elem,
        columns=column_name,
    )

    for col_name in column_name:
        df_orb_elem[col_name] = pd.to_numeric(df_orb_elem[col_name], errors="ignore")

    return df_orb_elem


def compute_df_orbit_param(
    trajectory_df,
    cpu_count,
    ram_dir,
    n_triplets=10,
    noise_ntrials=10,
    prop_epoch=None,
    verbose_orbfit=1,
    verbose=None
):
    """
    Compute the orbital elements of a set of trajectories. Computation are done in parallel.

    Parameters
    ----------
    trajectory_df : dataframe
        the set of trajectories, the following columns are required : "ra", "dec", "magpsf", "fid", "jd", "trajectory_id"
    cpu_count : integer
        the number of core for the parallel computation
    ram_dir : string
        Path where files are located
    n_triplets : integer
        max number of triplets of observations to be tried for the initial orbit determination
    noise_ntrials : integer
        number of trials for each triplet for the initial orbit determination
    prop_epoch : float
        Epoch at which output orbital elements in JD.
    verbose : integer
        Verbosity levels of Orbfit
        1 = summary information on the solution found
        2 = summary information on all trials
        3 = debug

    Returns
    -------
    orbit_elem : dataframe
        the orbital elements computed by OrbFit for each inputs trajectories.
        column description:
            ref_epoch: referent epoch of the orbit (Julian date)
            a: semi-major axis (Astronomical unit)
            e: eccentricity
            i: inclination
            long. node: longitude of the ascending node (degree)
            arg. peri: argument of periapsis (degree)
            mean anomaly: (degree)

    Examples
    --------

    >>> orb_elem = compute_df_orbit_param(ts.orbfit_samples, 2, "")

    >>> assert_frame_equal(orb_elem, ts.orbfit_output)
    """

    # of.prep_orbitfit(ram_dir)

    all_traj_id = np.unique(trajectory_df["trajectory_id"].values)

    trajectory_id_chunks = np.array_split(all_traj_id, cpu_count)

    chunk_ramdir = [
        os.path.join(ram_dir, "chunkid_{}".format(chunk_id), "")
        for chunk_id in np.arange(len(trajectory_id_chunks))
    ]

    for chunk_dir in chunk_ramdir:
        os.mkdir(chunk_dir)
        of.prep_orbitfit(chunk_dir)

    chunks = [
        (
            chunk_dir,
            trajectory_df[trajectory_df["trajectory_id"].isin(tr_chunk)],
            n_triplets,
            noise_ntrials,
            prop_epoch,
            verbose_orbfit,
            verbose
        )
        for tr_chunk, chunk_dir in zip(trajectory_id_chunks, chunk_ramdir)
        if len(tr_chunk) > 0
    ]

    pool = mp.Pool(cpu_count)

    results = pool.starmap(get_orbit_param, chunks)
    results = [el2 for el1 in results for el2 in el1]

    pool.close()

    for chunk_dir in chunk_ramdir:
        rmtree(chunk_dir)

    of.final_clean(ram_dir)

    if len(results) > 0:
        return orbit_elem_dataframe(np.array(results), orbfit_column_name)
    else:  # pragma: no cover
        return pd.DataFrame(columns=orbfit_column_name)


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
