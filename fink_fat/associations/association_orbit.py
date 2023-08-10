import numpy as np
import pandas as pd
from typing import Tuple

from fink_fat.orbit_fitting.orbfit_local import compute_df_orbit_param
from fink_fat.command_line.orbit_cli import cluster_mode


def generate_fake_traj_id(orb_pdf: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Generate fake integer trajectory_id based on the ssoCandId for the orbit fitting.

    Parameters
    ----------
    orb_pdf : pd.DataFrame
        contains the orbital parameters

    Returns
    -------
    Tuple[pd.DataFrame, dict]
        * the input orbital dataframe with the trajectory_id column
        * a dictionary to translate back to the ssoCandId
    """
    sso_id = orb_pdf["ssoCandId"].unique()
    map_tr = {
        sso_name: tr_id for sso_name, tr_id in zip(sso_id, np.arange(len(sso_id)))
    }
    tr_map = {
        tr_id: sso_name for sso_name, tr_id in zip(sso_id, np.arange(len(sso_id)))
    }
    orb_pdf["trajectory_id"] = orb_pdf["ssoCandId"].map(map_tr)
    return orb_pdf, tr_map


def orbit_associations(
    config: dict,
    new_alerts: pd.DataFrame,
    trajectory_df: pd.DataFrame,
    orbits: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Add the alerts 5-flagged to the trajectory_df and recompute the orbit with the new points.

    Parameters
    ----------
    config : dict
        the data from the configuration file
    new_alerts : pd.DataFrame
        alerts from the current nights with the roid column
    trajectory_df : pd.DataFrame
        contains the alerts of the trajectories
    orbits : pd.DataFrame
        contains the orbital parameters of the orbits

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        the new orbits and the updated trajectories
    """
    orbit_cols_to_keep = list(orbits.columns)
    traj_cols_to_keep = list(trajectory_df.columns)
    if "ffdistnr" not in traj_cols_to_keep:
        traj_cols_to_keep.append("ffdistnr")

    orbit_alert_assoc = (
        new_alerts[new_alerts["flag"] == 5]
        .rename({"estimator_id": "ssoCandId"}, axis=1)
        .drop("flag", axis=1)
    )
    updated_sso_id = orbit_alert_assoc["ssoCandId"].unique()
    # get the trajectories to update
    traj_to_update = trajectory_df[trajectory_df["ssoCandId"].isin(updated_sso_id)]
    # add the new alerts
    traj_to_new_orbit = pd.concat([traj_to_update, orbit_alert_assoc])
    traj_to_new_orbit, trid_to_ssoid = generate_fake_traj_id(traj_to_new_orbit)

    duplicated_id = orbit_alert_assoc[
        orbit_alert_assoc[["ssoCandId", "jd"]].duplicated()
    ]["ssoCandId"]
    assert len(duplicated_id) == 0

    nb_orb = len(traj_to_new_orbit["trajectory_id"].unique())
    if nb_orb > int(config["SOLVE_ORBIT_PARAMS"]["local_mode_limit"]):
        new_orbit_pdf = cluster_mode(config, traj_to_new_orbit)
    else:
        new_orbit_pdf = compute_df_orbit_param(
            traj_to_new_orbit,
            int(config["SOLVE_ORBIT_PARAMS"]["cpu_count"]),
            config["SOLVE_ORBIT_PARAMS"]["ram_dir"],
            int(config["SOLVE_ORBIT_PARAMS"]["n_triplets"]),
            int(config["SOLVE_ORBIT_PARAMS"]["noise_ntrials"]),
            config["SOLVE_ORBIT_PARAMS"]["prop_epoch"],
            int(config["SOLVE_ORBIT_PARAMS"]["orbfit_verbose"]),
        ).drop("provisional designation", axis=1)

    # remove the failed orbits
    new_orbit_pdf = new_orbit_pdf[new_orbit_pdf["a"] != -1.0]
    new_orbit_pdf["ssoCandId"] = new_orbit_pdf["trajectory_id"].map(trid_to_ssoid)
    updated_id = new_orbit_pdf["ssoCandId"]

    # update orbit
    old_orbit = orbits[~orbits["ssoCandId"].isin(updated_id)]
    old_orbit_len = len(orbits)
    orbits = pd.concat([old_orbit, new_orbit_pdf])[orbit_cols_to_keep]
    assert len(orbits) == old_orbit_len

    # update traj
    new_traj = traj_to_new_orbit[traj_to_new_orbit["ssoCandId"].isin(updated_id)]
    old_traj = trajectory_df[~trajectory_df["ssoCandId"].isin(updated_id)]

    trajectory_df = pd.concat([old_traj, new_traj])[traj_cols_to_keep]

    return orbits, trajectory_df
