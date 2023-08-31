import numpy as np
import pandas as pd
from typing import Tuple

from fink_fat.command_line.orbit_cli import switch_local_cluster


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
        new_alerts[new_alerts["roid"] == 5]
        .rename({"estimator_id": "ssoCandId"}, axis=1)
        .drop("roid", axis=1)
    )
    if len(orbit_alert_assoc) == 0:
        return orbits, trajectory_df

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

    new_orbit_pdf = switch_local_cluster(config, traj_to_new_orbit)

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
