import numpy as np
import pandas as pd
from typing import Tuple

from fink_fat.command_line.orbit_cli import switch_local_cluster
from fink_fat.others.utils import LoggerNewLine


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
    logger: LoggerNewLine,
    verbose: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    logger : LoggerNewLine
        logger class used to print the logs
    verbose : bool
        if true, print the logs

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        the new orbits, the updated trajectories and the old orbits
    """
    if verbose:
        logger.info("start the associations with the orbits")
    orbit_cols_to_keep = list(orbits.columns)
    traj_cols_to_keep = list(trajectory_df.columns)
    if "ffdistnr" not in traj_cols_to_keep:
        traj_cols_to_keep.append("ffdistnr")

    orbit_alert_assoc = (
        new_alerts[new_alerts["roid"] == 5]
        .explode(["estimator_id", "ffdistnr"])
        .rename({"estimator_id": "ssoCandId"}, axis=1)
        .drop("roid", axis=1)
    )
    if len(orbit_alert_assoc) == 0:
        return orbits, trajectory_df, pd.DataFrame(columns=orbits.columns)

    updated_sso_id = orbit_alert_assoc["ssoCandId"].unique()
    # get the trajectories to update
    traj_to_update = trajectory_df[trajectory_df["ssoCandId"].isin(updated_sso_id)]
    # add the new alerts
    traj_to_new_orbit = pd.concat([traj_to_update, orbit_alert_assoc])
    traj_to_new_orbit, trid_to_ssoid = generate_fake_traj_id(traj_to_new_orbit)

    if verbose:
        logger.info(f"number of orbits to update: {len(updated_sso_id)}")

    duplicated_id = orbit_alert_assoc[
        orbit_alert_assoc[["ssoCandId", "jd"]].duplicated()
    ]["ssoCandId"]
    assert len(duplicated_id) == 0

    new_orbit_pdf = switch_local_cluster(config, traj_to_new_orbit)

    # remove the failed orbits
    new_orbit_pdf = new_orbit_pdf[new_orbit_pdf["a"] != -1.0]
    if verbose:
        logger.info(
            f"number of successfull updated orbits: {len(new_orbit_pdf)} ({(len(new_orbit_pdf) / len(updated_sso_id)) * 100} %)"
        )
        logger.newline(2)
    new_orbit_pdf["ssoCandId"] = new_orbit_pdf["trajectory_id"].map(trid_to_ssoid)
    updated_id = new_orbit_pdf["ssoCandId"]

    # update orbit
    mask_orbit_update = orbits["ssoCandId"].isin(updated_id)
    current_orbit = orbits[~mask_orbit_update]
    old_orbit = orbits[mask_orbit_update]
    old_orbit_len = len(orbits)
    orbits = pd.concat([current_orbit, new_orbit_pdf])[orbit_cols_to_keep]
    assert len(orbits) == old_orbit_len

    # update traj
    new_traj = traj_to_new_orbit[traj_to_new_orbit["ssoCandId"].isin(updated_id)]
    old_traj = trajectory_df[~trajectory_df["ssoCandId"].isin(updated_id)]

    trajectory_df = pd.concat([old_traj, new_traj])[traj_cols_to_keep]
    return orbits, trajectory_df, old_orbit
