import pandas as pd


def prep_traj_for_mpc(trajectory_orb: pd.DataFrame)->pd.DataFrame:
    """
    Filter and change the trajectories contains in trajectory_orb to fit the mpc submission requirements.

    Parameters
    ----------
    trajectory_orb : pd.DataFrame
        contains observations of the trajectories with orbits return by Fink-FAT

    Returns
    -------
    pd.DataFrame
        trajectories with orbit ready to be send to the Minor Planet Center
    """
    pass