import numpy as np
import pandas as pd
from fink_fat.kalman.asteroid_kalman import KalfAst


def init_kalman(
    night_pdf: pd.DataFrame,
) -> pd.DataFrame:
    """
    Initialize kalman filters based on the seeds or tracklets.

    Parameters
    ----------
    night_pdf : pd.DataFrame
        alerts of one night

    Returns
    -------
    pd.DataFrame
        a dataframe containing the kalman filters
    """
    # remove dbscan noise
    night_pdf = night_pdf[night_pdf["trajectory_id"] != -1.0]

    prep_kalman = (
        night_pdf.sort_values("jd")
        .groupby("trajectory_id")
        .agg(
            ra_0=("ra", lambda x: list(x)[0] if len(list(x)) > 1 else np.nan),
            dec_0=("dec", lambda x: list(x)[0] if len(list(x)) > 1 else np.nan),
            ra_1=("ra", lambda x: list(x)[1]),
            dec_1=("dec", lambda x: list(x)[1]),
            jd_0=("jd", lambda x: list(x)[0] if len(list(x)) > 1 else np.nan),
            jd_1=("jd", lambda x: list(x)[1]),
            mag_1=("magpsf", lambda x: list(x)[1]),
            fid_1=("fid", lambda x: list(x)[1]),
        )
        .reset_index()
    )
    prep_kalman = prep_kalman[~prep_kalman["ra_0"].isna()]
    prep_kalman["dt"] = prep_kalman["jd_1"] - prep_kalman["jd_0"]
    prep_kalman["vel_ra"] = (prep_kalman["ra_1"] - prep_kalman["ra_0"]) / prep_kalman[
        "dt"
    ]
    prep_kalman["vel_dec"] = (
        prep_kalman["dec_1"] - prep_kalman["dec_0"]
    ) / prep_kalman["dt"]

    kalman_list = [
        KalfAst(
            ra,
            dec,
            ra_vel,
            dec_vel,
            [
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
        )
        for ra, dec, ra_vel, dec_vel, dt in zip(
            prep_kalman["ra_1"],
            prep_kalman["dec_1"],
            prep_kalman["vel_ra"],
            prep_kalman["vel_dec"],
            prep_kalman["dt"],
        )
    ]

    prep_kalman["kalman"] = kalman_list

    return prep_kalman
