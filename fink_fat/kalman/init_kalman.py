import numpy as np
import pandas as pd
from fink_fat.kalman.asteroid_kalman import KalfAst
import sys
import doctest


def init_kalman(
    night_pdf: pd.DataFrame,
) -> pd.DataFrame:
    """
    Initialize kalman filters based on the seeds or tracklets.
    required columns: ra, dec, jd, magpsf, fid, trajectory_id

    Parameters
    ----------
    night_pdf : pd.DataFrame
        alerts of one night

    Returns
    -------
    pd.DataFrame
        a dataframe containing the kalman filters

    Examples
    --------
    >>> seeds = pd.DataFrame({
    ... "ra": [0, 1, 2, 3, 4, 8],
    ... "dec": [1, 4, 9, 12, 13, 24],
    ... "jd": [0, 1, 2, 3, 5, 8],
    ... "magpsf": [12, 14, 16, 17, 17.5, 18],
    ... "fid": [1, 2, 1, 1, 2, 2],
    ... "trajectory_id": [0, 0, 1, 1, 2, 2]
    ... })

    >>> init_kalman(seeds)
       trajectory_id  ra_0  dec_0  ra_1  dec_1  jd_0  jd_1  mag_1  fid_1  dt    vel_ra   vel_dec kalman
    0              0     0      1     1      4     0     1   14.0      2   1  1.000000  3.000000   kf 0
    1              1     2      9     3     12     2     3   17.0      1   1  1.000000  3.000000   kf 1
    2              2     4     13     8     24     5     8   18.0      2   3  1.333333  3.666667   kf 2
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
            tr_id,
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
        for tr_id, ra, dec, ra_vel, dec_vel, dt in zip(
            prep_kalman["trajectory_id"],
            prep_kalman["ra_1"],
            prep_kalman["dec_1"],
            prep_kalman["vel_ra"],
            prep_kalman["vel_dec"],
            prep_kalman["dt"],
        )
    ]

    prep_kalman["kalman"] = kalman_list

    return prep_kalman


if __name__ == "__main__":  # pragma: no cover
    if "unittest.util" in __import__("sys").modules:
        # Show full diff in self.assertEqual.
        __import__("sys").modules["unittest.util"]._MAX_LENGTH = 999999999

    sys.exit(doctest.testmod()[0])
