import numpy as np
import pandas as pd
from fink_fat.kalman.asteroid_kalman import KalfAst
import sys
import doctest


def make_kalman(gb_input):
    """
    GroupBy input

    Parameters
    ----------
    gb_input : pd.DataFrame
        contains a group from a groupby.

    Examples
    --------
    >>> input = pd.DataFrame({'ra': [11, 12, 29, 21, 15], 'dec': [20, 4, 0, 17, 23], 'jd': [10, 12, 16, 24, 28], 'magpsf': [16, 12, 9, 8, 24], 'fid': [16, 27, 1, 7, 23], 'trajectory_id': [6, 6, 6, 6, 6]})
    >>> make_kalman(input)
    ra_0         21
    dec_0        17
    ra_1         15
    dec_1        23
    jd_0         24
    jd_1         28
    mag_1        24
    fid_1        23
    dt            4
    vel_ra     -1.5
    vel_dec     1.5
    kalman     kf 6
    dtype: object

    >>> input = pd.DataFrame({'ra': [18, 19], 'dec': [1, 16], 'jd': [3, 24], 'magpsf': [13, 2], 'fid': [18, 7], 'trajectory_id': [7, 7]})
    >>> make_kalman(input)
    ra_0             18
    dec_0             1
    ra_1             19
    dec_1            16
    jd_0              3
    jd_1             24
    mag_1             2
    fid_1             7
    dt               21
    vel_ra     0.047619
    vel_dec    0.714286
    kalman         kf 7
    dtype: object
    """

    def compute_vel(x1, x2, dt):
        if dt == 0:
            dt = 30 / 3600 / 24
        return (x2 - x1) / dt

    def make_A(dt):
        return np.array(
            [
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )

    tr_id, ra, dec, jd, mag, fid = (
        int(gb_input["trajectory_id"].values[0]),
        gb_input["ra"].values,
        gb_input["dec"].values,
        gb_input["jd"].values,
        gb_input["magpsf"].values,
        gb_input["fid"].values,
    )

    ra_0, ra_1 = ra[0], ra[1]
    dec_0, dec_1 = dec[0], dec[1]
    jd_0, jd_1 = jd[0], jd[1]
    mag_1 = mag[1]
    fid_1 = fid[1]

    dt = jd_1 - jd_0
    ra_vel = compute_vel(ra_0, ra_1, dt)
    dec_vel = compute_vel(dec_0, dec_1, dt)

    kaf = KalfAst(tr_id, ra_1, dec_1, ra_vel, dec_vel, make_A(dt))
    if len(ra) > 2:
        for i in range(2, len(ra)):
            ra_i = ra[i]
            dec_i = dec[i]
            jd_i = jd[i]
            mag_i = mag[i]
            fid_i = fid[i]

            ra_0 = ra_1
            dec_0 = dec_1
            jd_0 = jd_1

            ra_1 = ra_i
            dec_1 = dec_i
            jd_1 = jd_i
            mag_1 = mag_i
            fid_1 = fid_i

            dt = jd_1 - jd_0
            if dt == 0:
                continue
            ra_vel = compute_vel(ra_0, ra_1, dt)
            dec_vel = compute_vel(dec_0, dec_1, dt)

            Y = np.array([[ra_1], [dec_1], [ra_vel], [dec_vel]])
            kaf.update(Y, make_A(dt))
    return pd.Series(
        [ra_0, dec_0, ra_1, dec_1, jd_0, jd_1, mag_1, fid_1, dt, ra_vel, dec_vel, kaf],
        index=[
            "ra_0",
            "dec_0",
            "ra_1",
            "dec_1",
            "jd_0",
            "jd_1",
            "mag_1",
            "fid_1",
            "dt",
            "vel_ra",
            "vel_dec",
            "kalman",
        ],
    )


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
    >>> np.random.seed(3)
    >>> df = pd.DataFrame(
    ...     np.random.randint(0, 30, size=(30, 5)),
    ...     columns=["ra", "dec", "jd", "magpsf", "fid"],
    ... )
    >>> tr_id = np.repeat([0, 1, 2, 3, 4, 5, 6, 7], [2, 2, 5, 4, 8, 2, 5, 2])
    >>> df["trajectory_id"] = tr_id

    >>> init_kalman(df)
       trajectory_id  ra_0  dec_0  ra_1  dec_1  jd_0  jd_1  mag_1  fid_1  dt     vel_ra   vel_dec kalman
    0              0     8      0    10     24    21    25      3     24   4   0.500000  6.000000   kf 0
    1              1    21     23    10     11     6    25      9     10  19  -0.578947 -0.631579   kf 1
    2              2    26     17    10     24    26    27      7     24   1 -16.000000  7.000000   kf 2
    3              3    20     28    11      1    23    27     15     16   4  -2.250000 -6.750000   kf 3
    4              4     4      3    18      2    27    28      4     26   1  14.000000 -1.000000   kf 4
    5              5    22     16     8     12     4    16     27     10  12  -1.166667 -0.333333   kf 5
    6              6    21     17    15     23    24    28     24     23   4  -1.500000  1.500000   kf 6
    7              7    18      1    19     16     3    24      2      7  21   0.047619  0.714286   kf 7

    >>> df = pd.DataFrame(columns=["ra", "dec", "jd", "magpsf", "fid", "trajectory_id"])
    >>> init_kalman(df)
    Empty DataFrame
    Columns: [ra, dec, jd, magpsf, fid, trajectory_id]
    Index: []
    """
    if len(night_pdf) == 0:
        return pd.DataFrame(
            columns=[
                "trajectory_id",
                "ra_0",
                "dec_0",
                "ra_1",
                "dec_1",
                "jd_0",
                "jd_1",
                "mag_1",
                "fid_1",
                "dt",
                "vel_ra",
                "vel_dec",
                "kalman",
            ]
        )

    # TODO
    # detecter les jd dupliqués et appliquer une fonction spécial dans le cas ou dt = 0

    # remove dbscan noise
    night_pdf = night_pdf[night_pdf["trajectory_id"] != -1.0]
    return (
        night_pdf.sort_values(["trajectory_id", "jd"])
        .groupby("trajectory_id")
        .apply(make_kalman)
        .reset_index()
    )


if __name__ == "__main__":  # pragma: no cover
    if "unittest.util" in __import__("sys").modules:
        # Show full diff in self.assertEqual.
        __import__("sys").modules["unittest.util"]._MAX_LENGTH = 999999999

    sys.exit(doctest.testmod()[0])
