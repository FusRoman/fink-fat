import numpy as np
import pandas as pd
import sys
import doctest
import fink_fat.roid_fitting.utils_roid_fit as uf
import warnings


def fit_polyast(gb_input, poly_exp=2):
    """
    GroupBy input

    Parameters
    ----------
    gb_input : pd.DataFrame
        contains a group from a groupby.

    Examples
    --------
    >>> input = pd.DataFrame({'ra': [11, 12, 29, 21, 15], 'dec': [20, 4, 0, 17, 23], 'jd': [10, 12, 16, 24, 28], 'magpsf': [16, 12, 9, 8, 24], 'fid': [16, 27, 1, 7, 23], 'trajectory_id': [6, 6, 6, 6, 6]})
    >>> fit_polyast(input)
    ra_0                                                    21
    dec_0                                                   17
    ra_1                                                    15
    dec_1                                                   23
    jd_0                                                    24
    jd_1                                                    28
    mag_1                                                   24
    fid_1                                                   23
    xp       [0.0003689548973376653, -0.017083339224524503,...
    yp       [-0.003374983061298368, 0.13154198411573034, -...
    zp       [0.003889943130855235, -0.13737266595516961, 1...
    dtype: object

    >>> input = pd.DataFrame({'ra': [18, 19], 'dec': [1, 16], 'jd': [3, 24], 'magpsf': [13, 2], 'fid': [18, 7], 'trajectory_id': [7, 7]})
    >>> fit_polyast(input)
    ra_0                                                    18
    dec_0                                                    1
    ra_1                                                    19
    dec_1                                                   16
    jd_0                                                     3
    jd_1                                                    24
    mag_1                                                    2
    fid_1                                                    7
    xp       [-0.00022037675949518603, 0.003949178398779884...
    yp       [-5.387272454402794e-05, 0.0016443857868339133...
    zp       [0.0002562672237379922, 0.0053753063581082955,...
    dtype: object
    """

    ra, dec, jd, mag, fid = (
        gb_input["ra"].values,
        gb_input["dec"].values,
        gb_input["jd"].values,
        gb_input["magpsf"].values,
        gb_input["fid"].values,
    )

    ra_0, ra_1 = ra[-2], ra[-1]
    dec_0, dec_1 = dec[-2], dec[-1]
    jd_0, jd_1 = jd[-2], jd[-1]
    mag_1 = mag[-1]
    fid_1 = fid[-1]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", np.RankWarning)
        xp, yp, zp = uf.fit_traj(ra, dec, jd, poly_exp)

    return pd.Series(
        [ra_0, dec_0, ra_1, dec_1, jd_0, jd_1, mag_1, fid_1, xp, yp, zp],
        index=[
            "ra_0",
            "dec_0",
            "ra_1",
            "dec_1",
            "jd_0",
            "jd_1",
            "mag_1",
            "fid_1",
            "xp",
            "yp",
            "zp",
        ],
    )


def init_polyast(
    night_pdf: pd.DataFrame, poly_exp=2
) -> pd.DataFrame:
    """
    Initialize fit functions based on the seeds or tracklets.
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

    >>> init_polyast(df)
       trajectory_id  ra_0  ...                                                 yp                                                 zp
    0              0     8  ...  [5.8886683912253095e-05, 0.0021568132839048845...  [0.002130983101599165, 0.0036589380953885127, ...
    1              1    21  ...  [-0.0002667547833632723, -0.000121215680172234...  [-0.0003281002792150129, -0.000351108876583499...
    2              2    26  ...  [3.564698412338013e-05, 0.007580031292933427, ...  [0.0012283807329896033, -0.04112918641543257, ...
    3              3    20  ...  [-0.005899809113439577, 0.27815784005379157, -...  [-0.012428484041437362, 0.5283431168170123, -5...
    4              4     4  ...  [0.00033732318423005867, -0.020080309986114004...  [0.001367714730073901, -0.06476655293412147, 0...
    5              5    22  ...  [-0.0008599104450328825, -0.001465385641499352...  [-0.0003768027364769278, 0.0018922493129352458...
    6              6    21  ...  [-0.003374983061298368, 0.13154198411573034, -...  [0.003889943130855235, -0.13737266595516961, 1...
    7              7    18  ...  [-5.387272454402794e-05, 0.0016443857868339133...  [0.0002562672237379922, 0.0053753063581082955,...
    <BLANKLINE>
    [8 rows x 12 columns]

    >>> df = pd.DataFrame(columns=["ra", "dec", "jd", "magpsf", "fid", "trajectory_id"])
    >>> init_polyast(df)
    Empty DataFrame
    Columns: [ra_0, dec_0, ra_1, dec_1, jd_0, jd_1, mag_1, fid_1, xp, yp, zp]
    Index: []
    """
    if len(night_pdf) == 0:
        return pd.DataFrame(
            columns=[
                "ra_0",
                "dec_0",
                "ra_1",
                "dec_1",
                "jd_0",
                "jd_1",
                "mag_1",
                "fid_1",
                "xp",
                "yp",
                "zp",
            ]
        )

    # TODO
    # detecter les jd dupliqués et appliquer une fonction spécial dans le cas ou dt = 0

    # remove dbscan noise
    night_pdf = night_pdf[night_pdf["trajectory_id"] != -1.0]
    return (
        night_pdf.sort_values(["trajectory_id", "jd"])
        .groupby("trajectory_id")
        .apply(fit_polyast, poly_exp)
        .reset_index()
    )


if __name__ == "__main__":  # pragma: no cover
    if "unittest.util" in __import__("sys").modules:
        # Show full diff in self.assertEqual.
        __import__("sys").modules["unittest.util"]._MAX_LENGTH = 999999999

    sys.exit(doctest.testmod()[0])
