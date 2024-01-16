import numpy as np
import pandas as pd
import fink_fat.roid_fitting.utils_roid_fit as uf
import sys
import doctest


def predictions(
    trajectory_id: int,
    xp: np.ndarray,
    yp: np.ndarray,
    zp: np.ndarray,
    jd: list,
):
    """
    Make predictions of the fitting functions for all dt.

    Parameters
    ----------
    trajectory_id : int
        the id of the kalman filter
    kalman : KalfAst
        kalman filter used to make the predictions
    dt : list
        a list of float representing the delta time between the last point of the kalman filter
        and the time of the predictions.

    Returns
    -------
    numpy array
        contains the trajectory_id, the predict coordinates and the errors

    Examples
    --------

    >>> jd = [2460248.92125656, 2460252.92125656, 2460264.92125656, 2460293.92125656]
    >>> xp = [-3.34688754e-10,  9.73618005e-08,  2.02499018e+03]
    >>> yp = [-2.83996702e-10, -9.35467105e-08,  1.71741808e+03]
    >>> zp = [-5.10962759e-10, -1.17640302e-08,  3.09072379e+03]

    >>> predictions(0, xp, yp, zp, jd)
    [0, array([252.01023015, 251.87293013, 251.46736539, 250.5250238 ]), array([-47.69832558, -47.72534314, -47.80403179, -47.98046663])]
    """

    x = np.poly1d(xp)(jd)
    y = np.poly1d(yp)(jd)
    z = np.poly1d(zp)(jd)
    trajectory_id = np.ones(len(x), dtype=int) * trajectory_id
    return np.array([trajectory_id, x, y, z])


def fitroid_prediction(fit_pdf: pd.DataFrame, jd: list) -> pd.DataFrame:
    """
    Make predictions for all the kalman contains in the dataframe and for all the jd contains in the list

    Parameters
    ----------
    kalman_pdf : pd.DataFrame
        dataframe containing the kalman filters
    jd : list
        a list of jd for each predictions

    Returns
    -------
    pd.DataFrame
        a dataframe containing the predictions for each kalman at each jd.

    Examples
    --------
    >>> pdf_fit = pd.DataFrame(
    ... {
    ...     'trajectory_id': [0, 2],
    ...     'ra_0': [316.1443573, 315.1834854],
    ...     'dec_0': [-4.9800711, -5.2264238],
    ...     'ra_1': [316.141868, 315.1800646],
    ...     'dec_1': [-4.9829839, -5.2278022],
    ...     'jd_0': [2459458.7321528, 2459458.7321528],
    ...     'jd_1': [2459458.7523032, 2459458.7523032],
    ...     'mag_1': [18.198163986206055, 19.33443260192871],
    ...     'fid_1': [1, 1],
    ...     'xp': [
    ...         [-3.34688754e-10,  9.73618005e-08,  2.02499018e+03],
    ...         [-4.38514400e-10,  9.57428568e-08,  2.65301704e+03]],
    ...     'yp': [
    ...         [-2.83996702e-10, -9.35467105e-08,  1.71741808e+03],
    ...         [-4.09895419e-10, -9.51293965e-08,  2.47896373e+03]],
    ...     'zp': [
    ...         [-5.10962759e-10, -1.17640302e-08,  3.09072379e+03],
    ...         [-2.41704699e-10, -1.23451582e-08,  1.46199583e+03]]
    ... })
    >>> jd = [2460248.92125656, 2460252.92125656, 2460264.92125656, 2460293.92125656]

    >>> fitroid_prediction(pdf_fit, jd)
       trajectory_id          ra        dec
    0              0   252.01023 -47.698326
    0              0   251.87293 -47.725343
    0              0  251.467365 -47.804032
    0              0  250.525024 -47.980467
    1              2  246.497191 -22.380845
    1              2   246.39014 -22.386528
    1              2  246.074463 -22.402872
    1              2  245.343992 -22.438328
    """
    pred_list = np.hstack(
        [
            predictions(*args, jd)
            for args in zip(
                fit_pdf["trajectory_id"], fit_pdf["xp"], fit_pdf["yp"], fit_pdf["zp"]
            )
        ]
    )
    eq_coord = uf.xyz_to_equ(pred_list[1, :], pred_list[2, :], pred_list[3, :])

    pdf_prediction = pd.DataFrame(
        {
            "trajectory_id": pred_list[0, :],
            "ra": eq_coord.ra.deg,
            "dec": eq_coord.dec.deg,
        }
    )

    return pdf_prediction


if __name__ == "__main__":  # pragma: no cover
    if "unittest.util" in __import__("sys").modules:
        # Show full diff in self.assertEqual.
        __import__("sys").modules["unittest.util"]._MAX_LENGTH = 999999999

    sys.exit(doctest.testmod()[0])
