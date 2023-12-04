import numpy as np
import pandas as pd
from fink_fat.kalman.asteroid_kalman import KalfAst


def predictions(
    trajectory_id: int,
    kalman: KalfAst,
    dt: list,
):
    """
    Make predictions of the kalman filters for all dt.

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
    """
    # print(f"kalmanDfPrediction dt: {dt}")
    A = np.array(
        [
            [
                [1, 0, el, 0],
                [0, 1, 0, el],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
            for el in dt
        ]
    )

    pred, P = kalman.predict(A)

    if len(np.shape(A)) == 2:
        pred_coord = pred[:2, 0]
        error = np.sqrt(np.diag(P))[:2]
        return [trajectory_id, pred_coord[0], pred_coord[1], error[0], error[1]]
    elif len(np.shape(A)) == 3:
        pred_coord = pred[:, :2, 0]
        error = np.sqrt(
            np.diagonal(
                P,
                axis1=1,
                axis2=2,
            )[:, :2]
        )

        return [
            trajectory_id,
            pred_coord[:, 0],
            pred_coord[:, 1],
            error[:, 0],
            error[:, 1],
        ]


def kalmanDf_prediction(kalman_pdf: pd.DataFrame, jd: list) -> pd.DataFrame:
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
    """
    pred_list = [
        predictions(traj_id, kalman, jd - last_jd)
        for traj_id, kalman, last_jd in zip(
            kalman_pdf["trajectory_id"],
            kalman_pdf["kalman"],
            kalman_pdf["jd_1"],
        )
    ]

    pdf_prediction = pd.DataFrame(
        pred_list,
        columns=["trajectory_id", "ra", "dec", "delta_ra", "delta_dec"],
    )

    return pdf_prediction.explode(["ra", "dec", "delta_ra", "delta_dec"])
