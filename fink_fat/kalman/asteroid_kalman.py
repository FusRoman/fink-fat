import numpy as np
import warnings


class KalfAst:
    def __init__(
        self,
        id: int,
        initX: float,
        initY: float,
        initVx: float,
        initVy: float,
        state_transition,
    ) -> None:
        """
        Initalize the Kalman filter for the asteroids

        Parameters
        ----------
        initX : float
            right ascension coordinate
        initY : float
            declination coordinate
        initVx : float
            speed in right ascension
        initVy : float
            speed in declination
        state_transition : numpy array
            state transition matrix of the kalman filter
        """
        # Strong hypotesis: sso speed remains constant over a small tracking time window

        self.kf_id = id

        # Initialization of state matrices
        self.X = np.array(
            [
                [initX],
                [initY],
                [initVx],
                [initVy],
            ]
        )  # the mean state estimation of the previous step (k-1)
        self.H = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        )  # measurement matrix
        # self.R = np.eye(4)  # measurement noise covariance matrix
        error_pos = 1
        error_vel = 1
        self.R = np.diag((error_pos, error_pos, error_vel, error_vel))
        self.P = np.diag((1, 1, 1, 1))  # the state covariance of previous step (k-1)

        self.A = np.array(state_transition)  # the state transition matrix

        # self.Q = np.zeros(self.X.shape[0])  # the process noise covariance matrix
        self.B = np.eye(self.X.shape[0])  # the input effect matrix
        self.U = np.zeros((self.X.shape[0], 1))  # the control input

    def dec_warn(self, ra, dec, opp_ra, opp_dec):
        """
        raise a warning if the kalman prediction goes beyond the circle coordinates
        """
        warnings.warn(
            f"""\n!!! Warnings! Kalman filter for asteroids generate a bad prediction.
current kalman state:
{self.X}
current covariance:
{self.P}
Declination should be -90 < dec < 90, found: ra={ra}, dec={dec}
New position set to the opposite: new_ra={opp_ra}, new_dec={opp_dec}\n"""
        )

    def opposite_dec_2d(self, prediction):
        """
        If the kalman filter makes a prediction that goes beyond a circle,
        this function computes the opposite point on the circle.
        Used if the kalman filter makes one prediction at one time.

        Parameters
        ----------
        prediction : float
            the current prediction of the kalman filter

        Returns
        -------
        numpy array of float
            the opposite point on the circle
        """
        ra = prediction[0, 0]
        dec = prediction[1, 0]

        if np.any(dec > 90):
            opposite_ra = 360 - ra
            opposite_dec = 180 - dec
            self.dec_warn(ra, dec, opposite_ra, opposite_dec)
            prediction[0, 0] = opposite_ra
            prediction[1, 0] = opposite_dec

        if np.any(dec < -90):
            opposite_ra = 360 - ra
            opposite_dec = -(dec + 180)

            self.dec_warn(ra, dec, opposite_ra, opposite_dec)

            prediction[0, 0] = opposite_ra
            prediction[1, 0] = opposite_dec

        return prediction

    def opposite_dec_3d(
        self,
        prediction: float,
    ):
        """
        If the kalman filter makes a prediction that goes beyond a circle,
        this function computes the opposite point on the circle.
        Used if the kalman filter makes multiples predictions.

        Parameters
        ----------
        prediction : float
            the current prediction of the kalman filter

        Returns
        -------
        numpy array of float
            the opposite point on the circle
        """
        ra = prediction[:, 0, 0]
        dec = prediction[:, 1, 0]

        if np.any(dec > 90):
            idx_dec = np.where(dec > 90)
            opposite_ra = 360 - ra[idx_dec]
            opposite_dec = 180 - dec[idx_dec]
            self.dec_warn(ra, dec, opposite_ra, opposite_dec)
            ra[idx_dec] = opposite_ra
            dec[idx_dec] = opposite_dec

        if np.any(dec < -90):
            idx_dec = np.where(dec < -90)
            opposite_ra = 360 - ra[idx_dec]
            opposite_dec = -(dec[idx_dec] + 180)
            self.dec_warn(ra, dec, opposite_ra, opposite_dec)
            ra[idx_dec] = opposite_ra
            dec[idx_dec] = opposite_dec

        return prediction

    def predict(self, A):
        """
        Make the prediction of the kalman filter based on the previous estimations.

        Parameters
        ----------
        A : numpy matrix of float
            a state transition matrix

        Returns
        -------
        numpy vector of float
            the predictions and the errors
        """
        prediction = np.dot(A, self.X) + np.dot(self.B, self.U)

        if len(np.shape(A)) == 2:
            prediction = self.opposite_dec_2d(prediction)
            # print("---------------------------------------------------------")
            # print(A.T)
            # print()
            # print(np.dot(self.P, A.T))
            # print()
            # print(np.dot(A, np.dot(self.P, A.T)))
            # print("---------------------------------------------------------")
            P = np.dot(A, np.dot(self.P, A.T))  # + self.Q

        elif len(np.shape(A)) == 3:
            prediction = self.opposite_dec_3d(prediction)

            P = np.array([self.P for _ in range(A.shape[0])])
            # Q_exp = np.array([np.eye(A.shape[-1]) for _ in range(A.shape[0])])
            f1 = np.flip(P @ A, (1, 2))
            P = A @ f1  # + Q_exp
            # TODO
            # the computation of P bug here when shape of A == 3
            # fix that

            # print("===============================================================")
            # # print(A)
            # # print()
            # # print(A.T)
            # # print()
            # print(f1)
            # print()
            # print((A @ f1))
            # print()
            # print(P)
            # print("===============================================================")

        return prediction, P

    def update(self, Y, transition_update=None):
        """
        Update the kalman filter with a real data point

        Parameters
        ----------
        Y : numpy vector of float
            the new data point
        transition_update : numpy matrix of float, optional
            the state transition matrix, by default None

        Returns
        -------
        multiple numy array
            the status of the update

        Raises
        ------
        Exception
            error of the state transition matrix has not the same shape
            as the one stored in this kalman filter.
        """
        pred_x, P = self.predict(transition_update)

        IM = np.dot(self.H, pred_x)  # mean of predictive distribution of Y
        IS = self.R + np.dot(self.H, np.dot(P, self.H.T))  # predictive mean of Y
        K = np.dot(P, np.dot(self.H.T, np.linalg.inv(IS)))  # the kalman gain matrix

        self.X = pred_x + np.dot(K, (Y - IM))
        self.P = np.dot((np.eye(4) - np.dot(K, self.H)), self.P)

        LH = KalfAst.gauss(
            Y, IM, IS
        )  # the predictive probability (likelihood) of measurement assuming gaussian process

        # custom transition matrix update
        if transition_update is not None and np.shape(self.A) == np.shape(
            transition_update
        ):
            self.A = transition_update
        elif transition_update is not None:
            raise Exception(
                "the new transition matrix has not the same shape as the previous one. old:{} != new:{}".format(
                    np.shape(self.A),
                    np.shape(transition_update),
                )
            )

        return (K, IM, IS, LH)

    def gauss(X, M, S):
        """
        Compute the probability that the new point is a good point for this kalman filter.

        Parameters
        ----------
        X : numpy vector of float
            the new real data point
        M : numpy vector of float
            mean of predictive distribution of X
        S : numpy vector of float
            predictive mean of X

        Returns
        -------
        float
            the probability that X is the good point to add assuming a gaussian process.
        """
        if M.shape[1] == 1:
            DX = X - np.tile(M, X.shape[1])

            E = 0.5 * np.sum(DX * (np.dot(np.linalg.inv(S), DX)), axis=0)

            E = (
                E
                + 0.5 * M.shape[0] * np.log(2 * np.pi)
                + 0.5 * np.log(np.linalg.det(S))
            )
            P = np.exp(-E)
        elif X.shape[1] == 1:
            DX = np.tile(X, M.shape[1]) - M

            E = 0.5 * np.sum(DX * (np.dot(np.linalg.inv(S), DX)), axis=0)

            E = (
                E
                + 0.5 * M.shape[0] * np.log(2 * np.pi)
                + 0.5 * np.log(np.linalg.det(S))
            )

            P = np.exp(-E)
        else:
            DX = X - M
            E = 0.5 * np.dot(DX.T, np.dot(np.linalg.inv(S), DX))
            E = (
                E
                + 0.5 * M.shape[0] * np.log(2 * np.pi)
                + 0.5 * np.log(np.linalg.det(S))
            )
            P = np.exp(-E)
        return (
            P[0],
            E[0],
        )

    def __repr__(self) -> str:
        return f"""
identifier: {self.kf_id}
@kalman: {hex(id(self))}

X state:
{self.X}


P state:
{self.P}


current A state:
{self.A}
"""

    def __str__(self) -> str:
        return f"kf {self.kf_id}"
