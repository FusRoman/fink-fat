import pandas as pd

from fink_fat.kalman.asteroid_kalman import KalfAst


def data_test_1():
    trajectory_df = pd.DataFrame(
        {
            "trajectory_id": [0, 0, 0, 1, 1],
            "objectId": ["a", "b", "c", "d", "e"],
            "jd": [0, 1, 2, 3, 4],
        }
    )

    ra_0 = [0, 1]
    dec_0 = [0, 1]
    ra_1 = [2, 3]
    dec_1 = [2, 3]
    jd_0 = [0, 2]
    jd_1 = [1, 3]
    mag_1 = [16.3, 18.4]
    fid_1 = [1, 1]

    dt = []
    vel_ra = []
    vel_dec = []
    kal = []
    for i in range(len(ra_0)):
        dt.append(jd_1[i] - jd_0[i])
        vel_ra.append((ra_1[i] - ra_0[i]) / dt[i])
        vel_dec.append((dec_1[i] - dec_0[i]) / dt[i])
        kalman = KalfAst(
            i,
            ra_1[i],
            dec_1[i],
            vel_ra[i],
            vel_dec[i],
            [[1, 0, dt[i], 0], [0, 1, 0, dt[i]], [0, 0, 1, 0], [0, 0, 0, 1]],
        )
        kal.append(kalman)

    kalman_pdf = pd.DataFrame(
        {
            "trajectory_id": [0, 1],
            "ra_0": ra_0,
            "dec_0": dec_1,
            "ra_1": ra_1,
            "dec_1": dec_1,
            "jd_0": jd_0,
            "jd_1": jd_1,
            "mag_1": mag_1,
            "fid_1": fid_1,
            "dt": dt,
            "vel_ra": vel_ra,
            "vel_dec": vel_dec,
            "kalman": kal,
        }
    )

    new_alerts = pd.DataFrame(
        {
            "objectId": ["m", "n", "k", "l"],
            "candid": [0, 1, 2, 3],
            "ra": [3, 4, 5, 6],
            "dec": [3, 4, 5, 6],
            "jd": [10, 11, 12, 13],
            "nid": [5, 5, 5, 5],
            "fid": [1, 2, 1, 1],
            "ssnamenr": ["1", "1", "2", "3"],
            "ssdistnr": [1, 1, 1, 1],
            "magpsf": [17.5, 21.2, 19.6, 14.3],
            "trajectory_id": [0, 0, 1, 1],
            "ffdistnr": [[0.1], [0.3], [0.1], [0.5]],
            "estimator_id": [[0], [0], [1], [1]],
        }
    )

    return trajectory_df, kalman_pdf, new_alerts


def data_test_2():
    trajectory_df = pd.DataFrame(
        {
            "trajectory_id": [0, 0, 0, 1, 1],
            "objectId": ["a", "b", "c", "d", "e"],
            "jd": [0, 1, 2, 3, 4],
        }
    )

    ra_0 = [0, 1]
    dec_0 = [0, 1]
    ra_1 = [2, 3]
    dec_1 = [2, 3]
    jd_0 = [0, 2]
    jd_1 = [1, 3]
    mag_1 = [16.3, 18.4]
    fid_1 = [1, 1]

    dt = []
    vel_ra = []
    vel_dec = []
    kal = []
    for i in range(len(ra_0)):
        dt.append(jd_1[i] - jd_0[i])
        vel_ra.append((ra_1[i] - ra_0[i]) / dt[i])
        vel_dec.append((dec_1[i] - dec_0[i]) / dt[i])
        kalman = KalfAst(
            i,
            ra_1[i],
            dec_1[i],
            vel_ra[i],
            vel_dec[i],
            [[1, 0, dt[i], 0], [0, 1, 0, dt[i]], [0, 0, 1, 0], [0, 0, 0, 1]],
        )
        kal.append(kalman)

    kalman_pdf = pd.DataFrame(
        {
            "trajectory_id": [0, 1],
            "ra_0": ra_0,
            "dec_0": dec_1,
            "ra_1": ra_1,
            "dec_1": dec_1,
            "jd_0": jd_0,
            "jd_1": jd_1,
            "mag_1": mag_1,
            "fid_1": fid_1,
            "dt": dt,
            "vel_ra": vel_ra,
            "vel_dec": vel_dec,
            "kalman": kal,
        }
    )

    new_alerts = pd.DataFrame(
        {
            "objectId": ["m", "n", "k", "l"],
            "candid": [0, 1, 2, 3],
            "ra": [3, 4, 5, 6],
            "dec": [3, 4, 5, 6],
            "jd": [10, 11, 12, 13],
            "nid": [5, 5, 5, 5],
            "fid": [1, 2, 1, 1],
            "ssnamenr": ["1", "1", "2", "3"],
            "ssdistnr": [1, 1, 1, 1],
            "magpsf": [17.5, 21.2, 19.6, 14.3],
            "trajectory_id": [-1, -1, -1, -1],
            "roid": [3, 3, 3, 3],
            "t_estimator": ["kalman", "kalman", "kalman", "kalman"],
            "ffdistnr": [[0.1], [0.3], [0.1], [0.5]],
            "estimator_id": [[1], [0], [1], [0]],
        }
    )

    return trajectory_df, kalman_pdf, new_alerts


def data_test_3():
    trajectory_df = pd.DataFrame(
        {
            "trajectory_id": [0, 0, 0, 1, 1, 2, 2, 2, 2],
            "objectId": ["a", "b", "c", "d", "e", "ab", "ac", "ad", "af"],
            "jd": [0, 1, 2, 3, 4, 5, 6, 7, 8],
        }
    )

    ra_0 = [0, 1, 10]
    dec_0 = [0, 1, 10]
    ra_1 = [2, 3, 12]
    dec_1 = [2, 3, 12]
    jd_0 = [0, 2, 5]
    jd_1 = [1, 3, 7]
    mag_1 = [16.3, 18.4, 18.2]
    fid_1 = [1, 1, 2]

    dt = []
    vel_ra = []
    vel_dec = []
    kal = []
    for i in range(len(ra_0)):
        dt.append(jd_1[i] - jd_0[i])
        vel_ra.append((ra_1[i] - ra_0[i]) / dt[i])
        vel_dec.append((dec_1[i] - dec_0[i]) / dt[i])
        kalman = KalfAst(
            i,
            ra_1[i],
            dec_1[i],
            vel_ra[i],
            vel_dec[i],
            [[1, 0, dt[i], 0], [0, 1, 0, dt[i]], [0, 0, 1, 0], [0, 0, 0, 1]],
        )
        kal.append(kalman)

    kalman_pdf = pd.DataFrame(
        {
            "trajectory_id": [0, 1, 2],
            "ra_0": ra_0,
            "dec_0": dec_1,
            "ra_1": ra_1,
            "dec_1": dec_1,
            "jd_0": jd_0,
            "jd_1": jd_1,
            "mag_1": mag_1,
            "fid_1": fid_1,
            "dt": dt,
            "vel_ra": vel_ra,
            "vel_dec": vel_dec,
            "kalman": kal,
        }
    )

    new_alerts = pd.DataFrame(
        {
            "objectId": ["m", "n", "k", "l"],
            "candid": [0, 1, 2, 3],
            "ra": [3, 4, 5, 6],
            "dec": [3, 4, 5, 6],
            "jd": [10, 11, 12, 13],
            "nid": [5, 5, 5, 5],
            "fid": [1, 2, 1, 1],
            "ssnamenr": ["1", "1", "2", "3"],
            "ssdistnr": [1, 1, 1, 1],
            "magpsf": [17.5, 21.2, 19.6, 14.3],
            "trajectory_id": [-1, -1, -1, -1],
            "roid": [3, 3, 3, 3],
            "t_estimator": ["kalman", "kalman", "kalman", "kalman"],
            "ffdistnr": [[0.1, 0.3], [0.3], [0.1], [0.5, 0.4]],
            "estimator_id": [[1, 2], [0], [1], [0, 2]],
        }
    )

    return trajectory_df, kalman_pdf, new_alerts


def data_test_4():
    trajectory_df = pd.DataFrame(
        {
            "trajectory_id": [0, 0, 0, 1, 1, 2, 2, 2, 2],
            "objectId": ["a", "b", "c", "d", "e", "ab", "ac", "ad", "af"],
            "jd": [0, 1, 2, 3, 4, 5, 6, 7, 8],
        }
    )

    ra_0 = [0, 1, 10]
    dec_0 = [0, 1, 10]
    ra_1 = [2, 3, 12]
    dec_1 = [2, 3, 12]
    jd_0 = [0, 2, 5]
    jd_1 = [1, 3, 7]
    mag_1 = [16.3, 18.4, 18.2]
    fid_1 = [1, 1, 2]

    dt = []
    vel_ra = []
    vel_dec = []
    kal = []
    for i in range(len(ra_0)):
        dt.append(jd_1[i] - jd_0[i])
        vel_ra.append((ra_1[i] - ra_0[i]) / dt[i])
        vel_dec.append((dec_1[i] - dec_0[i]) / dt[i])
        kalman = KalfAst(
            i,
            ra_1[i],
            dec_1[i],
            vel_ra[i],
            vel_dec[i],
            [[1, 0, dt[i], 0], [0, 1, 0, dt[i]], [0, 0, 1, 0], [0, 0, 0, 1]],
        )
        kal.append(kalman)

    kalman_pdf = pd.DataFrame(
        {
            "trajectory_id": [0, 1, 2],
            "ra_0": ra_0,
            "dec_0": dec_1,
            "ra_1": ra_1,
            "dec_1": dec_1,
            "jd_0": jd_0,
            "jd_1": jd_1,
            "mag_1": mag_1,
            "fid_1": fid_1,
            "dt": dt,
            "vel_ra": vel_ra,
            "vel_dec": vel_dec,
            "kalman": kal,
        }
    )

    new_alerts = pd.DataFrame(
        {
            "objectId": ["m", "n", "k", "l", "ba", "bc"],
            "candid": [0, 1, 2, 3, 4, 5],
            "ra": [3, 4, 5, 6, 7, 8],
            "dec": [3, 4, 5, 6, 7, 8],
            "jd": [10, 11, 12, 13, 14, 15],
            "nid": [5, 5, 5, 5, 5, 5],
            "fid": [1, 2, 1, 1, 2, 1],
            "ssnamenr": ["1", "1", "2", "3", "4", "5"],
            "ssdistnr": [1, 1, 1, 1, 1, 1],
            "magpsf": [17.5, 21.2, 19.6, 14.3, 17.5, 19.8],
            "trajectory_id": [-1, -1, -1, -1, 0, 0],
            "roid": [3, 3, 3, 3, 3, 3],
            "t_estimator": ["kalman", "kalman", "kalman", "kalman", "kalman", "kalman"],
            "ffdistnr": [[0.1, 0.3], [0.3], [0.1], [0.5, 0.4], [0.2], [0.4]],
            "estimator_id": [[1, 2], [0], [1], [0, 2], [0], [0]],
        }
    )

    return trajectory_df, kalman_pdf, new_alerts


def data_test_5():
    trajectory_df = pd.DataFrame(
        {
            "trajectory_id": [0, 0, 0, 1, 1, 2, 2, 2, 2],
            "objectId": ["a", "b", "c", "d", "e", "ab", "ac", "ad", "af"],
            "jd": [0, 1, 2, 3, 4, 5, 6, 7, 8],
        }
    )

    ra_0 = [0, 1, 10]
    dec_0 = [0, 1, 10]
    ra_1 = [2, 3, 12]
    dec_1 = [2, 3, 12]
    jd_0 = [0, 2, 5]
    jd_1 = [1, 3, 7]
    mag_1 = [16.3, 18.4, 18.2]
    fid_1 = [1, 1, 2]

    dt = []
    vel_ra = []
    vel_dec = []
    kal = []
    for i in range(len(ra_0)):
        dt.append(jd_1[i] - jd_0[i])
        vel_ra.append((ra_1[i] - ra_0[i]) / dt[i])
        vel_dec.append((dec_1[i] - dec_0[i]) / dt[i])
        kalman = KalfAst(
            i,
            ra_1[i],
            dec_1[i],
            vel_ra[i],
            vel_dec[i],
            [[1, 0, dt[i], 0], [0, 1, 0, dt[i]], [0, 0, 1, 0], [0, 0, 0, 1]],
        )
        kal.append(kalman)

    kalman_pdf = pd.DataFrame(
        {
            "trajectory_id": [0, 1, 2],
            "ra_0": ra_0,
            "dec_0": dec_1,
            "ra_1": ra_1,
            "dec_1": dec_1,
            "jd_0": jd_0,
            "jd_1": jd_1,
            "mag_1": mag_1,
            "fid_1": fid_1,
            "dt": dt,
            "vel_ra": vel_ra,
            "vel_dec": vel_dec,
            "kalman": kal,
        }
    )

    new_alerts = pd.DataFrame(
        {
            "objectId": ["m", "n", "k", "l", "ba", "bc"],
            "candid": [0, 1, 2, 3, 4, 5],
            "ra": [3, 4, 5, 6, 7, 8],
            "dec": [3, 4, 5, 6, 7, 8],
            "jd": [10, 11, 12, 13, 14, 15],
            "nid": [5, 5, 5, 5, 5, 5],
            "fid": [1, 2, 1, 1, 2, 1],
            "ssnamenr": ["1", "1", "2", "3", "4", "5"],
            "ssdistnr": [1, 1, 1, 1, 1, 1],
            "magpsf": [17.5, 21.2, 19.6, 14.3, 17.5, 19.8],
            "trajectory_id": [1, 1, 2, 2, 0, 0],
            "roid": [3, 3, 3, 3, 3, 3],
            "t_estimator": ["kalman", "kalman", "kalman", "kalman", "kalman", "kalman"],
            "ffdistnr": [[0.1, 0.3], [0.3], [0.1], [0.5, 0.4], [0.2], [0.4]],
            "estimator_id": [[1, 2], [0], [1], [0, 2], [0], [0]],
        }
    )

    return trajectory_df, kalman_pdf, new_alerts


def data_test_6():
    trajectory_df = pd.DataFrame(
        {
            "trajectory_id": [0, 0, 0, 1, 1, 2, 2, 2, 2],
            "objectId": ["a", "b", "c", "d", "e", "ab", "ac", "ad", "af"],
            "jd": [0, 1, 2, 3, 4, 5, 6, 7, 8],
        }
    )

    ra_0 = [0, 1, 10]
    dec_0 = [0, 1, 10]
    ra_1 = [2, 3, 12]
    dec_1 = [2, 3, 12]
    jd_0 = [0, 2, 5]
    jd_1 = [1, 3, 7]
    mag_1 = [16.3, 18.4, 18.2]
    fid_1 = [1, 1, 2]

    dt = []
    vel_ra = []
    vel_dec = []
    kal = []
    for i in range(len(ra_0)):
        dt.append(jd_1[i] - jd_0[i])
        vel_ra.append((ra_1[i] - ra_0[i]) / dt[i])
        vel_dec.append((dec_1[i] - dec_0[i]) / dt[i])
        kalman = KalfAst(
            i,
            ra_1[i],
            dec_1[i],
            vel_ra[i],
            vel_dec[i],
            [[1, 0, dt[i], 0], [0, 1, 0, dt[i]], [0, 0, 1, 0], [0, 0, 0, 1]],
        )
        kal.append(kalman)

    kalman_pdf = pd.DataFrame(
        {
            "trajectory_id": [0, 1, 2],
            "ra_0": ra_0,
            "dec_0": dec_1,
            "ra_1": ra_1,
            "dec_1": dec_1,
            "jd_0": jd_0,
            "jd_1": jd_1,
            "mag_1": mag_1,
            "fid_1": fid_1,
            "dt": dt,
            "vel_ra": vel_ra,
            "vel_dec": vel_dec,
            "kalman": kal,
        }
    )

    new_alerts = pd.DataFrame(
        {
            "objectId": ["m", "n", "k", "l", "ba", "bc"],
            "candid": [0, 1, 2, 3, 4, 5],
            "ra": [3, 4, 5, 6, 7, 8],
            "dec": [3, 4, 5, 6, 7, 8],
            "jd": [10, 11, 12, 13, 14, 15],
            "nid": [5, 5, 5, 5, 5, 5],
            "fid": [1, 2, 1, 1, 2, 1],
            "ssnamenr": ["1", "1", "2", "3", "4", "5"],
            "ssdistnr": [1, 1, 1, 1, 1, 1],
            "magpsf": [17.5, 21.2, 19.6, 14.3, 17.5, 19.8],
            "trajectory_id": [-1, -1, 2, 2, 0, 0],
            "roid": [3, 3, 3, 3, 3, 3],
            "t_estimator": ["kalman", "kalman", "kalman", "kalman", "kalman", "kalman"],
            "ffdistnr": [[0.1, 0.3], [0.3, 0.2], [0.1], [0.5, 0.4], [0.2], [0.4]],
            "estimator_id": [[1, 2], [0, 1], [1], [0, 2], [0], [0]],
        }
    )

    return trajectory_df, kalman_pdf, new_alerts


def data_test_7():
    trajectory_df = pd.DataFrame(
        {
            "trajectory_id": [0, 0, 0, 1, 1, 2, 2, 2, 2],
            "objectId": ["a", "b", "c", "d", "e", "ab", "ac", "ad", "af"],
            "jd": [0, 1, 2, 3, 4, 5, 6, 7, 8],
        }
    )

    ra_0 = [0, 1, 10]
    dec_0 = [0, 1, 10]
    ra_1 = [2, 3, 12]
    dec_1 = [2, 3, 12]
    jd_0 = [0, 2, 5]
    jd_1 = [1, 3, 7]
    mag_1 = [16.3, 18.4, 18.2]
    fid_1 = [1, 1, 2]

    dt = []
    vel_ra = []
    vel_dec = []
    kal = []
    for i in range(len(ra_0)):
        dt.append(jd_1[i] - jd_0[i])
        vel_ra.append((ra_1[i] - ra_0[i]) / dt[i])
        vel_dec.append((dec_1[i] - dec_0[i]) / dt[i])
        kalman = KalfAst(
            i,
            ra_1[i],
            dec_1[i],
            vel_ra[i],
            vel_dec[i],
            [[1, 0, dt[i], 0], [0, 1, 0, dt[i]], [0, 0, 1, 0], [0, 0, 0, 1]],
        )
        kal.append(kalman)

    kalman_pdf = pd.DataFrame(
        {
            "trajectory_id": [0, 1, 2],
            "ra_0": ra_0,
            "dec_0": dec_1,
            "ra_1": ra_1,
            "dec_1": dec_1,
            "jd_0": jd_0,
            "jd_1": jd_1,
            "mag_1": mag_1,
            "fid_1": fid_1,
            "dt": dt,
            "vel_ra": vel_ra,
            "vel_dec": vel_dec,
            "kalman": kal,
        }
    )

    new_alerts = pd.DataFrame(
        {
            "objectId": ["m", "n", "k", "l", "ba", "bc"],
            "candid": [0, 1, 2, 3, 4, 5],
            "ra": [3, 4, 5, 6, 7, 8],
            "dec": [3, 4, 5, 6, 7, 8],
            "jd": [10, 11, 12, 13, 14, 15],
            "nid": [5, 5, 5, 5, 5, 5],
            "fid": [1, 2, 1, 1, 2, 1],
            "ssnamenr": ["1", "1", "2", "3", "4", "5"],
            "ssdistnr": [1, 1, 1, 1, 1, 1],
            "magpsf": [17.5, 21.2, 19.6, 14.3, 17.5, 19.8],
            "trajectory_id": [0, 0, 2, 2, 1, 1],
            "roid": [3, 3, 3, 3, 3, 3],
            "t_estimator": ["kalman", "kalman", "kalman", "kalman", "kalman", "kalman"],
            "ffdistnr": [[0.1, 0.3], [0.3, 0.2], [0.1], [0.5], [0.2], [0.4]],
            "estimator_id": [[1, 2], [1, 2], [0], [0], [0], [0]],
        }
    )

    return trajectory_df, kalman_pdf, new_alerts
