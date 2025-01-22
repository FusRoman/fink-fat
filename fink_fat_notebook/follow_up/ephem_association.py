import numpy as np
import pandas as pd

import time as t
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy.coordinates import search_around_sky
import datetime
from sbpy.data import Orbit, Ephem
import io
import math
import requests
import astropy.units as u
import matplotlib.pyplot as plt

from fink_fat.orbit_fitting.orbfit_local import compute_df_orbit_param


def df_to_orb(df_orb):
    prep_to_orb = df_orb[
        [
            "ssoCandId",
            "ref_epoch",
            "a",
            "e",
            "i",
            "long. node",
            "arg. peric",
            "mean anomaly",
        ]
    ].rename(
        {
            "ssoCandId": "targetname",
            "long. node": "node",
            "arg. peric": "argper",
            "mean anomaly": "M",
            "ref_epoch": "epoch",
        },
        axis=1,
    )

    prep_to_orb["orbtype"] = "KEP"

    prep_to_orb["H"] = 14.45
    prep_to_orb["G"] = 0.15

    orb_dict = prep_to_orb.to_dict(orient="list")

    orb_dict["a"] = orb_dict["a"] * u.au
    orb_dict["i"] = orb_dict["i"] * u.deg
    orb_dict["node"] = orb_dict["node"] * u.deg
    orb_dict["argper"] = orb_dict["argper"] * u.deg
    orb_dict["M"] = orb_dict["M"] * u.deg
    orb_dict["epoch"] = Time(orb_dict["epoch"], format="jd")
    orb_dict["H"] = orb_dict["H"] * u.mag

    ast_orb_db = Orbit.from_dict(orb_dict)
    return ast_orb_db


def ephem_cross_match(pdf_alerts, orbit, sep_assoc):

    ephem_epochs = Time(pdf_alerts["jd"], format="jd")
    ephem = Ephem.from_oo(
        df_to_orb(orbit), epochs=ephem_epochs, location="I41", scope="full"
    ).table.to_pandas()

    ztf_coord = SkyCoord(pdf_alerts["ra"], pdf_alerts["dec"], unit=u.degree)
    ephem_coord = SkyCoord(ephem["RA"], ephem["DEC"], unit=u.degree)

    idx1, idx2, sep, _ = search_around_sky(
        ztf_coord, ephem_coord, sep_assoc * u.arcsecond
    )

    ztf_assoc = pdf_alerts.iloc[idx1]

    return ztf_assoc.drop_duplicates("candid").reset_index(drop=True)


def get_stats_sso():
    r = requests.post(
        "https://api.fink-portal.org/api/v1/statistics",
        json={"date": "", "output-format": "json"},
    )

    pdf = pd.read_json(io.BytesIO(r.content))

    pdf["jd"] = [Time(f"{el[4:8]}-{el[8:10]}-{el[10:12]}").jd for el in pdf["key:key"]]
    return pdf[["class:Solar System candidate", "jd", "key:key"]]


def get_alerts(start, stop, nb_alerts):
    request_columns = (
        "i:ra,i:dec,i:jd,i:nid,i:fid,i:candid,i:magpsf,i:sigmapsf,i:ssnamenr,i:objectId"
    )
    required_columns = [
        "ra",
        "dec",
        "jd",
        "nid",
        "fid",
        "candid",
        "magpsf",
        "sigmapsf",
        "ssnamenr",
        "objectId",
    ]

    # with hbase correction
    r = requests.post(
        "https://api.fink-portal.org/api/v1/latests",
        json={
            "class": "Solar System candidate",
            "n": "{}".format(nb_alerts),
            "startdate": str(start - datetime.timedelta(hours=1)),
            "stopdate": str(stop + datetime.timedelta(hours=1)),
            "columns": request_columns,
        },
    )
    pdf = pd.read_json(io.BytesIO(r.content))

    # if len(pdf) != nb_alerts:
    #     print("--- Error ---")
    #     print("nb alert from API latests {} different from API stats {}".format(len(pdf), nb_alerts))
    #     print("------")

    if len(pdf) == 0:
        return pd.DataFrame(columns=required_columns)

    pdf.columns = [cols[2:] for cols in pdf.columns]
    return pdf[required_columns]


def ephem_association_new(traj, orbit, step=15, separation=5):

    # stats_sso = get_stats_sso()
    hist_orbit = []

    def ephem_association_aux(
        traj, orbit, step=15, separation=5, recursive_start=None, backward=False
    ):

        first_obs = math.modf(traj["jd"].values[0])[1] + 0.5
        last_obs = math.modf(traj["jd"].values[-1])[1] + 0.5

        if recursive_start is None:
            if backward:
                start = Time(first_obs - step, format="jd")
                stop = Time(first_obs, format="jd")
            else:
                start = Time(last_obs, format="jd")
                stop = Time(last_obs + step, format="jd")
        else:
            if backward:
                start = Time(recursive_start - step, format="jd")
                stop = Time(recursive_start, format="jd")
            else:
                start = Time(recursive_start, format="jd")
                stop = Time(recursive_start + step, format="jd")

        # nb_alerts = int(np.sum(
        #     stats_sso[(stats_sso["jd"] >= start_jd) & (stats_sso["jd"] < stop_jd)]["class:Solar System candidate"]
        # ))

        pdf_alert = get_alerts(start.datetime, stop.datetime, 999999).astype(
            {"candid": int}
        )

        if len(pdf_alert) == 0:
            if backward:
                print("no new alerts from API, Quit ephem_assoc from backward .")
                return traj, orbit
            else:
                print("no new alerts from API, recursive backward")
                return ephem_association_aux(
                    traj, orbit, step=step, recursive_start=None, backward=True
                )

        ztf_assoc = ephem_cross_match(pdf_alert, orbit, separation).reset_index(
            drop=True
        )

        if len(ztf_assoc) == 0:
            if backward:
                return traj, orbit
            else:
                return ephem_association_aux(
                    traj, orbit, step=step, recursive_start=None, backward=True
                )

        traj_test = traj
        for (
            i,
            rows,
        ) in (
            ztf_assoc.iterrows()
        ):  # big issue with iterrows and the candid columns (cast problem)

            # try:
            #     coord = SkyCoord(rows["ra"], rows["dec"], unit=u.degree)
            #     epoch = Time(rows["jd"], format="jd")
            #     results = Skybot.cone_search(coord, 30*u.arcsec, epoch).to_pandas()[["Name", "centerdist"]]

            #     print(results)
            # except RuntimeError:
            rows["candid"] = ztf_assoc["candid"].values[i]

            # append additional required columns
            rows["assoc_tag"] = "E"
            rows["not_updated"] = True
            rows_dt = Time(rows["jd"], format="jd").datetime
            rows["last_assoc_date"] = f"{rows_dt.year}-{rows_dt.month}-{rows_dt.day}"
            rows["ssoCandId"] = traj["ssoCandId"].values[0]

            traj_extend = pd.concat([traj_test, pd.DataFrame(rows).T])

            traj_extend["trajectory_id"] = 1

            new_orbit = compute_df_orbit_param(
                traj_extend, 1, "/media/virtuelram/", 30, 20, float(rows["jd"]), 3
            )

            if (
                new_orbit["a"].values[0] != -1.0
                and new_orbit["rms_a"].values[0] != -1.0
            ):
                traj_test = traj_extend
                new_orbit["ssoCandId"] = orbit["ssoCandId"].values[0]
                new_orbit = new_orbit.drop("trajectory_id", axis=1)
                hist_orbit.append(new_orbit)
                orbit = new_orbit

        if backward:
            return ephem_association_aux(
                traj_test, orbit, step=step, recursive_start=start.jd, backward=True
            )
        else:
            return ephem_association_aux(
                traj_test, orbit, step=step, recursive_start=stop.jd
            )

    return ephem_association_aux(traj, orbit, step, separation), hist_orbit


def ephem_association_old(
    traj, orbit, step=15, separation=5, recursive_start=None, backward=False
):

    first_obs = traj["jd"].values[0]
    last_obs = traj["jd"].values[-1]

    if recursive_start is None:
        if backward:
            start = Time(first_obs - step, format="jd")
            stop = Time(first_obs, format="jd")
        else:
            start = Time(last_obs, format="jd")
            stop = Time(last_obs + step, format="jd")
    else:
        if backward:
            start = Time(recursive_start - step, format="jd")
            stop = Time(recursive_start, format="jd")
        else:
            start = Time(recursive_start, format="jd")
            stop = Time(recursive_start + step, format="jd")

    pdf_alert = get_alerts(start.datetime, stop.datetime, 999999).astype(
        {"candid": int}
    )

    if len(pdf_alert) == 0:
        return traj, orbit

    ztf_assoc = ephem_cross_match(pdf_alert, orbit, separation).reset_index(drop=True)

    if len(ztf_assoc) == 0:
        return traj, orbit

    traj_test = traj
    for (
        i,
        rows,
    ) in (
        ztf_assoc.iterrows()
    ):  # big issue with iterrows and the candid columns (cast problem)

        rows["assoc_tag"] = "E"
        # rows = rows.astype({"candid": int}, axis=1)
        rows["candid"] = ztf_assoc["candid"].values[i]

        traj_extend = traj_test.append(rows)

        traj_extend["ssoCandId"] = traj_test["ssoCandId"].values[0]
        traj_extend["trajectory_id"] = 1

        new_orbit = compute_df_orbit_param(
            traj_extend, 1, "/media/virtuelram/", 30, 20, float(rows["jd"]), 3
        )

        if new_orbit["a"].values[0] != -1.0 and new_orbit["rms_a"].values[0] != -1.0:
            traj_test = traj_extend
            new_orbit["ssoCandId"] = orbit["ssoCandId"].values[0]
            orbit = new_orbit

    if backward:
        return ephem_association_old(
            traj_test, orbit, step=step, recursive_start=start.jd, backward=True
        )
    else:
        return ephem_association_old(
            traj_test, orbit, step=step, recursive_start=stop.jd
        )


def extend_sso_traj(sso_lc, sso_orb, output_lc_path, output_orb_path):

    new_traj_list = []
    new_orbit_list = []

    for tr_id in sso_orb["ssoCandId"]:

        tmp_traj = sso_lc[sso_lc["ssoCandId"] == tr_id]
        tmp_orb = sso_orb[sso_orb["ssoCandId"] == tr_id]

        # forward
        new_traj, new_orbit = ephem_association_old(
            tmp_traj, tmp_orb, step=30, separation=30
        )
        new_traj = new_traj.astype({"candid": int})

        # backward
        new_traj, new_orbit = ephem_association_old(
            new_traj, new_orbit, step=30, separation=30, backward=True
        )
        new_traj = new_traj.astype({"candid": int})

        if len(new_traj) - len(tmp_traj) > 0:
            new_traj_list.append(new_traj)
            new_orbit_list.append(new_orbit)

    new_traj = pd.concat(new_traj_list).astype({"candid": int})
    new_orbit = pd.concat(new_orbit_list)

    new_traj.to_parquet(output_lc_path, index=False)
    new_orbit.to_parquet(output_orb_path, index=False)

    return new_traj, new_orbit


def extended_traj_table(
    extended_traj, extended_orbit, assoc_follow_up_table, suffix="_x"
):
    stats_pdf = (
        extended_traj.sort_values(["ssoCandId", "jd"])
        .groupby("ssoCandId")
        .agg(
            nb_point=("ra", len),
            last_observation=("last_assoc_date", lambda x: list(x)[-1]),
            observation_window=("jd", lambda x: list(x)[-1] - list(x)[0]),
        )
        .reset_index()
    )

    stats_with_id = assoc_follow_up_table.merge(
        stats_pdf, left_on="ssoCandId{}".format(suffix), right_on="ssoCandId"
    )

    mag_stats_dict = {
        "ssoCandId": [],
        "min_g": [],
        "max_g": [],
        "min_r": [],
        "max_r": [],
    }

    for ssoId in stats_pdf.ssoCandId:
        mag_stats_dict["ssoCandId"].append(ssoId)
        for filter in [1, 2]:
            tmp_pdf = extended_traj[
                (extended_traj["ssoCandId"] == ssoId) & (extended_traj["fid"] == filter)
            ]

            mag_stats_dict["min_{}".format("g" if filter == 1 else "r")].append(
                tmp_pdf["magpsf"].min()
            )
            mag_stats_dict["max_{}".format("g" if filter == 1 else "r")].append(
                tmp_pdf["magpsf"].max()
            )

    mag_pdf = pd.DataFrame(mag_stats_dict)

    data_table = stats_with_id.merge(mag_pdf, on="ssoCandId").merge(
        extended_orbit, on="ssoCandId"
    )

    header = """
|Fink Internal Id|Nb point|last observation|observation window(day)|magnitude<br>g:[min, max]<br>r:[min, max]|a (AU)|e|i (degree)|
|----------------|--------|----------------|-----------------------|-----------------------------------------|------|-|----------|
"""

    for _, rows in data_table.iterrows():
        header += "|15_2_2: {}<br>old: {}|{}|{}|{:.0f}|g:[{:.2f}, {:.2f}]<br>r:[{:.2f}, {:.2f}]|{:.3f} ± {:.3f}|{:.3f} ± {:.3f}|{:.3f} ± {:.3f}\n".format(
            rows["ssoCandId"],
            rows["ssoCandId_y"],
            rows["nb_point"],
            rows["last_observation"],
            rows["observation_window"],
            rows["min_g"],
            rows["max_g"],
            rows["min_r"],
            rows["max_r"],
            rows["a"],
            rows["rms_a"],
            rows["e"],
            rows["rms_e"],
            rows["i"],
            rows["rms_i"],
        )

    return header


def plot_hist_orb(hist_orbit, candId):

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    fig.suptitle(candId, y=0.9)
    fig.text(0.5, 0.08, "new added point", ha="center")

    tmp_hist = hist_orbit[hist_orbit["ssoCandId"] == candId]

    ax1.errorbar(
        np.arange(len(tmp_hist)),
        tmp_hist["a"],
        yerr=tmp_hist["rms_a"],
        marker="o",
        linestyle=":",
    )
    ax2.errorbar(
        np.arange(len(tmp_hist)),
        tmp_hist["e"],
        yerr=tmp_hist["rms_e"],
        marker="o",
        linestyle=":",
    )
    ax3.errorbar(
        np.arange(len(tmp_hist)),
        tmp_hist["i"],
        yerr=tmp_hist["rms_i"],
        marker="o",
        linestyle=":",
    )
    ax1.set_ylabel("semi major axis (AU)")

    ax2.set_ylabel("eccentricity")

    ax3.set_ylabel("inclination (degree)")
    plt.show()


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    traj = pd.read_parquet(
        "four_fink_fat_out/candidates/trajectory_orb.parquet"
    ).sort_values(["trajectory_id", "jd"])
    orb = pd.read_parquet("four_fink_fat_out/candidates/orbital.parquet")

    new_traj_list = []
    new_orbit_list = []

    unknown_ast_cand = [
        2907,
        2913,
        2916,
        2959,
        2967,
        3079,
        3084,
        3236,
        3620,
        3624,
        3631,
        3636,
        3644,
        3648,
        3662,
        3666,
        3682,
        3724,
        3736,
        3754,
        3764,
        3778,
        3779,
        3797,
        3823,
        3896,
        3897,
        3898,
        3954,
        3955,
        3963,
        3971,
        3972,
        3989,
        3999,
        4058,
        4062,
        4064,
        4069,
        4075,
        4084,
        4115,
        4116,
        4147,
        4163,
        4167,
        4168,
        4175,
        4254,
        4259,
        4352,
        4389,
        4526,
        4563,
        4946,
        5341,
        6927,
        7194,
        7585,
        7685,
        7742,
        7810,
        7811,
        8354,
        8399,
        8836,
        13148,
        13157,
        14789,
        43815,
        56148,
        56154,
        56158,
    ]

    for tr_id in unknown_ast_cand:
        print("traj_id: {}".format(tr_id))

        tmp_traj = traj[traj["trajectory_id"] == tr_id]
        tmp_orb = orb[orb["trajectory_id"] == tr_id]

        t_before = t.time()
        # forward
        print("forward step")
        new_traj, new_orbit = ephem_association_old(
            tmp_traj, tmp_orb, step=30, separation=30
        )
        print()
        # backward
        print("backward step")
        new_traj, new_orbit = ephem_association_old(
            new_traj, new_orbit, step=30, separation=30, backward=True
        )

        if len(new_traj) - len(tmp_traj) > 0:
            new_traj_list.append(new_traj)
            new_orbit_list.append(new_orbit)

        print("diff old traj and new traj: {}".format(len(new_traj) - len(tmp_traj)))
        print("forward time: {}".format(t.time() - t_before))

        print()
        print("------")
        print()

    new_traj = pd.concat(new_traj_list)
    new_orbit = pd.concat(new_orbit_list)

    new_traj.to_parquet("ephem_traj_all_unknown.parquet", index=False)
    new_orbit.to_parquet("ephem_orbit_all_unknown.parquet", index=False)
