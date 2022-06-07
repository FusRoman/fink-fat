import datetime
import os
import numpy as np
import pandas as pd
import requests
from fink_utils.photometry.conversion import dc_mag
import shutil


def request_fink(
    object_class, startdate, stopdate, request_columns, verbose, nb_tries, current_tries
):
    r = requests.post(
        "https://fink-portal.org/api/v1/latests",
        json={
            "class": object_class,
            "n": "200000",
            "startdate": str(startdate),
            "stopdate": str(stopdate),
            "columns": request_columns,
        },
    )

    try:
        return pd.read_json(r.content)
    except ValueError:
        if current_tries == nb_tries:
            return pd.DataFrame(
                columns=[
                    "ra",
                    "dec",
                    "jd",
                    "nid",
                    "fid",
                    "dcmag",
                    "candid",
                    "not_updated",
                ]
            )
        else:
            if verbose:
                print("error when trying to get fink alerts, try again !")

            return request_fink(
                object_class,
                startdate,
                stopdate,
                request_columns,
                verbose,
                nb_tries,
                current_tries + 1,
            )


def get_last_sso_alert(object_class, date, verbose=False):
    """
    Get the alerts from Fink corresponding to the object_class for the given date.

    Parameters
    ----------
    object_class : string
        the class of the requested alerts
    date : string
        the requested date of the alerts, format is YYYY-MM-DD

    Returns
    -------
    pdf : pd.DataFrame
        the alerts from Fink with the following columns:
            ra, dec, jd, nid, fid, dcmag, dcmag_err, candid, not_updated
    """
    startdate = datetime.datetime.strptime(date, "%Y-%m-%d")
    stopdate = startdate + datetime.timedelta(days=1)

    if verbose:
        print(
            "Query fink broker to get sso alerts for the night between {} and {}".format(
                startdate.strftime("%Y-%m-%d"), stopdate.strftime("%Y-%m-%d")
            )
        )

    request_columns = "i:ra, i:dec, i:jd, i:nid, i:fid, i:candid, i:magpsf, i:sigmapsf, i:magnr, i:sigmagnr, i:magzpsci, i:isdiffpos"
    if object_class == "Solar System MPC":
        request_columns += ", i:ssnamenr"

    pdf = request_fink(
        object_class, startdate, stopdate, request_columns, verbose, 5, 0
    )

    required_columns = [
        "ra",
        "dec",
        "jd",
        "nid",
        "fid",
        "dcmag",
        "dcmag_err",
        "candid",
        "not_updated",
    ]
    translate_columns = {
        "i:ra": "ra",
        "i:dec": "dec",
        "i:jd": "jd",
        "i:nid": "nid",
        "i:fid": "fid",
        "i:candid": "candid",
    }
    if object_class == "Solar System MPC":
        required_columns.append("ssnamenr")
        translate_columns["i:ssnamenr"] = "ssnamenr"

    _dc_mag = np.array(
        pdf.apply(
            lambda x: dc_mag(
                x["i:fid"],
                x["i:magpsf"],
                x["i:sigmapsf"],
                x["i:magnr"],
                x["i:sigmagnr"],
                x["i:magzpsci"],
                x["i:isdiffpos"],
            ),
            axis=1,
            result_type="expand",
        ).values
    )

    pdf = pdf.rename(columns=translate_columns)
    if len(_dc_mag) > 0:
        pdf.insert(len(pdf.columns), "dcmag", _dc_mag[:, 0])
        pdf.insert(len(pdf.columns), "dcmag_err", _dc_mag[:, 1])
    else:
        return pd.DataFrame(columns=required_columns)

    pdf.insert(len(pdf.columns), "not_updated", np.ones(len(pdf), dtype=np.bool_))
    return pdf[required_columns]


def intro_reset():
    print("WARNING !!!")
    print(
        "you will loose the trajectory done by previous association, Continue ? [Y/n]"
    )


def yes_reset(arguments, tr_df_path, obs_df_path):
    if os.path.exists(tr_df_path) and os.path.exists(obs_df_path):
        print("Removing files :\n\t{}\n\t{}".format(tr_df_path, obs_df_path))
        try:
            os.remove(tr_df_path)
            os.remove(obs_df_path)
        except OSError as e:
            if arguments["--verbose"]:
                print("Failed with:", e.strerror)
                print("Error code:", e.code)
    else:
        print("File trajectory and old observations not exists.")

    dirname = os.path.dirname(tr_df_path)
    save_path = os.path.join(dirname, "save", "")
    if os.path.exists(save_path):
        shutil.rmtree(save_path)


def no_reset():
    print("Abort reset.")


def get_data(new_alerts, tr_df_path, obs_df_path, orb_res_path):
    last_nid = next_nid = new_alerts["nid"][0]
    trajectory_df = pd.DataFrame(columns=list(new_alerts.columns) + ["trajectory_id"])
    old_obs_df = pd.DataFrame(columns=new_alerts.columns)

    last_trajectory_id = 0

    # test if the trajectory_df and old_obs_df exists in the output directory.
    if os.path.exists(tr_df_path) and os.path.exists(obs_df_path):

        trajectory_df = pd.read_parquet(tr_df_path)
        old_obs_df = pd.read_parquet(obs_df_path)
        last_nid = np.max([np.max(trajectory_df["nid"]), np.max(old_obs_df["nid"])])
        if last_nid == next_nid:
            print()
            print("ERROR !!!")
            print("Association already done for this night.")
            print("Wait a next observation night to do new association")
            print("or run 'fink_fat solve_orbit' to get orbital_elements.")
            exit()
        if last_nid > next_nid:
            print()
            print("ERROR !!!")
            print(
                "Query alerts from a night before the last night in the recorded trajectory/old_observations."
            )
            print(
                "Maybe try with a more recent night or reset the associations with 'fink_fat association -r'"
            )
            exit()

        if os.path.exists(orb_res_path):
            orb_cand = pd.read_parquet(orb_res_path)
            last_trajectory_id = np.max(
                np.union1d(trajectory_df["trajectory_id"], orb_cand["trajectory_id"])
            )
        else:
            last_trajectory_id = np.max(trajectory_df["trajectory_id"])

    return trajectory_df, old_obs_df, last_nid, next_nid, last_trajectory_id
