import datetime
import os
import numpy as np
import pandas as pd
import requests
from fink_science.conversion import dc_mag
import shutil


def get_last_sso_alert(object_class, date, verbose=False):
    startdate = datetime.datetime.strptime(date, "%Y-%m-%d")
    stopdate = startdate + datetime.timedelta(days=1)

    if verbose:
        print(
            "Query fink broker to get sso alerts for the night between {} and {}".format(
                startdate.strftime("%Y-%m-%d"), stopdate.strftime("%Y-%m-%d")
            )
        )

    r = requests.post(
        "https://fink-portal.org/api/v1/latests",
        json={
            "class": object_class,
            "n": "1000",
            "startdate": str(startdate),
            "stopdate": str(stopdate),
        },
    )
    pdf = pd.read_json(r.content)

    required_columns = [
        "ra",
        "dec",
        "jd",
        "nid",
        "fid",
        "dcmag",
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
    save_path = os.path.join(dirname, "save" , "")
    if os.path.exists(save_path):
        shutil.rmtree(save_path)


def no_reset():
    print("Abort reset.")


def get_data(new_alerts, tr_df_path, obs_df_path):
    last_nid = next_nid = new_alerts["nid"][0]
    trajectory_df = pd.DataFrame(columns=new_alerts.columns)
    old_obs_df = pd.DataFrame(columns=new_alerts.columns)

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

    return trajectory_df, old_obs_df, last_nid, next_nid
