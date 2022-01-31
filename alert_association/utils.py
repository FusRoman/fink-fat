import numpy as np
import glob
import os
import pandas as pd


def load_data(object_class, nb_indirection=1):
    all_df = []

    parent_folder = "".join(np.repeat(np.array(["../"]), nb_indirection))

    all_path = sorted(glob.glob(os.path.join(parent_folder, "data", "month=*")))[:-1]

    # load all data
    for path in all_path:
        df_sso = pd.read_pickle(path)
        all_df.append(df_sso)

    df_sso = pd.concat(all_df).sort_values(["jd"]).drop_duplicates()

    df_sso = df_sso.drop_duplicates(["candid"])
    df_sso = df_sso[df_sso["fink_class"] == object_class]

    return df_sso


def get_mpc_database(nb_indirection=0):

    parent_folder = "".join(np.repeat(np.array(["../"]), nb_indirection))

    mpc_database = pd.read_json(
        os.path.join(parent_folder, "data", "mpc_database", "mpcorb_extended.json")
    )
    mpc_database["Number"] = mpc_database["Number"].astype("string").str[1:-1]
    return mpc_database
