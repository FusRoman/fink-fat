import numpy as np
import glob
import os
import pandas as pd
from fink_fat import __file__


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


def create_ranges(starts, ends, chunks_list):
    clens = chunks_list.cumsum()
    ids = np.ones(clens[-1], dtype=int)
    ids[0] = starts[0]
    ids[clens[:-1]] = starts[1:] - ends[:-1] + 1
    out = ids.cumsum()
    return out


def repeat_chunk(a, chunks, repeats):
    s = np.r_[0, chunks.cumsum()]
    starts = a[np.repeat(s[:-1], repeats)]
    repeated_chunks = np.repeat(chunks, repeats)
    ends = starts + repeated_chunks
    out = create_ranges(starts, ends, repeated_chunks)
    return out


def cast_obs_data(trajectories):
    dict_new_types = {
        "ra": np.float64,
        "dec": np.float64,
        "jd": np.float64,
        "fid": np.int8,
        "nid": np.int32,
        "candid": np.int64,
    }
    if "trajectory_id" in trajectories:
        dict_new_types["trajectory_id"] = np.int64
    if "not_updated" in trajectories:
        dict_new_types["not_updated"] = np.bool_
    tr_orb_columns = [
        "dcmag",
        "dcmagerr",
        "ref_epoch",
        "a",
        "e",
        "i",
        "long. node",
        "arg. peric",
        "mean anomaly",
        "rms_a",
        "rms_e",
        "rms_i",
        "rms_long. node",
        "rms_arg. peric",
        "rms_mean anomaly",
    ]

    for c in tr_orb_columns:
        if c in trajectories:
            dict_new_types[c] = np.float64

    return trajectories.astype(dict_new_types)


if __name__ == "__main__":
    path = os.path.dirname(__file__)

    data_path = 'file://{}/data/month=*'.format(path)

    print(data_path)