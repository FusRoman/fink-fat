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
