import numpy as np


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
    if "ssnamenr" in trajectories:
        dict_new_types["ssnamenr"] = str
    tr_orb_columns = [
        "magpsf",
        "sigmapsf",
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
