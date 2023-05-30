import numpy as np
import pandas as pd

import filecmp
import sys
import os

from fink_utils.photometry.vect_conversion import vect_dc_mag
from fink_fat.orbit_fitting.mpcobs_files import write_observation_file


if __name__ == "__main__":

    mpc_file_obs_path = "fink_fat/test/test_orbit_file"

    mpc_obs = pd.read_parquet("fink_fat/test/mpc_example.parquet")

    mpc_obs["magpsf"], mpc_obs["sigmapsf"] = vect_dc_mag(
        mpc_obs["i:fid"],
        mpc_obs["i:magpsf"],
        mpc_obs["i:sigmapsf"],
        mpc_obs["i:magnr"],
        mpc_obs["i:sigmagnr"],
        mpc_obs["i:magzpsci"],
        mpc_obs["i:isdiffpos"],
    )
    ssnamenr = np.unique(mpc_obs["i:ssnamenr"])

    ssnamenr_to_trid = {
        sso_name: tr_id for tr_id, sso_name in zip(np.arange(len(ssnamenr)), ssnamenr)
    }

    mpc_obs["trajectory_id"] = mpc_obs.apply(
        lambda x: ssnamenr_to_trid[x["i:ssnamenr"]], axis=1
    )

    mpc_obs = mpc_obs.rename(
        {"i:ra": "ra", "i:dec": "dec", "i:fid": "fid", "i:jd": "jd"}, axis=1
    )

    os.mkdir("{}/mpcobs/".format(mpc_file_obs_path))

    res_test = 0
    for tr_id in np.unique(mpc_obs["trajectory_id"]):
        tmp_tr = mpc_obs[mpc_obs["trajectory_id"] == tr_id]

        prov_desig = write_observation_file("{}/".format(mpc_file_obs_path), tmp_tr)

        if not filecmp.cmp(
            "{}/mpcobs/{}.obs".format(mpc_file_obs_path, prov_desig),
            "{}/mpcobs_temoin/{}.obs".format(mpc_file_obs_path, prov_desig),
        ):  # pragma: no cover
            print("failed test: {}".format(prov_desig))
            res_test = 1

        os.remove("{}/mpcobs/{}.obs".format(mpc_file_obs_path, prov_desig))

    os.rmdir("{}/mpcobs/".format(mpc_file_obs_path))
    sys.exit(res_test)
