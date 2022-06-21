import numpy as np
import pandas as pd

import filecmp
import sys
import os

from fink_utils.photometry.vect_conversion import vect_dc_mag
from fink_fat.orbit_fitting.mpcobs_files import write_observation_file


if __name__ == "__main__":

    mpc_obs = pd.read_parquet("fink_fat/test/mpc_example.parquet")

    mpc_obs["dcmag"], mpc_obs["dc_magerr"] = vect_dc_mag(
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

    os.mkdir("fink_fat/test/test_obs/mpcobs/")

    res_test = 0
    for tr_id in np.unique(mpc_obs["trajectory_id"]):
        tmp_tr = mpc_obs[mpc_obs["trajectory_id"] == tr_id]

        prov_desig = write_observation_file("fink_fat/test/test_obs/", tmp_tr)

        if not filecmp.cmp(
            "fink_fat/test/test_obs/mpcobs/{}.obs".format(prov_desig),
            "fink_fat/test/test_obs/mpcobs_temoin/{}.obs".format(prov_desig),
        ):  # pragma: no cover
            print("failed test: {}".format(prov_desig))
            res_test = 1

        os.remove("fink_fat/test/test_obs/mpcobs/{}.obs".format(prov_desig))

    os.rmdir("fink_fat/test/test_obs/mpcobs/")
    sys.exit(res_test)
