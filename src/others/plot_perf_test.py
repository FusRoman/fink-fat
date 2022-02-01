from itsdangerous import json
import seaborn as sns
import numpy as np
import pandas as pd
from src.others.utils import load_data
import json


if __name__ == "__main__":
    test_name = "perf_test_1"

    df_sso = load_data("Solar System MPC", 0)

    trajectory_df = pd.read_parquet("src/others/perf_test/{}.parquet".format(test_name))

    with open("src/others/perf_test/{}.json".format(test_name), "r") as json_file:
        stat = json.load(json_file)
    

    