from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import io

from fink_fat.others.utils import init_logging
from fink_fat.orbit_fitting.utils import best_orbits


def init_slackbot() -> WebClient:
    try:
        token_slack = os.environ["ANOMALY_SLACK_TOKEN"]
    except KeyError:
        logger = init_logging()
        logger.error(
            "ANOMALY_SLACK_TOKEN environement variable not found !!", exc_info=1
        )
    client = WebClient(token=token_slack)
    return client


def post_msg_on_slack(webclient: WebClient, msg: list):
    logging = init_logging()
    try:
        for tmp_msg in msg:
            webclient.chat_postMessage(
                channel="#bot_asteroid_test",
                text=tmp_msg,
                blocks=[
                    {"type": "section", "text": {"type": "mrkdwn", "text": tmp_msg}}
                ],
            )
            logging.info("Post msg on slack successfull")
            time.sleep(1)
    except SlackApiError as e:
        if e.response["ok"] is False:
            logging.error("Post slack msg error", exc_info=1)


def post_assoc_on_slack(
    last_night: str,
    statistic_string: str,
    trajectory_df: pd.DataFrame,
    trajectory_orb: pd.DataFrame,
    orbits: pd.DataFrame,
    old_orbits: pd.DataFrame,
    ssoCandId: list,
):
    def size_hist(data, title):
        plt.figure()
        plt.title(title)
        plt.xlabel("number of observations")
        plt.hist(data)
        bytes_fig = io.BytesIO()
        plt.savefig(bytes_fig, format="png")
        bytes_fig.seek(0)
        return bytes_fig

    slack_client = init_slackbot()

    result = slack_client.files_upload_v2(
        file_uploads=[
            {
                "file": size_hist(
                    trajectory_df["trajectory_id"].value_counts(),
                    "Distribution of the polyfit trajectories number of observations",
                ),
                "title": "traj_size_distrib",
            },
            {
                "file": size_hist(
                    trajectory_orb["ssoCandId"].value_counts(),
                    "Distribution of the orbit trajectories number of observations",
                ),
                "title": "traj_size_distrib",
            },
        ]
    )
    time.sleep(3)

    traj_size_perml = f"<{result['files'][0]['permalink']}|{' '}>"
    orb_size_perml = f"<{result['files'][1]['permalink']}|{' '}>"

    best_orb_msg = []
    if len(ssoCandId) == 0:
        best_orb_msg.append("No updated or new orbits during this night")
    else:
        best_orb = best_orbits(orbits)
        tmp_msg = "### 10 best new or updated orbits of the night ###\n"
        best_orb_msg.append(tmp_msg)
        for i, (_, rows) in enumerate(best_orb.iterrows()):
            tmp_best = "------------------------------------------------------\n"

            prev_orb = old_orbits[
                old_orbits["ssoCandId"] == rows["ssoCandId"]
            ].sort_values(["ref_epoch"])
            if len(prev_orb) > 0:
                tmp_best += f"{i+1}. Orbit updated: {rows['ssoCandId']}\n"
                tmp_best += f"\tclass: {rows['class']} (previous class: {prev_orb.iloc[-1]['class']})\n"
            else:
                tmp_best += f"{i+1}. New orbit: {rows['ssoCandId']}\n"
                tmp_best += f"\tclass: {rows['class']}\n"

            traj = trajectory_orb[
                trajectory_orb["ssoCandId"] == rows["ssoCandId"]
            ].sort_values("jd")
            obs_window = traj["jd"].values[-1] - traj["jd"].values[0]
            last_magpsf = traj["magpsf"].values[-1]
            last_sigmapsf = traj["sigmapsf"].values[-1]

            tmp_best += f"\tnumber of observations in the trajectory: {len(traj)}\n"
            tmp_best += f"\ttrajectory time window: {obs_window:.4f} days\n"
            tmp_best += f"\tlast magpsf: {last_magpsf} ± {last_sigmapsf}\n"
            tmp_best += f"\t\ta = {rows['a']} ± {rows['rms_a']}\n"
            tmp_best += f"\t\te = {rows['e']} ± {rows['rms_e']}\n"
            tmp_best += f"\t\ti = {rows['i']} ± {rows['rms_i']}\n\n"
            best_orb_msg.append(tmp_best)

    msg_list = []
    slack_msg = f"""
============================================
FINK-FAT STATISTICS OF THE NIGHT: {last_night}

{statistic_string}

{traj_size_perml}

{orb_size_perml}
"""
    msg_list += [slack_msg] + best_orb_msg
    post_msg_on_slack(slack_client, msg_list)
