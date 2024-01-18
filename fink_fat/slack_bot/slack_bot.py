from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import io

from fink_fat.others.utils import init_logging


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


def post_msg_on_slack(webclient: WebClient, msg: str):
    logging = init_logging()
    try:
        webclient.chat_postMessage(
            channel="#bot_asteroid_test",
            text=msg,
            blocks=[{"type": "section", "text": {"type": "mrkdwn", "text": msg}}],
        )
        logging.info("Post msg on slack successfull")
    except SlackApiError as e:
        if e.response["ok"] is False:
            logging.error("Post slack msg error", exc_info=1)


def post_assoc_on_slack(
    last_night: str,
    statistic_string: str,
    trajectory_df: pd.DataFrame,
    trajectory_orb: pd.DataFrame,
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

    slack_msg = f"""
FINK-FAT STATISTICS OF THE NIGHT: {last_night}

{statistic_string}

{traj_size_perml}

{orb_size_perml}
"""

    post_msg_on_slack(slack_client, slack_msg)
