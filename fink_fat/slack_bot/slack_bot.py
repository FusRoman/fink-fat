from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import os

from fink_fat.others.utils import init_logging


def init_slackbot() -> WebClient:
    try:
        token_slack = os.environ["ANOMALY_SLACK_TOKEN"]
    except KeyError:
        logger = init_logging()
        logger.error("ANOMALY_SLACK_TOKEN environement variable not found !!", exc_info=1)
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


if __name__ == "__main__":
    slack_client = init_slackbot()

    post_msg_on_slack(slack_client, "bonjour")
