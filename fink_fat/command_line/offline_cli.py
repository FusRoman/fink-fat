import os
import shutil
from fink_fat.others.utils import init_logging


def offline_intro_reset():  # pragma: no cover
    logger = init_logging()
    logger.info("WARNING !!!")
    logger.info(
        "you will loose all the data from the previous associations including the orbits, Continue ? [Y/n]"
    )


def offline_yes_reset(
    arguments, tr_df_path, obs_df_path, orb_res_path, traj_orb_path
):  # pragma: no cover
    logger = init_logging()
    # fmt: off
    test = os.path.exists(tr_df_path) and os.path.exists(obs_df_path) and os.path.exists(orb_res_path) and os.path.exists(traj_orb_path)
    # fmt: on
    if test:
        logger.info(
            "Removing files :\n\t{}\n\t{}\n\t{}\n\t{}".format(
                tr_df_path, obs_df_path, orb_res_path, traj_orb_path
            )
        )
        try:
            os.remove(tr_df_path)
            os.remove(obs_df_path)
            os.remove(orb_res_path)
            os.remove(traj_orb_path)
        except OSError as e:
            if arguments["--verbose"]:
                logger.info("Failed with:", e.strerror)
                logger.info("Error code:", e.code)
    else:
        logger.info("Data from previous associations and solve orbits not exists.")

    dirname = os.path.dirname(tr_df_path)
    save_path = os.path.join(dirname, "save", "")
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
