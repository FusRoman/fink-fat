import os
import shutil


def offline_intro_reset():
    print("WARNING !!!")
    print(
        "you will loose all the data from the previous associations including the orbits, Continue ? [Y/n]"
    )


def offline_yes_reset(arguments, tr_df_path, obs_df_path, orb_res_path, traj_orb_path):
    # fmt: off
    test = os.path.exists(tr_df_path) and os.path.exists(obs_df_path) and os.path.exists(orb_res_path) and os.path.exists(traj_orb_path)
    # fmt: on
    if test:
        print(
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
                print("Failed with:", e.strerror)
                print("Error code:", e.code)
    else:
        print("Data from previous associations and solve orbits not exists.")

    dirname = os.path.dirname(tr_df_path)
    save_path = os.path.join(dirname, "save", "")
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
