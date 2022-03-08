import configparser
import os

import fink_fat


def string_to_bool(bool_str):
    if bool_str.casefold() == "false".casefold():
        return False
    else:
        return True


def init_cli(arguments):

    # read the config file
    config = configparser.ConfigParser()

    if arguments["--config"]:
        config.read(arguments["--config"])
    else:
        config_path = os.path.join(
            os.path.dirname(fink_fat.__file__), "data", "fink_fat.conf"
        )
        config.read(config_path)

    output_path = config["OUTPUT"]["association_output_file"]

    if arguments["--output"]:
        output_path = arguments["--output"]

    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    return config, output_path


def get_class(arguments, path):
    if arguments["mpc"]:
        path = os.path.join(path, "mpc", "")
        if not os.path.isdir(path):
            os.mkdir(path)
        object_class = "Solar System MPC"

    elif arguments["candidates"]:
        path = os.path.join(path, "candidates", "")
        if not os.path.isdir(path):
            os.mkdir(path)
        object_class = "Solar System candidate"

    return path, object_class


def yes_or_no(
    intro_function, yes_function, no_function, intro_args=(), yes_args=(), no_args=()
):

    intro_function(*intro_args)

    answer = ""
    while answer.upper() not in ["Y", "YES", "N", "NO"]:
        answer = input("Continue?")
        if answer.upper() in ["Y", "YES"]:
            yes_function(*yes_args)
        elif answer.upper() in ["N", "NO"]:
            no_function(*no_args)
        else:
            print("please, answer with y or n.")
