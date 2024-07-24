"""
Usage:
    fink_fat associations (mpc | candidates | fitroid) [--night <date>] [--filepath <filepath>] [options]
    fink_fat solve_orbit (mpc | candidates) (local | cluster) [options]
    fink_fat merge_orbit (mpc | candidates) [options]
    fink_fat offline (mpc (local | cluster) | candidates (local | cluster) | fitroid) <end> [<start>] [options]
    fink_fat stats (mpc | candidates) [--mpc-data <path>] [options]
    fink_fat -h | --help
    fink_fat --version

Options:
  associations                        Perform associations of alert to return a set of trajectories candidates.
  solve_orbit                         Resolve a dynamical inverse problem to return a set of orbital elements from
                                      the set of trajectories candidates.
  merge_orbit                         Merge the orbit candidates if the both trajectories can belong to the same solar system objects.
  offline                             Associate the alerts to form trajectories candidates then solve the orbit
                                      until the end parameters. Starts from saved data or from the start parameters
                                      if provided.
  stats                               Print statistics about trajectories detected by assocations, the old observations
                                      and, if exists, the orbital elements for some trajectories.
  mpc                                 Return the associations on the solar system mpc alerts (only for tests purpose).
  candidates                          Run the associations on the solar system candidates alerts.
  fitroid                             Merge the trajectories on the alerts associated with the trajectories by the roid science module of fink
                                      (the roid science module from fink-science must have been run before)
  local                               Run the orbital solver in local mode. Use multiprocessing to speed-up the computation.
  cluster                             Run the orbital solver in cluster mode. Use a Spark cluster to significantly speed-up the computation.
                                      The cluster mode need to be launch on a system where pyspark are installed and a spark cluster manager are setup.
  -n <date> --night <date>            Specify the night to request sso alerts from fink broker.
                                      Format is yyyy-mm-dd as yyyy = year, mm = month, dd = day.
                                      Example : 2022-03-04 for the 2022 march 04.
                                      [intervall of day between the day starting at night midday until night midday + 1]
  -f <filepath> --filepath <filepath> Path to the Euclid SSOPipe output file.
  -m <path> --mpc-data <path>         Compute statistics according to the minor planet center database.
                                      <path> of the mpc database file.
                                      The mpc database can be downloaded by pasting this url in your browser: https://minorplanetcenter.net/Extended_Files/mpcorb_extended.json.gz
  -r --reset                          Remove the file containing the trajectories candidates, the old observations and the orbits.
  -s --save                           Save the alerts sent by Fink before the associations for statistics purposes.
                                      Save also additional statistics : computation time, number of alerts from the current days, number of candidates trajectories, number of old observations.
  -h --help                           Show help and quit.
  --version                           Show version.
  --config FILE                       Specify the config file
  --verbose                           Print information and progress bar during the process
"""

from fink_fat import __version__
from docopt import docopt
from fink_fat.command_line.utils_cli import init_cli


def fink_fat_main(arguments):
    """
    Main function of fink_fat. Execute a process according to the arguments given by the user.

    Parameters
    ----------
    arguments : dictionnary
        arguments parse by docopt from the command line

    Returns
    -------
    None

    Examples
    --------

    """

    config, output_path = init_cli(arguments)

    if arguments["associations"]:
        from fink_fat.command_line.cli_main.associations import cli_associations

        cli_associations(arguments, config, output_path)

    # DEPRECATED
    # elif arguments["kalman"]:
    #     from fink_fat.command_line.cli_main.kalman import cli_kalman_associations

    #     cli_kalman_associations(arguments, config, output_path)

    elif arguments["solve_orbit"]:
        from fink_fat.command_line.cli_main.solve_orbit import cli_solve_orbit

        cli_solve_orbit(arguments, config, output_path)

    elif arguments["merge_orbit"]:
        from fink_fat.command_line.cli_main.merge_orbit import cli_merge_orbit

        cli_merge_orbit(arguments, config, output_path)

    elif arguments["stats"]:
        from fink_fat.command_line.cli_main.stats import cli_stats

        cli_stats(arguments, config, output_path)

    elif arguments["offline"]:
        from fink_fat.command_line.cli_main.offline import cli_offline

        cli_offline(arguments, config, output_path)

    else:
        exit()


def main():
    # parse the command line and return options provided by the user.
    arguments = docopt(__doc__, version=__version__)
    fink_fat_main(arguments)


def main_test(argv):
    # parse the command line and return options provided by the user.
    arguments = docopt(__doc__, argv=argv, version=__version__)

    fink_fat_main(arguments)
