import numpy as np
import fink_fat
import logging
import types
import datetime
import pytz


def create_ranges(starts, ends, chunks_list):
    clens = chunks_list.cumsum()
    ids = np.ones(clens[-1], dtype=int)
    ids[0] = starts[0]
    ids[clens[:-1]] = starts[1:] - ends[:-1] + 1
    out = ids.cumsum()
    return out


def repeat_chunk(a, chunks, repeats):
    s = np.r_[0, chunks.cumsum()]
    starts = a[np.repeat(s[:-1], repeats)]
    repeated_chunks = np.repeat(chunks, repeats)
    ends = starts + repeated_chunks
    out = create_ranges(starts, ends, repeated_chunks)
    return out


def cast_obs_data(trajectories):
    dict_new_types = {
        "ra": np.float64,
        "dec": np.float64,
        "jd": np.float64,
        "fid": np.int8,
        "nid": np.int32,
        "candid": np.int64,
    }
    if "trajectory_id" in trajectories:
        dict_new_types["trajectory_id"] = np.int64
    if "not_updated" in trajectories:
        dict_new_types["not_updated"] = np.bool_
    if "ssnamenr" in trajectories:
        dict_new_types["ssnamenr"] = str
    tr_orb_columns = [
        "magpsf",
        "sigmapsf",
        "ref_epoch",
        "a",
        "e",
        "i",
        "long. node",
        "arg. peric",
        "mean anomaly",
        "rms_a",
        "rms_e",
        "rms_i",
        "rms_long. node",
        "rms_arg. peric",
        "rms_mean anomaly",
    ]

    for c in tr_orb_columns:
        if c in trajectories:
            dict_new_types[c] = np.float64

    return trajectories.astype(dict_new_types)


class CustomTZFormatter(logging.Formatter):  # pragma: no cover
    """override logging.Formatter to use an aware datetime object"""

    def converter(self, timestamp):
        dt = datetime.datetime.fromtimestamp(timestamp)
        tzinfo = pytz.timezone("Europe/Paris")
        return tzinfo.localize(dt)

    def formatTime(self, record, datefmt=None):
        dt = self.converter(record.created)
        if datefmt:
            s = dt.strftime(datefmt)
        else:
            try:
                s = dt.isoformat(timespec="milliseconds")
            except TypeError:
                s = dt.isoformat()
        return s


def init_logging(logger_name=fink_fat.__name__) -> logging.Logger:
    """
    Initialise a logger for the gcn stream

    Parameters
    ----------
    None

    Returns
    -------
    logger : Logger object
        A logger object for the logging management.

    Examples
    --------
    >>> l = init_logging()
    >>> type(l)
    <class 'logging.Logger'>
    """
    # create logger

    def log_newline(self, how_many_lines=1):
        # Switch handler, output a blank line
        self.removeHandler(self.console_handler)
        self.addHandler(self.blank_handler)
        for i in range(how_many_lines):
            self.info("\n")

        # Switch back
        self.removeHandler(self.blank_handler)
        self.addHandler(self.console_handler)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # create formatter
    formatter = CustomTZFormatter(
        "%(asctime)s - %(name)s - %(levelname)s \n\t message: %(message)s"
    )

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)

    blank_handler = logging.StreamHandler()
    blank_handler.setLevel(logging.DEBUG)
    blank_handler.setFormatter(logging.Formatter(fmt=""))

    logger.console_handler = ch
    logger.blank_handler = blank_handler
    logger.newline = types.MethodType(log_newline, logger)

    return logger
