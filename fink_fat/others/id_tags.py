import numpy as np

from astropy.time import Time


def int_to_tags(traj_id, jd):
    """
    Convert an integer into a Fink-FAT trajectory identifier.
    The identifier format is :

    "FFYYYYXXXXXXX" where
        'FF' is for 'Fink-FAT',
        'DDMMYYYY' is the full date (01012022),
        and 'XXXXXXX' is the number of trajectories since the beginning of the year in the base-26.

    Parameters
    ----------
    traj_id : integer
        trajectories identifier as an integer
    jd : float
        the discovery date of the trajectory

    Returns
    -------
    tags : str
        the fink-fat identifier

    Examples
    --------
    >>> int_to_tags(15, 2460135.98)
    'FF10072023aaaaaap'
    >>> int_to_tags(27, 2460135.98)
    'FF10072023aaaaabb'
    >>> int_to_tags(652, 2460135.98)
    'FF10072023aaaaazc'
    """
    res_tag = ""
    for _ in range(7):
        q = int(traj_id / 26)
        r = int(traj_id % 26)

        res_tag += chr(r + 97)
        traj_id = q

    discovery = Time(jd, format="jd").datetime
    return "FF{:02d}{:02d}{}{}".format(
        discovery.year, discovery.month, discovery.day, res_tag[::-1]
    )


def generate_tags(begin, end, jd):
    """
    Generate a list of tags between begin and end


    Parameters
    ----------
    begin : integer
        start tags
    end : integer
        end tags
    jd : float list
        list of discovery date

    Returns
    -------
    tags_list : str list
        all the tags between begin and end

    Examples
    --------
    >>> generate_tags(3, 6, [2460135.42, 2460137.57, 2460148.72])
    ['FF09072023aaaaaad', 'FF12072023aaaaaae', 'FF23072023aaaaaaf']
    """
    return [int_to_tags(i, date) for date, i in zip(jd, np.arange(begin, end))]


if __name__ == "__main__":  # pragma: no cover
    import sys
    import doctest

    sys.exit(doctest.testmod()[0])
