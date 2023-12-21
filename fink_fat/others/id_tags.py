import numpy as np

from astropy.time import Time


def alphabetic_tag(int_id: int, nb_alphabetic: int) -> str:
    """
    Make an alphabetic tag in base 26.

    Parameters
    ----------
    int_id : int
        tag identifier
    nb_alphabetic : int
        number of symbol in the tag

    Returns
    -------
    str
        the tag

    Examples
    --------
    >>> alphabetic_tag(0, 0)
    ''
    >>> alphabetic_tag(0, 3)
    'aaa'
    >>> alphabetic_tag(3, 5)
    'aaaad'
    >>> alphabetic_tag(100, 5)
    'aaadw'
    >>> alphabetic_tag(675, 2)
    'zz'
    >>> alphabetic_tag(676, 2)
    'aa'
    """
    res_tag = ""
    for _ in range(nb_alphabetic):
        q = int(int_id / 26)
        r = int(int_id % 26)

        res_tag += chr(r + 97)
        int_id = q

    return res_tag[::-1]


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
    'FF20230710aaaaaap'
    >>> int_to_tags(27, 2460135.98)
    'FF20230710aaaaabb'
    >>> int_to_tags(652, 2460135.98)
    'FF20230710aaaaazc'
    """
    res_tag = alphabetic_tag(traj_id, 7)
    discovery = Time(jd, format="jd").datetime
    return "FF{:04d}{:02d}{:02d}{}".format(
        discovery.year, discovery.month, discovery.day, res_tag
    )


def generate_tags(begin: int, end: int, jd: float) -> list:
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
    ['FF20230709aaaaaad', 'FF20230712aaaaaae', 'FF20230723aaaaaaf']
    """
    return [int_to_tags(i, date) for date, i in zip(jd, np.arange(begin, end))]


if __name__ == "__main__":  # pragma: no cover
    import sys
    import doctest

    sys.exit(doctest.testmod()[0])
