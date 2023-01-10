import datetime


def int_to_tags(traj_id):
    """
    Convert an integer into a Fink-FAT trajectory identifier.
    The identifier format is :

    "FFYYYYXXXXXXX" where
        'FF' is for 'Fink-FAT',
        'YYYY' is the full year (2022),
        and 'XXXXXXX' is the number of trajectories since the beginning of the year in the base-26.

    Parameters
    ----------
    traj_id : integer
        trajectories identifier as an integer

    Returns
    -------
    tags : str
        the fink-fat identifier

    Examples
    --------
    >>> r = int_to_tags(15)
    >>> test_value = 'FF{}aaaaaap'.format(datetime.date.today().year)
    >>> test_value == r
    True

    >>> r =int_to_tags(27)
    >>> test_value = 'FF{}aaaaabb'.format(datetime.date.today().year)
    >>> test_value == r
    True

    >>> r = int_to_tags(652)
    >>> test_value = 'FF{}aaaaazc'.format(datetime.date.today().year)
    >>> test_value == r
    True
    """
    res_tag = ""
    for _ in range(7):
        q = int(traj_id / 26)
        r = int(traj_id % 26)

        res_tag += chr(r + 97)
        traj_id = q

    return "FF" + str(datetime.date.today().year) + res_tag[::-1]


def generate_tags(begin, end):
    """
    Generate a list of tags between begin and end


    Parameters
    ----------
    begin : integer
        start tags
    end : integer
        end tags

    Returns
    -------
    tags_list : str list
        all the tags between begin and end

    Examples
    --------
    >>> l = generate_tags(3, 6)
    >>> year = datetime.date.today().year
    >>> test_value = ['FF{}aaaaaad'.format(year), 'FF{}aaaaaae'.format(year), 'FF{}aaaaaaf'.format(year)]
    >>> test_value == l
    True
    """
    return [int_to_tags(i) for i in range(begin, end)]


if __name__ == "__main__":  # pragma: no cover
    import sys
    import doctest

    sys.exit(doctest.testmod()[0])