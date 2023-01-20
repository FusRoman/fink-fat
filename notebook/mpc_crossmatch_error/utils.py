from astroquery.mpc import MPC
import pandas as pd


def mpc_crossmatch(mpc_orb, ssnamenr):
    explode_other = mpc_orb.explode("Other_desigs")
    # explode_other = explode_other[explode_other["Principal_desig"] != explode_other["Other_desigs"]].reset_index(drop=True)

    t1 = explode_other["Number"].str[1:-1].isin(ssnamenr)
    t2 = explode_other["Principal_desig"].str.replace(" ", "").isin(ssnamenr)
    t3 = explode_other["Name"].str.replace(" ", "").isin(ssnamenr)
    t4 = explode_other["Other_desigs"].str.replace(" ", "").isin(ssnamenr)

    reconstructed_mpc = explode_other[t1 | t2 | t3 | t4].drop_duplicates(
        ["Number", "Principal_desig"]
    )

    a = ~ssnamenr.isin(explode_other["Number"].str[1:-1])
    b = ~ssnamenr.isin(explode_other["Principal_desig"].str.replace(" ", ""))
    c = ~ssnamenr.isin(explode_other["Name"].str.replace(" ", ""))
    d = ~ssnamenr.isin(explode_other["Other_desigs"].str.replace(" ", ""))

    not_in_mpc = ssnamenr[a & b & c & d]

    return reconstructed_mpc, not_in_mpc


def load_data(columns=None):
    """
    Load all the observations of solar system object in Fink
    from the period between 1st of November 2019 and 16th of January 2022

    Parameters
    ----------
    None

    Return
    ------
    sso_data : Pandas Dataframe
        all sso alerts with the following columns
            - 'objectId', 'candid', 'ra', 'dec', 'jd', 'nid', 'fid', 'ssnamenr',
                'ssdistnr', 'magpsf', 'sigmapsf', 'magnr', 'sigmagnr', 'magzpsci',
                'isdiffpos', 'day', 'nb_detection', 'year', 'month'
    """
    return pd.read_parquet("../parameters_selection/sso_data", columns=columns)


def queryMPC(number, kind="asteroid"):
    """Query MPC for information about object 'designation'.
    Parameters
    ----------
    designation: str
        A name for the object that the MPC will understand.
        This can be a number, proper name, or the packed designation.
    kind: str
        asteroid or comet
    Returns
    -------
    pd.Series
        Series containing orbit and select physical information.
    """
    try:
        mpc = MPC.query_object(target_type=kind, number=number)
        mpc = mpc[0]
    except IndexError:
        try:
            mpc = MPC.query_object(target_type=kind, designation=number)
            mpc = mpc[0]
        except IndexError:
            return pd.Series({})
    except IndexError:
        return pd.Series({})
    except RuntimeError:
        return pd.Series({})
    orbit = pd.Series(mpc)
    return orbit
