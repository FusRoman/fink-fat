import pandas as pd
import numpy as np


def orbit_score(orbits: pd.DataFrame) -> pd.Series:
    """
    Compute a score using the rms

    Parameters
    ----------
    orbits : pd.DataFrame
        orbital parameters with their rms

    Returns
    -------
    pd.Series
        the score for each orbit
    """
    rms_cols = orbits.columns[orbits.columns.str.contains("rms")]
    rms = orbits[rms_cols]
    return np.sqrt((rms**2).sum(axis=1))


def best_orbits(orbits: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the score and return the 10 best candidates.

    Parameters
    ----------
    orbits : pd.DataFrame
        orbital parameters with their rms

    Returns
    -------
    pd.DataFrame
        10 best orbits based on the score
    """
    score = orbit_score(orbits)
    orbits["score"] = score
    orbits = orbits.sort_values("score")
    return orbits.iloc[:10]


def asteroid_class(a: float, e: float) -> str:
    """
    Return the dynamical class of the asteroids based on the semi-major axis and the eccentricity.

    Parameters
    ----------
    a : float
        semi-major axis
    e : float
        eccentricity

    Returns
    -------
    str
        dynamical class
    """
    if 0.08 <= a <= 0.21:
        return "Vulcanoid"

    if 0.21 <= a <= 1.0 and a * (1 + e) < 0.983:
        return "NEA>Atira"

    if 0.21 <= a <= 1.0 and a * (1 + e) >= 0.983:
        return "NEA>Aten"

    cond_nea = 1.0 < a < 2.0
    peri = a * (1 - e)

    if cond_nea and peri < 1.017:
        return "NEA>Apollo"

    if cond_nea and 1.017 <= peri < 1.3:
        return "NEA>Amor"

    if cond_nea and 1.3 <= peri <= 1.58:
        return "Mars-Crosser>Deep"

    if cond_nea and 1.58 <= peri <= 1.666:
        return "Mars-Crosser>Shallow"

    if cond_nea and peri > 1.666:
        return "Hungaria"

    if 2.0 <= a < 2.5:
        return "MB>Inner"

    if 2.5 <= a < 2.82:
        return "MB>Middle"

    if 2.82 <= a < 3.27:
        return "MB>Outer"

    if 3.27 <= a < 3.7:
        return "MB>Cybele"

    if 3.7 <= a < 4.6:
        return "MB>Hilda"

    if 4.6 <= a < 5.5:
        return "Trojan"

    if 5.5 <= a < 30.1:
        return "Centaur"

    cond_kbo = 30.1 <= a < 2000
    if cond_kbo and e >= 0.24:
        return "KBO>Detached"

    if cond_kbo and peri <= 30.1 * (2 ** (2 / 3)) * (1 - 0.24):
        return "KBO>SDO"

    if 30.1 <= a < 39.4:
        return "KBO>Classical>Inner"

    if 39.4 <= a < 47.8:
        return "KBO>Classical>Main"

    if 47.8 <= a < 2000:
        return "KBO>Classical>Outer"

    if cond_kbo:
        return "KBO>Classical"

    if a >= 2000:
        return "Inner Oort cloud"


def orb_class(orbits: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the dynamical class of all the orbits

    Parameters
    ----------
    orbits : pd.DataFrame
        orbital parameters with their rms

    Returns
    -------
    pd.DataFrame
        dynamical class of all the orbits
    """
    if len(orbits) > 0:
        orbits["class"] = orbits.apply(lambda x: asteroid_class(x["a"], x["e"]), axis=1)
        return orbits
    else:
        return orbits
