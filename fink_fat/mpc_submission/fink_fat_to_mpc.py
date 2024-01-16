import pandas as pd
import numpy as np
from typing import Tuple
from lxml import etree as XMLTree
from lxml.etree import Element as XMLElement
from lxml.etree import _Element

from astropy.time import Time
import traceback
import requests

from fink_fat.others.id_tags import alphabetic_tag


def prep_traj_for_mpc(
    trajectory_orb: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Filter and change the trajectories contains in trajectory_orb to fit the mpc submission requirements.

    Parameters
    ----------
    trajectory_orb : pd.DataFrame
        contains observations of the trajectories with orbits return by Fink-FAT

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        trajectories with orbit ready to be send to the Minor Planet Center
        * Trajectories with multiple track during multiple night
        * Trajectories with only two points within the same night
        * Trajectories with more then two points within the same night

    Examples
    --------

    >>> test = pd.DataFrame(columns=["ssoCandId", "nid", "candid"])
    >>> prep_traj_for_mpc(test)
    (Empty DataFrame
    Columns: [ssoCandId, nid, candid]
    Index: [], Empty DataFrame
    Columns: [ssoCandId, nid, candid]
    Index: [], Empty DataFrame
    Columns: [ssoCandId, nid, candid]
    Index: [])

    >>> test = pd.DataFrame({
    ... "ssoCandId": ["a", "a"],
    ... "nid": [1, 1],
    ... "candid": [0, 1]
    ... })
    >>> prep_traj_for_mpc(test)
    (Empty DataFrame
    Columns: [ssoCandId, nid, candid]
    Index: [],   ssoCandId  nid  candid
    0         a    1       0
    1         a    1       1, Empty DataFrame
    Columns: [ssoCandId, nid, candid]
    Index: [])

    >>> test = pd.DataFrame({
    ... "ssoCandId": ["a", "a", "a", "a"],
    ... "nid": [1, 1, 2, 2],
    ... "candid": [0, 1, 2, 3]
    ... })
    >>> prep_traj_for_mpc(test)
    (  ssoCandId  nid  candid
    0         a    1       0
    1         a    1       1
    2         a    2       2
    3         a    2       3, Empty DataFrame
    Columns: [ssoCandId, nid, candid]
    Index: [], Empty DataFrame
    Columns: [ssoCandId, nid, candid]
    Index: [])

    >>> test = pd.DataFrame({
    ... "ssoCandId": ["a", "a", "a", "a", "b", "b", "b", "c", "c"],
    ... "nid": [1, 1, 2, 2, 1, 1, 1, 3, 3],
    ... "candid": [0, 1, 2, 3, 4, 5, 6, 7, 8]
    ... })
    >>> prep_traj_for_mpc(test)
    (  ssoCandId  nid  candid
    0         a    1       0
    1         a    1       1
    2         a    2       2
    3         a    2       3,   ssoCandId  nid  candid
    7         c    3       7
    8         c    3       8,   ssoCandId  nid  candid
    4         b    1       4
    5         b    1       5
    6         b    1       6)
    """
    gb_obs = (
        trajectory_orb[["ssoCandId", "nid", "candid"]]
        .groupby(["ssoCandId", "nid"])
        .agg(nb_obs=("candid", lambda x: len(np.unique(x))), candid=("candid", list))
    )
    tracklets = gb_obs[gb_obs["nb_obs"] > 1].reset_index()
    nb_tracklets = tracklets["ssoCandId"].value_counts()

    multiple_track = nb_tracklets[nb_tracklets > 1].index
    one_track = nb_tracklets[nb_tracklets == 1].index

    multiple_track = tracklets[tracklets["ssoCandId"].isin(multiple_track)]

    one_track = tracklets[tracklets["ssoCandId"].isin(one_track)]
    one_track_two_point = one_track[one_track["nb_obs"] == 2]
    one_track_more_point = one_track[one_track["nb_obs"] > 2]

    multiple_night_track = trajectory_orb[
        trajectory_orb["candid"].isin(multiple_track["candid"].explode())
    ]

    one_night_two_point = trajectory_orb[
        trajectory_orb["candid"].isin(one_track_two_point["candid"].explode())
    ]
    one_night_more_point = trajectory_orb[
        trajectory_orb["candid"].isin(one_track_more_point["candid"].explode())
    ]

    return multiple_night_track, one_night_two_point, one_night_more_point


def createADESTree() -> _Element:
    """createADESTree creates and returns
    a top-level ADES tree.
    """
    property = {}
    property["version"] = "2022"
    ades = XMLElement("ades", property)
    return ades


def addDataElement(branch: _Element, tag: str, text: str) -> _Element:
    """addDataElement(branch, tag, text)
    makes a new element with tag and text
    extends branch with the new element (which now owns memory)
    and returns the new element for later extension itself
    """
    el = XMLElement(tag)
    el.text = text
    branch.extend([el])
    return el


def addElement(branch, tag):
    """addElement(branch, tag)
    makes a new element with tag
    extends branch with the new element (which now owns memory)
    and returns the new element for later extension itself
    """
    el = XMLElement(tag)
    branch.extend([el])
    return el


def make_obs_context(obs_context: _Element):
    """
    Create the context block containing the sender informations

    Parameters
    ----------
    obs_context : _Element
        the root block of the ades xml
    """
    observatory = addElement(obs_context, "observatory")
    addDataElement(observatory, "mpcCode", "I41")
    addDataElement(observatory, "name", "ZTF")

    submitter = addElement(obs_context, "submitter")
    addDataElement(submitter, "name", "J. Peloton")
    addDataElement(
        submitter,
        "institution",
        "Laboratoire de Physique des 2 infinis Irène Joliot-Curie, Orsay, FRANCE",
    )

    observers = addElement(obs_context, "observers")
    addDataElement(observers, "name", "J. Peloton")
    addDataElement(observers, "name", "R. Le Montagner")

    measurers = addElement(obs_context, "measurers")
    addDataElement(measurers, "name", "J. Peloton")
    addDataElement(measurers, "name", "R. Le Montagner")

    telescope = addElement(obs_context, "telescope")
    addDataElement(telescope, "design", "Reflector")
    addDataElement(telescope, "aperture", "1.22")
    addDataElement(telescope, "detector", "CCD")

    contact = addElement(obs_context, "comment")
    addDataElement(contact, "line", "[contact@fink-broker.org]")


def make_obs_data(
    submission_id: str,
    ra: float,
    dec: float,
    jd: float,
    magpsf: float,
    sigmapsf: float,
    fid: int,
    obs_data: _Element,
):
    """
    Create the xml block containing one observation of the trajectory.

    Parameters
    ----------
    submission_id : str
        the id for the submission
    ra : float
        right ascension
    dec : float
        declination
    jd : float
        julian date
    magpsf : float
        apparent magnitude
    sigmapsf : float
        rms of the apparent magnitude
    fid : int
        filter id
    obs_data : _Element
        the xml block to which the obs data block will be attached
    """
    obstime = Time(jd, format="jd")
    xsd_good_format = obstime.datetime.isoformat() + "Z"

    optical = addElement(obs_data, "optical")
    addDataElement(optical, "trkSub", submission_id)
    addDataElement(optical, "mode", "CCD")
    addDataElement(optical, "stn", "I41")
    addDataElement(optical, "obsTime", xsd_good_format)
    addDataElement(optical, "ra", f"{ra:.9f}")
    addDataElement(optical, "dec", f"{dec:.9f}")
    addDataElement(optical, "astCat", "Gaia3")
    addDataElement(optical, "mag", f"{magpsf:.4f}")
    addDataElement(optical, "rmsMag", f"{sigmapsf:.4f}")
    addDataElement(optical, "band", "g" if fid == 1 else "r")


def validate(schema: XMLTree.XMLSchema, ades: XMLTree._ElementTree):
    """
    Validate an ades xml with the corresponding ades schema

    Parameters
    ----------
    schema : XMLTree.XMLSchema
        _description_
    ades : XMLTree._ElementTree
        _description_
    """
    try:
        schema.assertValid(ades)
    except XMLTree.DocumentInvalid:
        print(traceback.format_exc())


def readXML(xmlFile):
    """reads in xml file -- this is encoding agnostic.

    Input:  xml file name

    Return: xml tree of xslFile

    Errors:
      The xmlFile might not be readable or might not
      be a valid XML document
    """
    return XMLTree.parse(xmlFile)


def XMLtoSchema(xml_tree: XMLTree._Element) -> XMLTree.XMLSchema:
    """Re-interprets and xml_tree as a schema
    Inputs:
       xml_tree:  an xml tree

    Return Values: the xml tree interpreted as a schema

    This works with xml files read in from files, but
    see XMLtoSchemaViaXSLT below for a caveat if an XSLT
    transform is used to make the xml_tree.


    Errors:
       The xml_tree might not be usable as a schema

    """
    return XMLTree.XMLSchema(xml_tree)  # now parsed as schema


def pdf_to_ades(trajectory: pd.DataFrame, schema: XMLTree.XMLSchema) -> str:
    """
    Convert a trajectory dataframe to a valid aded xml.
    Raise an exception if the ades is not valid regarding the ades validation schema get from github

    Parameters
    ----------
    trajectory : pd.DataFrame
        one trajectory
    schema : XMLTree.XMLSchema
        validation schema get from github

    Returns
    -------
    str
        the ades xml file describing the trajectory
    """
    ades = createADESTree()

    obsBlock = addElement(ades, "obsBlock")

    obsContext = addElement(obsBlock, "obsContext")
    make_obs_context(obsContext)

    obsData = addElement(obsBlock, "obsData")
    trajectory.apply(
        lambda x: make_obs_data(
            x.submission_id, x.ra, x.dec, x.jd, x.magpsf, x.sigmapsf, x.fid, obsData
        ),
        axis=1,
    )

    ades_tree = XMLTree.ElementTree(ades)
    validate(schema, ades_tree)

    str_xml = XMLTree.tostring(ades_tree, pretty_print=True)
    return str_xml


def get_schema_from_github() -> XMLTree.XMLSchema:
    """
    Get the ades validation schema from the Ades github repository.

    Returns
    -------
    XMLTree.XMLSchema
        the ades validation schema
    """
    r = requests.get(
        "https://raw.githubusercontent.com/IAU-ADES/ADES-Master/master/xsd/submit.xsd"
    )
    xml_string = r.content.decode("utf-8")
    xml = XMLTree.fromstring(xml_string)
    schema = XMLtoSchema(xml)
    return schema


def submit_to_mpc(
    xml_ades: XMLTree._ElementTree, ack_msg: str, ack_mail: str, test: bool
):
    """
    Submit an ades xml containing the observation of one trajectories to the MPC.
    Print the response in the standard output.

    Parameters
    ----------
    xml_ades : XMLTree._ElementTree
        an xml ades valid containing the observations
    test : bool
        if true, send the ades xml to the test end-point of MPC.
    """
    files = {
        "ack": (None, ack_msg),
        "ac2": (None, ack_mail),
        "obj_type": (None, "Unclassified"),
        "source": (None, xml_ades),
    }

    submit_url = "https://minorplanetcenter.net/submit_xml"
    if test:
        submit_url = "https://minorplanetcenter.net/submit_xml_test"
    response = requests.post(submit_url, files=files)
    print(response.text)


def mpc_submission_batch(
    trajectory_orb: pd.DataFrame, ack_msg: str, ack_mail: str, test: bool = False
):
    """
    Submit all the trajectories in trajectory_orb to the MPC.

    Parameters
    ----------
    trajectory_orb : pd.DataFrame
        trajectories with orbital elements
    test : bool, optional
        if true, send the observations to the test end-point of MPC, by default False

    Examples
    --------
    >>> test_pdf = pd.read_parquet("fink_fat/test/test_mpc_submission/pdf_test.parquet")
    >>> mpc_submission_batch(test_pdf, "ok test", "toto@gmail.com", True)
    Submission valid
    <BLANKLINE>
    Submission valid
    <BLANKLINE>
    Submission valid
    <BLANKLINE>
    """
    validation_schema = get_schema_from_github()

    uniq_id = trajectory_orb["ssoCandId"].unique()
    submission_id = {
        sso_id: alphabetic_tag(id_int, 8)
        for id_int, sso_id in zip(np.arange(len(uniq_id)), uniq_id)
    }
    trajectory_orb["submission_id"] = trajectory_orb["ssoCandId"].map(submission_id)

    multiple_track, two_point, more_point = prep_traj_for_mpc(trajectory_orb)

    def submit_trajectories(tr: pd.DataFrame):
        if len(tr) != 0:
            t = tr.groupby("submission_id").apply(pdf_to_ades, schema=validation_schema)
            t.apply(submit_to_mpc, ack_msg=ack_msg, ack_mail=ack_mail, test=test)

    submit_trajectories(multiple_track)
    submit_trajectories(two_point)
    submit_trajectories(more_point)


if __name__ == "__main__":  # pragma: no cover
    import sys
    import doctest
    from pandas.testing import assert_frame_equal  # noqa: F401
    import fink_fat.test.test_sample as ts  # noqa: F401
    from unittest import TestCase  # noqa: F401
    import shutil  # noqa: F401
    import datetime  # noqa: F401

    if "unittest.util" in __import__("sys").modules:
        # Show full diff in self.assertEqual.
        __import__("sys").modules["unittest.util"]._MAX_LENGTH = 999999999

    sys.exit(doctest.testmod()[0])
