#!/usr/bin/env python

from distutils.core import setup

setup(
    name="Asteroids and Associations",
    version="0.2",
    description="Associations of ZTF alerts to detect moving objects",
    author="Roman Le Montagner",
    author_email="roman.le-montagner@ijclab.in2p3.fr",
    url="https://github.com/FusRoman/Asteroids_and_Associations",
    packages=["alert_association", "alert_association.orbit_fitting"],
)
