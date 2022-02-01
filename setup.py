#!/usr/bin/env python

from setuptools import setup, find_packages

print(find_packages())

setup(
    name="Asteroids and Associations",
    version="0.2",
    description="Associations of ZTF alerts to detect moving objects",
    author="Roman Le Montagner",
    author_email="roman.le-montagner@ijclab.in2p3.fr",
    url="https://github.com/FusRoman/Asteroids_and_Associations",
    packages=find_packages(),
    license="Apache-2.0 License",
    platforms="Linux",
)
