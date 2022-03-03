#!/usr/bin/env python

from setuptools import setup, find_packages

print(find_packages(where="fink_fat"))

setup(
    name="fink-fat",
    version="0.2.5",
    description="Associations of alerts to detect moving objects",
    author="Roman Le Montagner",
    author_email="roman.le-montagner@ijclab.in2p3.fr",
    url="https://github.com/FusRoman/fink-fat",
    packages=find_packages(),
    package_data={"fink_fat": ["data/month=*"]},
    license="Apache-2.0 License",
    platforms="Linux Debian distribution",
    project_urls={
        "Documentation": "https://github.com/FusRoman/fink-fat/wiki",
        "Source": "https://github.com/FusRoman/fink-fat",
    },
    python_requires=">=3.7",
)
