#!/usr/bin/env python

from setuptools import setup, find_packages
import fink_fat


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="fink-fat",
    version=fink_fat.__version__,
    description="Associations of astronomical alerts to detect moving objects",
    author="Roman Le Montagner",
    author_email="roman.le-montagner@ijclab.in2p3.fr",
    url="https://github.com/FusRoman/fink-fat",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    package_data={
        "fink_fat": [
            "data/fink_fat.conf",
            "orbit_fitting/AST17.*",
            "orbit_fitting/template.oop",
        ]
    },
    install_requires=[
        "docopt>=0.6.2",
        "terminaltables>=3.1.10",
        "fink-science>=0.5.1",
        "fink-utils>=0.13.8",
        "numpy==1.23.5",
        "pandas==1.3.5",
        "scikit-learn==1.0.2",
        "astropy==5.1",
        "fastavro==1.6.0",
        "pyarrow==9.0.0",
        "pytest==7.2.2",
        "sbpy==0.4.0",
        "scipy>=1.4.1",
        "healpy==1.16.6",
        "pandera==0.18.0"
    ],
    entry_points={
        "console_scripts": ["fink_fat=fink_fat.command_line.fink_fat_cli:main"]
    },
    license="Apache-2.0 License",
    platforms="Linux Debian distribution",
    project_urls={
        "Documentation": "https://github.com/FusRoman/fink-fat/wiki",
        "Source": "https://github.com/FusRoman/fink-fat",
    },
    python_requires=">=3.7",
)
