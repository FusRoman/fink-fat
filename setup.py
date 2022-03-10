#!/usr/bin/env python

from setuptools import setup, find_packages
import fink_fat

setup(
    name="fink-fat",
    version=fink_fat.__version__,
    description="Associations of astronomical alerts to detect moving objects",
    author="Roman Le Montagner",
    author_email="roman.le-montagner@ijclab.in2p3.fr",
    url="https://github.com/FusRoman/fink-fat",
    packages=find_packages(),
    package_data={
        "fink_fat": [
            "data/month=*",
            "data/fink_fat.conf",
            "orbit_fitting/AST17.*",
            "orbit_fitting/template.oop",
        ]
    },
    install_requires=[
    'docopt>=0.6.2',
    'terminaltables>=3.1.10',
    'fink-science>=0.5.1',
    'numpy>=1.17',
    'pandas>=1.3.5',
    'astropy>=4.2.1'
    ],
    entry_points={"console_scripts": ["fink_fat=bin.fink_fat_cli:main"]},
    license="Apache-2.0 License",
    platforms="Linux Debian distribution",
    project_urls={
        "Documentation": "https://github.com/FusRoman/fink-fat/wiki",
        "Source": "https://github.com/FusRoman/fink-fat",
    },
    python_requires=">=3.7",
)
