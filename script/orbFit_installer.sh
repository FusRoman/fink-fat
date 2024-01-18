#!/bin/bash
# Copyright 2022
# Author: Le Montagner Roman
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
## Script to install the OrbFit software

set -e
message_help="""
Download and install locally OrbFit \n\n
Usage:\n
    ./orbFit_installer.sh [--install_path] [-h] \n\n
Specify the OrbFit installation path with --install_path.\n
e.g. ./orbFit_installer.sh --install_path /home\n
Use -h to display this help.
"""


# Show help if no arguments is given
if [[ $1 == "" ]]; then
  echo -e $message_help
  exit 1
fi

# Grab the command line arguments
while [ "$#" -gt 0 ]; do
  case "$1" in
    -h)
        echo -e $message_help
        exit
        ;;
    --install_path)
        if [[ $2 == "" ]]; then
          echo "$1 requires an argument" >&2
          exit 1
        fi
        ORBLOCATE="$2"
        shift 2
        ;;
  esac
done

if [[ $ORBLOCATE == "" ]]; then
  echo "You need to specify the OrbFit installation path with the option --install_path."
  exit
fi

if [[ ! -d $ORBLOCATE ]]; then
    echo "The path ${ORBLOCATE} does not exist"
    exit
fi

wget http://adams.dm.unipi.it/orbfit/OrbFit5.0.7.tar.gz


tar -xf OrbFit5.0.7.tar.gz -C $ORBLOCATE

rm OrbFit5.0.7.tar.gz

cd $ORBLOCATE

./config -O gfortran

make

cd lib/

wget https://ssd.jpl.nasa.gov/ftp/eph/planets/Linux/de440/linux_p1550p2650.440

mv linux_p1550p2650.440 jpleph

echo "export ORBFIT_HOME=${ORBLOCATE}" >> ~/.bash_profile

echo "OrbFit installation done, location is ${ORBLOCATE}, source your .bash_profile before using fink-fat."