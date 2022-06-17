#!/bin/bash
# Copyright 2021 Le Montagner Roman
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

ORBLOCATE=~/OrbitFit

aria2c -x8 http://adams.dm.unipi.it/orbfit/OrbFit5.0.7.tar.gz

if [[ ! -d $ORBLOCATE ]]
then
    echo "OrbitFit/ directory does not exists on your filesystem."
    mkdir $ORBLOCATE
fi

tar -xf OrbFit5.0.7.tar.gz -C $ORBLOCATE

rm OrbFit5.0.7.tar.gz

cd $ORBLOCATE

./config -O gfortran

make

cd lib/

aria2c -x8 https://ssd.jpl.nasa.gov/ftp/eph/planets/Linux/de440/linux_p1550p2650.440

mv linux_p1550p2650.440 jpleph

echo "export ORBFIT_HOME=${ORBLOCATE}" >> ~/.bash_profile

source ~/.bash_profile

echo "OrbFit installation done, location is ${ORBLOCATE}"