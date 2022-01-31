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
## Script to launch the python test suite and measure the coverage.


# source the intel fortran compiler and libraries, needed for OrbFit 
source /opt/intel/oneapi/setvars.sh

set -e

export ROOTPATH=`pwd`
export COVERAGE_PROCESS_START="${ROOTPATH}/.coveragerc"

FILE1=alert_association/performance_test.py
FILE2=alert_association/night_report.py
FILE3=alert_association/ephem.py

make simple_build


# Run the test suite
for filename in alert_association/*.py
do
  case $filename in
  $FILE1 ) continue ;;
  $FILE2 ) continue ;;
  $FILE3 ) continue ;;
  * )

    echo $filename
    # Run test suite + coverage
    coverage run \
      --append \
      --source=${ROOTPATH} \
      --rcfile ${ROOTPATH}/.coveragerc $filename
    ;;
  esac
done

for filename in alert_association/orbit_fitting/*.py
do
  echo $filename
  # Run test suite + coverage
  coverage run \
    --append \
    --source=${ROOTPATH} \
    --rcfile ${ROOTPATH}/.coveragerc $filename
done

unset COVERAGE_PROCESS_START

coverage report -m
coverage html