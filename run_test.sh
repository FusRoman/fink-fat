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

set -e

export ROOTPATH=`pwd`

export PYTHONPATH="${SPARK_HOME}/python/test_coverage:$PYTHONPATH"
export COVERAGE_PROCESS_START="${ROOTPATH}/.coveragerc"

python -m pip install -U .

# Run the test suite
for filename in fink_fat/associations/*.py
do
  echo $filename
  # Run test suite + coverage
  coverage run \
    --source=${ROOTPATH} \
    --rcfile ${ROOTPATH}/.coveragerc $filename
done

# Run the test suite
for filename in fink_fat/others/*.py
do
  echo $filename
  # Run test suite + coverage
  coverage run \
    --source=${ROOTPATH} \
    --rcfile ${ROOTPATH}/.coveragerc $filename
done


# Run the test suite
for filename in fink_fat/seeding/*.py
do
  echo $filename
  # Run test suite + coverage
  coverage run \
    --source=${ROOTPATH} \
    --rcfile ${ROOTPATH}/.coveragerc $filename
done

# Run the test suite for the orbit fitting

ORBITFITTING_PATH="fink_fat/orbit_fitting"
echo "${ORBITFITTING_PATH}/mpcobs_files.py"
coverage run --source=${ROOTPATH} --rcfile ${ROOTPATH}/.coveragerc ${ORBITFITTING_PATH}/mpcobs_files.py
echo "${ORBITFITTING_PATH}/orbfit_files.py"
coverage run --source=${ROOTPATH} --rcfile ${ROOTPATH}/.coveragerc ${ORBITFITTING_PATH}/orbfit_files.py
echo "${ORBITFITTING_PATH}/orbfit_local.py"
coverage run --source=${ROOTPATH} --rcfile ${ROOTPATH}/.coveragerc ${ORBITFITTING_PATH}/orbfit_local.py
# echo "${ORBITFITTING_PATH}/orbfit_cluster.py test"
# coverage run --source=${ROOTPATH} --rcfile ${ROOTPATH}/.coveragerc ${ORBITFITTING_PATH}/orbfit_cluster.py "test"
echo "${ORBITFITTING_PATH}/orbfit_merger.py"
coverage run --source=${ROOTPATH} --rcfile ${ROOTPATH}/.coveragerc ${ORBITFITTING_PATH}/orbfit_merger.py
# echo "${ORBITFITTING_PATH}/orbfit_ephem.py"
# coverage run --source=${ROOTPATH} --rcfile ${ROOTPATH}/.coveragerc ${ORBITFITTING_PATH}/orbfit_ephem.py

for filename in fink_fat/command_line/*.py
do
  echo $filename
  # Run test suite + coverage
  coverage run \
    --source=${ROOTPATH} \
    --rcfile ${ROOTPATH}/.coveragerc $filename "test"
done

for filename in fink_fat/test/*/*.py
do
  echo $filename
  coverage run \
    --source=${ROOTPATH} \
    --rcfile ${ROOTPATH}/.coveragerc $filename
done

for filename in fink_fat/test/*.py
do
  echo $filename
  coverage run \
    --source=${ROOTPATH} \
    --rcfile ${ROOTPATH}/.coveragerc $filename
done


# run the test suite for the kalman filter
echo "run kalman test"
coverage run --source=${ROOTPATH} --rcfile=${ROOTPATH}/.coveragerc -m \
  pytest fink_fat/test/kalman_test/update_kalman_test/test_runner.py


coverage combine

unset COVERAGE_PROCESS_START

coverage report -m
coverage html