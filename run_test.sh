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
export COVERAGE_PROCESS_START="${ROOTPATH}/.coveragerc"

set -e

python -m pip install .

# Run the test suite
for filename in fink_fat/associations/*.py
do
  echo $filename
  # Run test suite + coverage
  coverage run \
    --append \
    --source=${ROOTPATH} \
    --rcfile ${ROOTPATH}/.coveragerc $filename
done

echo fink_fat/orbit_fitting/orbfit_local.py
# Run test suite + coverage
coverage run \
  --append \
  --source=${ROOTPATH} \
  --rcfile ${ROOTPATH}/.coveragerc fink_fat/orbit_fitting/orbfit_local.py

echo bin/utils_cli.py
coverage run \
  --append \
  --source=${ROOTPATH} \
  --rcfile ${ROOTPATH}/.coveragerc bin/utils_cli.py

echo fink_fat/test/continuous_integration.py
# Run test suite + coverage
coverage run \
  --append \
  --source=${ROOTPATH} \
  --rcfile ${ROOTPATH}/.coveragerc fink_fat/test/continuous_integration.py

unset COVERAGE_PROCESS_START

coverage report -m
coverage html