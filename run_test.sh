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

export ROOTPATH=`pwd`


export COVERAGE_PROCESS_START="${ROOTPATH}/.coveragerc"

# Run the test suite on the utilities
for filename in alert_association/first_method/*.py
do
  # Run test suite + coverage
  coverage run \
    --source=${ROOTPATH} \
    --rcfile ${ROOTPATH}/.coveragerc $filename
done

# Combine individual reports in one
coverage combine

unset COVERAGE_PROCESS_START

coverage report
coverage html