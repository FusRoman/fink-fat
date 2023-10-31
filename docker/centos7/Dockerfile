# Copyright 2022 AstroLab Software
# Author: Roman Le Montagner
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
ARG PYTHON_VERSION=py37_4.11.0
ARG SPARK_VERSION=3.1.3
ARG HADOOP_VERSION=hadoop2.7

FROM centos:centos7
LABEL maintainer="roman.le-montagner@ijclab.in2p3.fr"

ARG PYTHON_VERSION
ENV PYTHON_VERSION=$PYTHON_VERSION

ARG SPARK_VERSION
ENV SPARK_VERSION=$SPARK_VERSION

ARG HADOOP_VERSION
ENV HADOOP_VERSION=$HADOOP_VERSION

ENV USRLIBS /home/libs
ENV PATH=${USRLIBS}/miniconda/bin:/usr/local/bin:${PATH}

WORKDIR $USRLIBS

RUN mkdir fink-fat

COPY . fink-fat

# Install system build dependencies
RUN yum -y update \
    && yum -y install which git wget java-11-openjdk-devel \
    && echo "export JAVA_HOME=$(dirname $(dirname $(readlink -f $(type -P java))))" > /etc/profile.d/javahome.sh \
    && yum -y groupinstall "Development Tools" \
    && yum -y clean all \
    && rm -rf /var/cache
# && echo "export JAVA_HOME=$(dirname $(dirname $(readlink -f $(type -P java))))" > /etc/profile.d/javahome.sh

# install python and the dependencies
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-${PYTHON_VERSION}-Linux-x86_64.sh -O ~/miniconda.sh \
    && bash ~/miniconda.sh -b -p ${USRLIBS}/miniconda

RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && cd ${USRLIBS}/fink-fat/script \
    && source ./install_python_deps.sh \
    && cd ${USRLIBS}


# install spark
RUN wget --quiet https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-${HADOOP_VERSION}.tgz \
    && tar -xf spark-${SPARK_VERSION}-bin-${HADOOP_VERSION}.tgz \
    && rm spark-${SPARK_VERSION}-bin-${HADOOP_VERSION}.tgz \
    && echo "export SPARK_HOME=$PWD/spark-${SPARK_VERSION}-bin-${HADOOP_VERSION}" > /etc/profile.d/sparkhome.sh

# install OrbFit
RUN yum -y install epel-release \
    && yum -y install aria2 \
    && mkdir OrbFit \
    && cd ${USRLIBS}/fink-fat/script \
    && source ./orbFit_installer.sh --install_path ${USRLIBS}/OrbFit \
    && echo "export ORBFIT_HOME=${USRLIBS}/OrbFit" > /etc/profile.d/orbfithome.sh \
    && cd ${USRLIBS}

ENV SPARK_HOME=$USRLIBS/spark-${SPARK_VERSION}-bin-${HADOOP_VERSION}
ENV SPARKLIB=${SPARK_HOME}/python:${SPARK_HOME}/python/lib/py4j-0.10.9-src.zip
ENV ORBFIT_HOME=${USRLIBS}/OrbFit
# ENV JAVA_HOME="$(dirname $(dirname $(readlink -f $(type -P java))))"
ENV PATH=${SPARK_HOME}/bin:${SPARK_HOME}/sbin:/usr/local/bin:${PATH}

ENV BINPATH=$PATH
ENV PYTHONPATH=${SPARKLIB}:${PYTHONPATH}