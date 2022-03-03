#!/bin/bash

# Total number of slaves for the cluster
nslaves=11

# Installation on the driver
cp ../orbFit_installer.sh /tmp
/tmp/orbFit_installer.sh

# Intallation on the slaves
for i in $(seq 1 $nslaves); do
  echo slave $i installing orbFit ...
  scp ../orbFit_installer.sh slave$i:/tmp
  ssh slave$i /tmp/orbFit_installer.sh
done
