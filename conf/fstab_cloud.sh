#!/bin/bash

# Total number of slaves for the cluster
nslaves=11

# Installation on the driver
install_fstab.sh

# Intallation on the slaves
for i in $(seq 1 $nslaves); do
  echo slave $i configuring fstab ...
  scp install_fstab.sh slave$i:/tmp
  ssh slave$i /tmp/install_fstab.sh
done
