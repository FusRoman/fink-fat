#!/bin/bash

echo "fink_tmp /tmp/ramdisk  tmpfs  defaults,size=2G,x-gvfs-show  0  0" >> /etc/fstab
mkdir -p /tmp/ramdisk
mount -a
