#!/bin/bash

# script running Fink-FAT 1.0 in offline mode between input_start and input_end

input_start=$1
input_end=$3
offline_path=$5
path_config=$6
log_path=$7
verbose=$8

# After this, startdate and enddate will be valid ISO 8601 dates,
# or the script will have aborted when it encountered unparseable data
# such as input_end=abcd
startdate=$(date -I -d "$input_start") || exit -1
enddate=$(date -I -d "$input_end")     || exit -1

d="$startdate"
while [ "$d" != "$enddate" ]; do 
  echo "PROCESS DATE $d"
  echo

  echo "------------------------ RUN ROID SCIENCE MODULE ------------------------"
  python $offline_path/launch_roid.py \
    ${d:0:4} ${d:5:2} ${d:8:2} \
    $offline_path $path_config $verbose \
    >> $log_path/roid_stream_${d:0:4}_${d:5:2}_${d:8:2}.log 2>&1

  echo "------------------------ RUN FINK FAT ASSOCIATION -----------------------"
  fink_fat associations fitroid --night $d --config $path_config --verbose \
    >> $log_path/fitroid_associations_${d:0:4}_${d:5:2}_${d:8:2}.log 2>&1

  echo
  echo "### GO TO NEXT DAY ###"
  d=$(date -I -d "$d + 1 day")
done
