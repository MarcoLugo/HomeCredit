#!/bin/bash
# Marco Lugo

for fname in *.csv *.txt;
  do
    newfname=$(echo "$fname" | awk -F '_' '{print $2}' )
    if (( $(echo "$newfname < 0.79" | bc -l) )); then
      mv $fname archive/$fname
      echo $fname
    fi
done

