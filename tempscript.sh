#!/bin/bash

x=1
while [ $x -le 60 ]
do
  timeout 1s top -b > cpuusg
  sleep 2
  grep "%Cpu" cpuusg >> cpudata
  x=$(( $x + 1 ))
done
