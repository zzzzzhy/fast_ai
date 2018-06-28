#!/bin/bash
FILES=/root/imgs/*
#Xvfb :1 &
#export DISPLAY=:1
for f in $FILES
do
  echo "Processing $f file..."
  ./ncnn/build/examples/ssd/ssdmobilenet $f
done
