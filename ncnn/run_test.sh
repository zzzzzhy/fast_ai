#!/bin/bash
FILES=/root/imgs/*
for f in $FILES
do
  echo "Processing $f file..."
  ./ncnn/build/examples/ssd/ssdmobilenet $f
done
