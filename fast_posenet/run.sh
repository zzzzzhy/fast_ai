#!/bin/bash
if [[ -z "$1" ]]; then
  docker run --privileged -v `pwd`/:/root/test -v /sys/class/misc/mali0:/sys/class/misc/mali0 -v/dev/mali0:/dev/mali0 -v`pwd`/export:/export -ti armnn_aarch64:latest /bin/bash
else
  docker run --privileged -v `pwd`/:/root/test -v /sys/class/misc/mali0:/sys/class/misc/mali0 -v/dev/mali0:/dev/mali0 -v`pwd`/export:/export -ti $1 /bin/bash
fi
