#!/bin/bash
if [[ -z "$1" ]]; then
  docker run -p 9090:9090 --rm --privileged -v `pwd`/../:/root/fast_ai -v /sys/class/misc/mali0:/sys/class/misc/mali0 -v/dev/mali0:/dev/mali0 -v`pwd`/export:/export -ti armnn_aarch64:latest /bin/bash
else
  if [[ -z "$2" ]]; then
    docker run -p 9090:9090 --rm --privileged -v `pwd`/../:/root/fast_ai -v /sys/class/misc/mali0:/sys/class/misc/mali0 -v/dev/mali0:/dev/mali0 -v`pwd`/export:/export -ti $1 /bin/bash
  else
    docker run -p 9090:9090 --rm --privileged -v `pwd`/../:/root/fast_ai -v /sys/class/misc/mali0:/sys/class/misc/mali0 -v/dev/mali0:/dev/mali0 -v`pwd`/export:/export -ti $1 $2
  fi
fi
