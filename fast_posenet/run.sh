#!/bin/bash
docker build -t local_test .
docker run --privileged -v `pwd`/:/root/test -v /sys/class/misc/mali0:/sys/class/misc/mali0 -v/dev/mali0:/dev/mali0 \
      -v`pwd`/export:/export -ti local_test /export/export.sh
docker run --privileged -v `pwd`/:/root/test -v /sys/class/misc/mali0:/sys/class/misc/mali0 -v/dev/mali0:/dev/mali0 \
      -v`pwd`/export:/export -ti local_test /root/runtime.sh
