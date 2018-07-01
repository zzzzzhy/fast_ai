#!/bin/bash
docker run --privileged -v `pwd`/:/root/test -v /sys/class/misc/mali0:/sys/class/misc/mali0 -v/dev/mali0:/dev/mali0 -v`pwd`/export:/export -ti solderzzc/rocketchat:tf_1.6g_fast_runtime /bin/bash
