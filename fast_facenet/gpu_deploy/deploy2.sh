#!/bin/bash


docker run --rm --privileged \
     -v /sys/class/misc/mali0:/sys/class/misc/mali0 \
     -v /dev/mali0:/dev/mali0 \
     -v `pwd`/runtime_gpu.sh:/root/runtime_gpu.sh \
     -v `pwd`/deploy2.py:/root/deploy.py \
     -ti solderzzc/rocketchat:tvm_06142018_net2 \
     /root/runtime_gpu.sh
