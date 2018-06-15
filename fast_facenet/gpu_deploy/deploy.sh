#!/bin/bash

docker run --rm --privileged \
     -v /sys/class/misc/mali0:/sys/class/misc/mali0 \
     -v/dev/mali0:/dev/mali0 \
     -v `pwd`/runtime_gpu.sh:/root/runtime_gpu.sh \
     -v `pwd`/deploy.py:/root/deploy.py \
     -v `pwd`/export/net1.tar:/root/net1.tar \
     -v `pwd`/export/net1.params:/root/net1.params \
     -v `pwd`/export/net1:/root/net1 \
     -ti solderzzc/rocketchat:tvm_06142018 \
     /root/runtime_gpu.sh
