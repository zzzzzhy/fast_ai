#!/bin/bash

docker run --rm --privileged \
     -v /sys/class/misc/mali0:/sys/class/misc/mali0 \
     -v/dev/mali0:/dev/mali0 \
     -v `pwd`/:/root/test \
     -v `pwd`/export:/export \
     -v `pwd`/runtime_gpu.sh:/root/runtime_gpu.sh \
     -v `pwd`/deploy_od.py:/root/deploy_od.py \
     -v `pwd`/export/_convert.so:/root/_convert.so \
     -v `pwd`/export/convert.py:/root/convert.py \
     -v `pwd`/export/od.tar:/root/od.tar \
     -v `pwd`/export/od.cfg:/root/od.cfg \
     -v `pwd`/export/libruntime.so:/root/libruntime.so \
     -v `pwd`/export/od.params:/root/od.params \
     -v `pwd`/export/od:/root/od \
     -v `pwd`/export/od.names:/root/od.names \
     -v `pwd`/export/dog.jpg:/root/dog.jpg \
     -ti lambdazhang/raidcdn:tf1.8_runtime_mx \
     /root/runtime_gpu.sh
