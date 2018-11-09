#!/bin/bash

docker run --rm -v `pwd`/export:/export -v `pwd`/../model-r50-am-lfw:/root/model-r50-am-lfw -v `pwd`/../gpu_deploy/convert_net_2_android_v7a.py:/root/convert_net_2.py -v `pwd`/runtime2.sh:/root/runtime2.sh -ti solderzzc/rocketchat:tvm_06152018_x86 /root/runtime2.sh

#docker build -t solderzzc/rocketchat:tvm_06142018_net2 .
#docker push solderzzc/rocketchat:tvm_06142018_net2
