#!/bin/bash

docker run --rm -v `pwd`/export:/export -v `pwd`/../model-y1-test2:/root/model-y1-test2 -v `pwd`/../gpu_deploy/convert_net_1.py:/root/convert_net_1.py -v `pwd`/runtime.sh:/root/runtime.sh -ti solderzzc/rocketchat:tvm_06152018_x86 /root/runtime.sh
