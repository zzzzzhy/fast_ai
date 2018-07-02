#!/bin/bash

docker run --rm -v `pwd`/export:/export -v `pwd`/posenet513_v1_075.mlmodel:/root/posenet513_v1_075.mlmodel -v `pwd`/convert_net_2.py:/root/convert_net_2.py -v `pwd`/runtime2.sh:/root/runtime2.sh -ti solderzzc/rocketchat:tvm_06282018_x86 /root/runtime2.sh
