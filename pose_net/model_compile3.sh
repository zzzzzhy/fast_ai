#!/bin/bash

docker run --rm -v `pwd`/export:/export -v `pwd`/model.onnx:/root/model.onnx -v `pwd`/convert_net_3.py:/root/convert_net_3.py -v `pwd`/convert_net_3.py:/root/convert_net_3.py -v `pwd`/runtime3.sh:/root/runtime3.sh -ti tvm_onnx /root/runtime3.sh
