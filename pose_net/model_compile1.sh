#!/bin/bash

docker run --rm -v `pwd`/export:/export -v `pwd`/tensorflow.py:/root/tensorflow_frontend.py -v `pwd`/frozen_model.pb:/root/frozen_model.pb -v `pwd`/convert_net_1.py:/root/convert_net_1.py -v `pwd`/convert_net_1.py:/root/convert_net_1.py -v `pwd`/runtime1.sh:/root/runtime1.sh -ti tvm_tf /root/runtime1.sh
