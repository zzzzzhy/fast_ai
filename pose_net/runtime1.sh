#!/bin/bash

#ls /build -alh
#cp /root/tensorflow_frontend.py /build/tvm/nnvm/python/nnvm/frontend/tensorflow.py
#cd /build/tvm/nnvm/python
#python setup.py install --user
#pip install tensorflow

cd /root

python convert_net_1.py
mv ./net1.tar /export/
mv ./net1.params /export/
mv ./net1 /export/
