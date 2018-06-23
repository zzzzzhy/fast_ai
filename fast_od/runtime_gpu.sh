#!/bin/bash
cp /usr/install/lib/libnnvm_compiler.so /usr/local/lib/python2.7/dist-packages/nnvm/libnnvm_compiler.so
apt-get install -y python-cffi
cd /root; python deploy_od.py
