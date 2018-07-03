#!/bin/bash

cd /root/armnn/pypose
make -C /root/armnn/build/
cp /root/armnn/build/pypose/pose.so ./
python test.py
