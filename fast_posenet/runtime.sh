#!/bin/bash

cd /root/armnn/pypose
pip install ../dist/pose-1.0.0-cp27-cp27mu-linux_aarch64.whl
python test.py
