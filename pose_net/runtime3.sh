#!/bin/bash
cd /root
pip install onnx
python convert_net_3.py

mv ./net1.tar /export/
mv ./net1.params /export/
mv ./net1 /export/
