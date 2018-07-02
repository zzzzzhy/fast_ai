#!/bin/bash
cd /root
pip install coremltools
python convert_net_2.py

mv ./net1.tar /export/
mv ./net1.params /export/
mv ./net1 /export/
