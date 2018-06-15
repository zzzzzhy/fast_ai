#!/bin/bash
cd /root
python convert_net_2.py

mv ./net2.tar /export/
mv ./net2.params /export/
mv ./net2 /export/
