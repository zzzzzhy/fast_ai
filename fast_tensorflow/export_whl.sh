#!/bin/bash


docker run -v `pwd`/export:/export -ti solderzzc/rocketchat:tf_1.6_fast cp /root/tensorflow_temp/tensorflow-1.6.0rc0-cp27-cp27mu-linux_aarch64.whl /export/
