#!/bin/bash

cd runtime
docker build -t solderzzc/rocketchat:mxnet .
docker push solderzzc/rocketchat:mxnet
