#!/bin/bash
docker build -t solderzzc/rocketchat:tf_1.6_fast .
./export_whl.sh
docker push solderzzc/rocketchat:tf_1.6_saved
