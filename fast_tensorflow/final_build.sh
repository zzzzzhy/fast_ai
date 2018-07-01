#!/bin/bash
docker pull solderzzc/rocketchat:tf_1.6_saved
docker build -f Dockerfile.final -t solderzzc/rocketchat:tf_1.6g_fast_runtime .
