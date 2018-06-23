#!/bin/bash

docker build -f Dockerfile.x86 -t tmp_build_od_model .
docker run -v `pwd`/export:/export -ti tmp_build_od_model /export/export.sh
