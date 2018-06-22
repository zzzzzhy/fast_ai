#!/bin/bash
docker build -f Dockerfile.build_runtime -t build_runtime .
rm ./export/wheels/*
rm ./export/install.tgz
docker run --rm -v `pwd`/export:/export -ti build_runtime  /export/in_docker.sh

