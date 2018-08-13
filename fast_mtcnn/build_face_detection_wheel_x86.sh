#!/bin/bash
rm -rf mtcnn-ncnn/build
docker build -t tmp_build_x86 -f Dockerfile.x86 .

docker run --rm -v `pwd`/mtcnn_ncnn_neon/:/build/mtcnn-ncnn -v `pwd`/export/:/export -ti tmp_build_x86 /export/build_in_docker.sh
