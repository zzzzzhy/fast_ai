#!/bin/bash
docker build -t tmp_build .

docker run --rm -v `pwd`/mtcnn_ncnn_neon/:/build/mtcnn-ncnn -v `pwd`/export/:/export -ti tmp_build /export/build_in_docker.sh
