#!/bin/bash
cd cross_build
docker build -t mxnet/build.arm64 -f Dockerfile.build.arm64 .
docker run --rm -v `pwd`/build/:/work/build -t mxnet/build.arm64 /work/runtime_functions.sh build_arm64

cd ../

cp ./build/mxnet-1.2.0-py2.py3-none-any.whl ./runtime/assets
cp ./build/libmxnet.so ./runtime/assets

cd runtime

docker build -t solderzzc/rocketchat:mxnet_save -f Dockerfile.save .

