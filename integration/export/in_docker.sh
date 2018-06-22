#!/bin/bash

mkdir -p /build/install/include
mkdir -p /build/install/lib

cp /build/ComputeLibrary/build/*.so /build/install/lib/
cp /build/ComputeLibrary/build/*.a /build/install/lib/
cp /build/ComputeLibrary/include/* /build/install/ -rf

cp /build/tvm/build/*.so /build/install/lib/

cp -rf /build/tvm/topi/include/topi /build/install/include/
cp -rf /build/tvm/include/tvm /build/install/include/
cp -rf /build/tvm/nnvm/include/nnvm /build/install/include/
cp -rf /build/tvm/dlpack/include/dlpack /build/install/include/
cp -rf /build/tvm/dmlc-core/include/dmlc /build/install/include/

mkdir -p /build/wheels/

cd /build/tvm/python
python setup.py sdist bdist_wheel
ls -alh ./dist/
cp /build/tvm/python/dist/*.whl /build/wheels/

cd /build/tvm/topi/python/
python setup.py sdist bdist_wheel
cp /build/tvm/topi/python/dist/*.whl /build/wheels/

cd /build/tvm/nnvm/python
python setup.py sdist bdist_wheel
cp /build/tvm/nnvm/python/dist/*.whl /build/wheels/

cd /build
tar -zcf install.tgz ./install

cp /build/install.tgz /export/
cp /build/wheels /export/ -rf

