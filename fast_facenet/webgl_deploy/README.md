
# How to Build TVM with WebGL support

## Clone Source Code
```
git clone --recursive https://github.com/dmlc/tvm -b v0.4
cd tvm
```
## Install Packages

```
cd docker/install
sudo bash ubuntu_install_core.sh
sudo bash ubuntu_install_python.sh
sudo bash ubuntu_install_python_package.sh
sudo bash ubuntu_install_emscripten.sh
sudo bash ubuntu_install_opengl.sh
sudo bash ubuntu_install_opencl.sh
```

## Edit config.cmake
```
set(USE_CUDA OFF)
set(USE_ROCM OFF)
set(USE_OPENCL ON)
set(USE_METAL OFF)
set(USE_VULKAN OFF)
set(USE_OPENGL OFF)
set(USE_RPC ON)
set(USE_GRAPH_RUNTIME ON)
set(USE_GRAPH_RUNTIME_DEBUG OFF)
set(USE_LLVM llvm-config)
set(USE_BLAS openblas)
set(USE_RANDOM OFF)
set(USE_NNPACK OFF)
set(USE_CUDNN OFF)
set(USE_CUBLAS OFF)
set(USE_MIOPEN OFF)
set(USE_MPS OFF)
set(USE_ROCBLAS OFF)
set(USE_SORT OFF)
```
```
mkdir build
cd build
```



```
cd python; python2 setup.py install --user; cd .. && \
    cd topi/python; python2 setup.py install --user; cd ../.. && \
    cd nnvm/python; python2 setup.py install --user; cd ../..
```

```
cd python; python3 setup.py install --user; cd .. && \
    cd topi/python; python3 setup.py install --user; cd ../.. && \
    cd nnvm/python; python3 setup.py install --user; cd ../..
```
