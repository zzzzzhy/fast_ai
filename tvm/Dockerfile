FROM solderzzc/rocketchat:mxnet
RUN apt-get install -y llvm-4.0 scons libopenblas-dev git cmake build-essential \
    liblapack-dev libblas-dev libopencv-dev opencl-headers

ADD Mali/libmali-midgard-4th-r13p0.so /usr/lib/libmali.so
RUN ln -s /usr/lib/libmali.so /usr/lib/libOpenCL.so && \
    ln -s /usr/lib/libmali.so /usr/lib/libGLES_mali.so

WORKDIR /build
RUN git clone --recursive https://github.com/dmlc/nnvm.git && \
    git clone https://github.com/ARM-software/ComputeLibrary.git --branch v17.12

RUN cd nnvm/tvm && \
    make USE_OPENCL=1  LLVM_CONFIG=llvm-config-4.0 -j4 && \
    cd .. && make -j4

RUN cd ComputeLibrary && \
    scons Werror=1 neon=1 opencl=1 examples=1 os=linux arch=arm64-v8a embed_kernels=1 build=native -j4
ADD acl_test.cc /build/ComputeLibrary/acl_test.cc
RUN cd /build/ComputeLibrary/build && cp *.so *.a /usr/lib/
RUN cd ComputeLibrary && \
    g++ acl_test.cc build/utils/*.o -O2 -std=c++11 -I. -Iinclude -Lbuild -larm_compute -larm_compute_graph -larm_compute_core -lOpenCL -o acl_test
#RUN /build/ComputeLibrary/acl_test all

ENV PYTHONPATH "/build/nnvm/python:/build/nnvm/tvm/python:/build/nnvm/tvm/topi/python"
ADD benchmark /root/benchmark
ADD from_mxnet.py /root/from_mxnet.py
