FROM arm64v8/debian:stretch
RUN apt-get update && apt-get install -y git scons cmake wget unzip curl build-essential
WORKDIR /root
RUN wget https://github.com/google/protobuf/releases/download/v3.5.1/protobuf-cpp-3.5.1.tar.gz && \
    tar -xzvf protobuf-cpp-3.5.1.tar.gz && \
    cd protobuf-3.5.1 && \
    ./configure --prefix=/usr && \
    make -j6 install

RUN wget https://dl.bintray.com/boostorg/release/1.64.0/source/boost_1_64_0.tar.gz && \
    tar -zxf boost_1_64_0.tar.gz && \
    ls -alh && cd boost_1_64_0 && \
    ./bootstrap.sh --prefix=/usr --without-libraries=python && \
    ./b2 install -j 6

ENV ARMNN_DIR /root/armnn
ENV CL_DIR /root/ComputeLibrary

RUN git clone https://github.com/ARM-software/ComputeLibrary && \
    cd /root/ComputeLibrary && \
    scons -j 6 arch=arm64-v8a neon=1 opencl=1 openvx=0 Werror=0 debug=0 validation_tests=1 benchmark_tests=1 debug=0 Werror=0 asserts=1 examples=0 standalone=0 cppthreads=1 openmp=1 

RUN git clone https://github.com/Arm-software/armnn
ENV TF_VER 1.5.1
RUN cd /root && wget https://github.com/tensorflow/tensorflow/archive/v${TF_VER}.tar.gz && \
    tar -zxvf v$TF_VER.tar.gz && \
    cd /root/tensorflow-$TF_VER && \
    ${ARMNN_DIR}/scripts/generate_tensorflow_protobuf.sh /root/v$TF_VER/protobuf /usr

RUN cd /root && git clone https://github.com/jschmidt42/stb-cmake
ADD ./Patchs/CMakeLists_tests.txt ${ARMNN_DIR}/tests/CMakeLists.txt
ADD ./Patchs/CMakeLists.txt ${ARMNN_DIR}/CMakeLists.txt

RUN cd /root && \
    git clone https://github.com/BVLC/caffe && \
    cd /root/caffe/src/caffe/proto && \
    protoc --cpp_out=. caffe.proto
RUN cd ${ARMNN_DIR} && sed -i 's/SHARED/STATIC/g' CMakeLists.txt && \
    mkdir build && cd build  && \
    cmake -DBUILD_UNIT_TESTS=1 -DARMCOMPUTE_ROOT=$CL_DIR -DARMCOMPUTE_BUILD_DIR=$CL_DIR/build -DTF_GENERATED_SOURCES=/root/v$TF_VER/protobuf -DBUILD_TF_PARSER=ON -DBUILD_CAFFE_PARSER=ON -DPROTOBUF_ROOT=/usr -DBUILD_TESTS=1 -DTHIRD_PARTY_INCLUDE_DIRS="/root/stb-cmake" -DARMCOMPUTECL=ON -DARMCOMPUTENEON=OFF -DCAFFE_GENERATED_SOURCES=/root/caffe/src ../
    
RUN cd ${ARMNN_DIR}/build && make -j6

RUN cd ${ARMNN_DIR}/build/tests && \
    wget http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224.tgz && \
    tar -zxvf mobilenet_v1_1.0_224.tgz && \
    mv mobilenet_v1_1.0_224_frozen.pb mobilenet_v1_1.0_224_fp32.pb
RUN cat ${ARMNN_DIR}/build/tests/mobilenet_v1_1.0_224_info.txt

ADD ./Patchs/TfMobileNet-Armnn.cpp ${ARMNN_DIR}/tests/TfMobileNet-Armnn/TfMobileNet-Armnn.cpp 
ADD ./imgs /root/imgs
ADD Mali/libmali-midgard-4th-r13p0.so /usr/lib/libmali.so
RUN apt-get update && apt-get install -y libdrm2 libx11-xcb1 libxcb-dri2-0 && \
    ln -s /usr/lib/libmali.so /usr/lib/libOpenCL.so && \
    ln -s /usr/lib/libmali.so /usr/lib/libGLES_mali.so
ADD ./app/Stub_CMakeLists.txt ${ARMNN_DIR}/CMakeLists.txt
ADD ./app/app ${ARMNN_DIR}/app

RUN find ${ARMNN_DIR}/tests  -type f -readable -writable -exec sed -i "s/armnn::Compute::CpuAcc/armnn::Compute::GpuAcc/g" {} \; && \
    cd ${ARMNN_DIR}/build && make -j6
RUN echo "Please run on board: \
    cd ${ARMNN_DIR}/build/tests && ./TfMobileNet-Armnn -d /root/imgs --validation-file-in ../../tests/TfMobileNet-Armnn/Validation.txt  --validation-file-out ./1.txt  -m ./ "

