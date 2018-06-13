FROM arm64v8/ubuntu:18.04
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y git scons cmake wget unzip curl build-essential \
    libprotobuf-dev protobuf-compiler libopencv-dev
WORKDIR /root
RUN git clone https://github.com/Tencent/ncnn
RUN sed -i -e 's/# add_subdirectory(examples)/add_subdirectory(examples)/g' /root/ncnn/CMakeLists.txt && \
    sed -i -e 's/# add_subdirectory(benchmark)/add_subdirectory(benchmark)/g' /root/ncnn/CMakeLists.txt
RUN cd ncnn && \
    mkdir build && cd build && \
    cmake ../ && make -j6
ADD model-y1-test2 /root/model-y1-test2
RUN ./ncnn/build/tools/mxnet/mxnet2ncnn /root/model-y1-test2/model-symbol.json /root/model-y1-test2/model-0000.params
ADD mobilenet_ssd_voc_ncnn/mobilenet_ssd_voc_ncnn.bin /root/mobilenet_ssd_voc_ncnn.bin
ADD mobilenet_ssd_voc_ncnn/mobilenet_ssd_voc_ncnn.param /root/mobilenet_ssd_voc_ncnn.param
ADD ssdmobilenet.cpp /root/ncnn/examples/ssd/ssdmobilenet.cpp
RUN cd ncnn/build && make
ADD imgs /root/imgs
ADD run_test.sh /root/run_test.sh
RUN /root/run_test.sh
