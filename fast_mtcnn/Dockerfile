FROM arm64v8/ubuntu:18.04
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y git scons cmake wget unzip curl build-essential \
    libprotobuf-dev protobuf-compiler libopencv-dev
WORKDIR /root
#RUN git clone https://github.com/deepinsight/mtcnn-ncnn mtcnn && \
ADD ./mtcnn-ncnn /root/mtcnn
RUN cd mtcnn && \
    mkdir build && cd build && \
    cmake ../ && make -j6
WORKDIR /root/mtcnn/mtcnn
RUN ../build/mtcnn/mtcnn ./1_854x480.jpg
