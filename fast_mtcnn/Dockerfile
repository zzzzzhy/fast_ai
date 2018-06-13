FROM aarch64/debian:stretch
RUN apt-get update && apt-get install -y git scons cmake wget unzip curl build-essential \
    libprotobuf-dev protobuf-compiler libopencv-dev
WORKDIR /root
#RUN git clone https://github.com/deepinsight/mtcnn-ncnn mtcnn && \
ADD ./mtcnn-ncnn /root/mtcnn
RUN cd mtcnn && \
    mkdir build && cd build && \
    cmake ../ && make -j6
