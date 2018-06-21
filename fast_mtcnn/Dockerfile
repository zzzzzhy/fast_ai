FROM arm64v8/ubuntu:18.04
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y git scons cmake wget unzip curl build-essential \
    libprotobuf-dev protobuf-compiler libopencv-dev python-pip python-setuptools python-dev
WORKDIR /root
#RUN git clone https://github.com/deepinsight/mtcnn-ncnn mtcnn && \
RUN git clone --recursive https://github.com/oktal/pistache.git && \
    cd pistache && mkdir build && cd build && \
    cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release .. && \
    make -j5 && \
    make install

ADD ./mtcnn-ncnn /root/mtcnn
#RUN cd mtcnn && \
#    mkdir build && cd build && \
#    cmake ../ && make -j6
WORKDIR /root/mtcnn
RUN pip install wheel && \
    python setup.py sdist bdist_wheel
#RUN ../build/mtcnn/mtcnn ./1_854x480.jpg
