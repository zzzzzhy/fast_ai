FROM lambdazhang/raidcdn:tf1.8_runtime_mx
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y git scons cmake wget unzip curl build-essential \
    libprotobuf-dev protobuf-compiler libopencv-dev python-pip python-setuptools python-dev
WORKDIR /root

ADD ./mtcnn_ncnn_neon /root/mtcnn
WORKDIR /root/mtcnn
RUN pip install wheel && \
    python setup.py sdist bdist_wheel
#RUN ../build/mtcnn/mtcnn ./1_854x480.jpg
