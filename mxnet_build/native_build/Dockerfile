FROM arm64v8/ubuntu:16.04

RUN  apt-get update -y && \
     apt-get install -y \
       build-essential git libatlas-base-dev libopencv-dev python-opencv \
       libcurl4-openssl-dev libgtest-dev cmake wget unzip liblapack-dev clang \  
       python-dev python-setuptools python-pip libgfortran3  python-numpy python-scipy wget
RUN pip install --upgrade pip

WORKDIR /build
RUN git clone --recursive https://github.com/apache/incubator-mxnet -b v1.2.0
#ADD ./config.mk /build/incubator-mxnet/config.mk
RUN cd incubator-mxnet && git checkout 1.2.0 && \
    make USE_OPENCV=1 USE_BLAS=atlas ARCH=aarch64 USE_F16C=0 USE_OPERATOR_TUNING=1 USE_LAPACK=1 USE_OPENMP=1 
    # CC=clang CXX=clang++
RUN cd /build/incubator-mxnet/python && \
    pip install -e .

ADD test.py /test.py
RUN python /test.py

