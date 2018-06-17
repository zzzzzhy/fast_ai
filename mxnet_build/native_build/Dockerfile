FROM arm64v8/ubuntu:18.04

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y build-essential git libatlas-base-dev  libblas-dev libopencv-dev \
    python-pip python-dev python-numpy python-scipy scons

RUN pip install --upgrade pip

WORKDIR /build/

RUN apt-get install -y wget

RUN wget ftp://ftp.openailab.net/tools/package/AID-tools.tar.gz && \
  git clone --recursive https://github.com/OAID/MXNet-HRT.git && \
  tar -xvf AID-tools.tar.gz -C /usr/local && \
  /usr/local/AID/gen-pkg-config-pc.sh /usr/local/AID

ADD Mali/libmali-midgard-4th-r13p0.so /usr/lib/libmali.so
RUN ln -s /usr/lib/libmali.so /usr/lib/libOpenCL.so && \
    ln -s /usr/lib/libmali.so /usr/lib/libGLES_mali.so && \
    apt-get install -y libopenblas-dev

WORKDIR  /build/MXNet-HRT
ADD ./config.mk.acl /build/MXNet-HRT
RUN make -j3
RUN make install

WORKDIR /build/MXNet-HRT 
#RUN cd python && python setup.py install && \
#    apt-get install -y python-opencv python-sklearn python-skimage && \
#    rm -rf /build/MXNet-HRT

WORKDIR /root

#RUN cd example/image-classification/predict-cpp && USE_ACL=1 make -j6
