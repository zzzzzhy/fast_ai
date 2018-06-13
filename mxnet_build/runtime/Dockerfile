#FROM arm64v8/ubuntu:16.04
FROM solderzzc/rocketchat:mxnet_save

RUN  apt-get update -y && \
     apt-get install -y \
       libopenblas-base python-setuptools python-pip libgfortran3  python-numpy python-scipy wget \
       python-opencv 
RUN pip install --upgrade pip

RUN pip install /root/mxnet-1.2.0-py2.py3-none-any.whl && rm /root/mxnet-1.2.0-py2.py3-none-any.whl

RUN ln -s /usr/lib/libmxnet.so /usr/local/lib/python2.7/dist-packages/mxnet/
ADD test.py /test.py
RUN python /test.py

