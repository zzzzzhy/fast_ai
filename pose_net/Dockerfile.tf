FROM solderzzc/rocketchat:tvm_06282018_x86
RUN pip install tensorflow
ADD tensorflow.py /build/tvm/nnvm/python/nnvm/frontend/tensorflow.py
RUN cd /build/tvm/nnvm/python && \
    python setup.py install --user
