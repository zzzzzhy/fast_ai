FROM solderzzc/rocketchat:armnn_aarch64
RUN apt-get install -y python-pip python-dev && \
    pip install pytest
RUN apt-get install -y python-opencv
RUN pip install setuptools wheel

ADD CMakeLists.txt ${ARMNN_DIR}/CMakeLists.txt
ADD pybind11 ${ARMNN_DIR}/pybind11
ADD pypose ${ARMNN_DIR}/pypose
ADD runtime.sh /root/runtime.sh
ADD setup.py ${ARMNN_DIR}/
WORKDIR ${ARMNN_DIR}/
RUN python setup.py sdist bdist_wheel
