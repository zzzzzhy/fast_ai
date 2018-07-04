FROM solderzzc/rocketchat:armnn_aarch64
RUN apt-get install -y python-pip python-dev && \
    pip install pytest
ADD CMakeLists.txt ${ARMNN_DIR}/CMakeLists.txt
ADD pybind11 ${ARMNN_DIR}/pybind11
ADD pypose ${ARMNN_DIR}/pypose
ADD runtime.sh /root/runtime.sh
ADD setup.py ${ARMNN_DIR}/
RUN apt-get install -y python-opencv
RUN pip install setuptools wheel
#RUN rm *.txt && cmake -DBUILD_UNIT_TESTS=OFF -DARMCOMPUTE_ROOT=$CL_DIR -DARMCOMPUTE_BUILD_DIR=$CL_DIR/build -DTF_GENERATED_SOURCES=/root/v$TF_VER/protobuf -DBUILD_TF_PARSER=ON -DBUILD_CAFFE_PARSER=ON -DPROTOBUF_ROOT=/usr -DBUILD_TESTS=OFF -DTHIRD_PARTY_INCLUDE_DIRS="/root/stb-cmake" -DARMCOMPUTECL=ON -DARMCOMPUTENEON=ON -DCAFFE_GENERATED_SOURCES=/root/caffe/src ../
WORKDIR ${ARMNN_DIR}/
RUN python setup.py sdist bdist_wheel
