FROM ubuntu:16.04

RUN dpkg --add-architecture arm64
ADD ./patch.sh /root/patch.sh
ADD sources.list /etc/apt/sources.list
WORKDIR /root
RUN /root/patch.sh && rm /root/patch.sh
RUN apt-get update
RUN apt-get install -y git libpython-all-dev:arm64 opencl-headers openjdk-8-jdk python python-numpy python-pip zlib1g-dev:arm64 wget
RUN wget https://releases.linaro.org/components/toolchain/binaries/6.3-2017.05/aarch64-linux-gnu/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu.tar.xz
RUN tar -xf gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu.tar.xz && \
    mkdir -p $HOME/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu/aarch64-linux-gnu/libc/usr/include/aarch64-linux-gnu && \
    ln -s /usr/include/aarch64-linux-gnu/python2.7/ $HOME/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu/aarch64-linux-gnu/libc/usr/include/aarch64-linux-gnu/

RUN wget https://github.com/bazelbuild/bazel/releases/download/0.11.1/bazel_0.11.1-linux-x86_64.deb && \
    apt install -y ./bazel_0.11.1-linux-x86_64.deb && \
    bazel version
COPY ComputeCpp-CE-0.8.0-Ubuntu.16.04-64bit.tar.gz /root
COPY ComputeCpp-CE-0.8.0-Ubuntu.16.04-ARM64.tar.gz /root
RUN tar -xf ComputeCpp-CE-0.8.0-Ubuntu.16.04-64bit.tar.gz && \
    tar -xf ComputeCpp-CE-0.8.0-Ubuntu.16.04-ARM64.tar.gz && \
    cp ComputeCpp-CE-0.8.0-Ubuntu-16.04-x86_64/bin/compute++ ComputeCpp-CE-0.8.0-Ubuntu-16.04-ARM_64/bin
RUN git clone http://github.com/codeplaysoftware/tensorflow && \
    cd tensorflow && \
    git checkout 4f28eaa6 && \
    export TMPDIR=/root/tensorflow_temp && \
    mkdir -p $TMPDIR && \
    export PYTHON_BIN_PATH=/usr/bin/python && \
    export USE_DEFAULT_PYTHON_LIB_PATH=1 && \
    export CC_OPT_FLAGS="-march=armv8-a" && \
    export TF_NEED_MKL=0 && \
    export TF_NEED_JEMALLOC=1 && \
    export TF_NEED_GCP=0 && \
    export TF_NEED_HDFS=0 && \
    export TF_ENABLE_XLA=0 && \
    export TF_NEED_OPENCL_SYCL=1 && \
    export TF_NEED_COMPUTECPP=1 && \
    export TF_USE_DOUBLE_SYCL=0 && \
    export TF_USE_HALF_SYCL=0 && \
    export COMPUTECPP_TOOLKIT_PATH=$HOME/ComputeCpp-CE-0.8.0-Ubuntu-16.04-ARM_64 && \
    export TF_NEED_CUDA=0 && \
    export TF_NEED_VERBS=0 && \
    export TF_NEED_MPI=0 && \
    export TF_NEED_GDR=0 && \
    export TF_NEED_S3=0 && \
    export TF_NEED_KAFKA=0 && \
    export TF_SET_ANDROID_WORKSPACE=0 && \
    export TF_SYCL_CROSS_TOOLCHAIN=$HOME/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu && \
    export TF_SYCL_CROSS_TOOLCHAIN_NAME=aarch64-linux-gnu && \
    export TF_SYCL_BITCODE_TARGET=spirv64 && \
    ./configure && \
    bazel --output_user_root=$TMPDIR build --config=sycl_arm --config=opt --verbose_failures //tensorflow/tools/pip_package:build_pip_package && \
    bazel-bin/tensorflow/tools/pip_package/build_pip_package $TMPDIR && \
    mv $TMPDIR/tensorflow-1.6.0rc0-cp27-cp27mu-linux_x86_64.whl $TMPDIR/tensorflow-1.6.0rc0-cp27-cp27mu-linux_aarch64.whl



#bazel --output_user_root=$TMPDIR build --config=sycl_arm -c opt --verbose_failures --copt=-DNO_LOCAL_MEM--copt=-DEIGEN_DONT_VECTORIZE_SYCL //tensorflow/tools/pip_package:build_pip_package
