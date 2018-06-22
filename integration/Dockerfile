FROM arm64v8/debian:stretch

RUN apt-get update && apt-get install -y libopenblas-base git \
    opencl-headers python-setuptools python-pip libgfortran3 \
    python-numpy python-scipy wget \
    python-matplotlib python-h5py python-requests python-psutil \
    unzip python-sklearn curl python-pil \
    python-opencv

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py 

RUN wget http://releases.llvm.org/6.0.0/clang+llvm-6.0.0-aarch64-linux-gnu.tar.xz
RUN tar -xvf clang+llvm-6.0.0-aarch64-linux-gnu.tar.xz --strip 1 && rm clang+llvm-6.0.0-aarch64-linux-gnu.tar.xz

ADD export/install.tgz /root/install.tgz
RUN tar -xvf /root/install.tgz  --strip 1 && ls -alh /usr/include && rm /root/install.tgz

ADD export/wheels /root/wheels

RUN cd /root/wheels && pip install *.whl

ADD Mali/libmali-midgard-4th-r13p0.so /usr/lib/libmali.so
RUN ln -s /usr/lib/libmali.so /usr/lib/libOpenCL.so && \
    ln -s /usr/lib/libmali.so /usr/lib/libGLES_mali.so

RUN pip install --ignore-installed /tensorflow-1.8.0-cp27-none-linux_aarch64.whl && \
RUN apt-get update && apt-get install -y python-pip python-matplotlib \
    python-h5py python-requests python-psutil git cmake wget unzip \
    python-sklearn curl wget python-pil \
    zlib1g-dev libjpeg-dev libtiff5-dev && \
    curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py && \
    pip install --ignore-installed /tensorflow-1.8.0-cp27-none-linux_aarch64.whl && \
    pip install --ignore-installed -r /root/requirements.txt && rm /root/requirements.txt && \
    python /test.py && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    rm -rf /root/.cache/pip/ && rm /tensorflow-1.8.0-cp27-none-linux_aarch64.whl && rm /test.py
ADD assets/deploy.py /deploy.py
RUN python /deploy.py
