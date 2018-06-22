FROM lambdazhang/lambdazhang:tvm_06222018

ADD assets/tensorflow-1.8.0-cp27-none-linux_aarch64.whl /tensorflow-1.8.0-cp27-none-linux_aarch64.whl
ADD assets/requirements.txt /root/requirements.txt
ADD assets/test.py /test.py
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
