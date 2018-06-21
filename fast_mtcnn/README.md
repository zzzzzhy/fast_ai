## Build Face Detection Python Module
```
./build_face_detection_wheel.sh
```

Built whl will be placed under ./export/

## Build Wheel
```
apt-get install python-pip python-setuptools
pip install wheel
python setup.py sdist bdist_wheel
```
## MTCNN NCNN [Commit 4e29bc](https://github.com/solderzzc/fast_ai/commit/4e29bc36e6bdcb0c9059a5362e653362fe7f9344)
Just run `docker build .` to test mtcnn on NCNN.

Docker image: solderzzc/rocketchat:mtcnn_ncnn

| Runtime Lib | Run On | Thread Num |Computation Cost | Speed |
| :---------: |:-----: |:----------:|:---------------:|:-----:|
| NCNN | NEON | 1.5 core | 8 | 233ms |
| NCNN | NEON | 1 core   | 1 | 135ms |
## MTCNN GPU C++

```
git checkout 4e29bc36e6bdcb0c9059a5362e653362fe7f9344
cd fast_mtcnn
./run.sh solderzzc/rocketchat:mxnet_mali_no_profile
apt-get install -y python-opencv python-sklearn python-skimage cmake
pip install setuptools
cd /root/test/mtcnn-gpu

pip install .
cp ./mtcnn-gpu/lib/libmtcnn.so /usr/lib/
cp /usr/local/lib/python2.7/dist-packages/mxnet-0.10.1-py2.7.egg/mxnet/libmxnet.so /usr/lib/

cd mtcnn-gpu/tests/

python test.py
```
