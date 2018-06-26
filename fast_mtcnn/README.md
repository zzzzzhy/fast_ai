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
## MTCNN NCNN NEON [Commit b5d653bd](https://github.com/solderzzc/fast_ai/tree/b5d653bd1107d1167a81fa858513cb1883509b37)

| Light Mode | Num Threads | CPU Usage | Min Time | Max Time | Avg Time |
|:----------:|:-----------:|:---------:|:--------:|:--------:|:--------:|
| ❌ | 1 | 100% | 187ms | 206ms| 190 ms|


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
