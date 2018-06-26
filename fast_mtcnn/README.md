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

### 480P

| Light Mode | Num Threads | CPU Usage | Memory Usage| Min Time | Max Time | Avg Time |
|:----------:|:-----------:|:---------:|:-----------:|:--------:|:--------:|:--------:|
| No | 1 | 100% | 57.2 MB | 40 ms | 97 ms| 41 ms|
| No | 2 | 165% | 58.5 MB | 29 ms | 137 ms| 32 ms|
| Yes | 2 | 165% | 57.9 MB | 27 ms | 53 ms| 28 ms|

### 1080P

| Light Mode | Num Threads | CPU Usage | Memory Usage| Min Time | Max Time | Avg Time |
|:----------:|:-----------:|:---------:|:-----------:|:--------:|:--------:|:--------:|
| No | 1 | 100% |  MB |187 ms | 206ms| 190 ms|
| No | 2 | 165% | 102 MB |138 ms | 239ms| 142 ms|
| Yes | 2 | 162% | 102 MB | 129 ms | 207 ms | 134 ms|

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
