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
| Min Face Size| Detected Number|
|:------------:|:--------------:|
| 40x40 | 8 |

| Light Mode | Num Threads | CPU Usage | Memory Usage| Min Time | Max Time | Avg Time |
|:----------:|:-----------:|:---------:|:-----------:|:--------:|:--------:|:--------:|
| No | 1 | 100% | 60.5 MB | 142 ms | 188 ms| 146 ms|
| No | 2 | 182% | 60.5 MB | 103 ms | 201 ms| 113 ms|
| Yes | 1 | 100% | 57.5 MB | 138 ms | 195 ms| 141 ms|
| Yes | 2 | 182% | 57.8 MB | 96 ms | 221 ms| 100 ms|

### 1080P
| Min Face Size| Detected Number|
|:------------:|:--------------:|
| 80x80 | 8 |

| Light Mode | Num Threads | CPU Usage | Memory Usage| Min Time | Max Time | Avg Time |
|:----------:|:-----------:|:---------:|:-----------:|:--------:|:--------:|:--------:|
| No | 1 | 100% | 102 MB |188 ms | 203 ms| 192 ms|
| No | 2 | 160% | 102 MB |138 ms | 249 ms| 147 ms|
| Yes | 1 | 100% | 101 MB | 179 ms | 190 ms | 183 ms|
| Yes | 2 | 162% | 102 MB | 130 ms | 201 ms | 137 ms|

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
