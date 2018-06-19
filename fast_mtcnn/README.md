
## MTCNN NCNN
Just run `docker build .` to test mtcnn on NCNN.

```
>= 0.6s
```

## MTCNN GPU C++

```
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
