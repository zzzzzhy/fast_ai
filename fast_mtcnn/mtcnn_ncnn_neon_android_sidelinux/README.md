# MTCNN NCNN NEON Implementation
-----------------------------

## Install wheel

```
LDFLAGS=" -lm -lcompiler_rt" pip2 install wheel
```

## BUILD
```
python2 setup.py sdist bdist_wheel
```

## RUN

```
LD_LIBRARY_PATH=/data/data/com.termux/files/usr/lib64:/data/data/com.termux/files/usr/lib ./test_video
```
