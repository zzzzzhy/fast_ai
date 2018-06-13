
# ARMNN Way to run Tensorflow/Caffe model on Mali GPU
The issue to tracking Facenet model convert error: https://github.com/ARM-software/armnn/issues/15

## Build
```
./build.sh
```
## Run
```
./run.sh
```
## Tag image
```
docker tag armnn_aarch64:latest solderzzc/rocketchat:armnn_aarch64
```
## Push Base Image
```
docker push solderzzc/rocketchat:armnn_aarch64
```
