# Benchmark Result
## Performance On GPU (ARM Mali)

| Model        | Time Cost Per Frame | Memory Usage | CPU Usage |
|:-----------:|:-------------------:| :-----------:|:---------:|
| High Speed Model | 0.054s | 150 MB | 70% |
| High Acc Model | 0.580s | 750 MB | 7% |

## Performance On CPU
| Model        | Time Cost Per Frame | Memory Usage | CPU Usage |
|:-----------:|:-------------------:| :-----------:|:---------:|
| High Speed Model | 0.20s | 191 MB | 600% |
| High Acc Model | 1.60s | 758 MB | 600% |

![wechatimg547](https://user-images.githubusercontent.com/3085564/41512352-0e1e327c-723c-11e8-9e47-fb6915c406c0.jpeg)

# How to run ARM Mali benchmark
## Compile High Speed Model into Runtime
### Compiled model will be placed in foder [gpu_deploy/export](gpu_deploy/export)
```
cd gpu_deploy
./model_compile.sh
```
### Run On Target
```
cd gpu_deploy
./deploy.sh
```
![Run On Target 1](https://user-images.githubusercontent.com/3085564/41492229-975363f4-70b2-11e8-89fa-1c57362ce378.png)
## Compile High Accurate Model into Runtime
```
cd gpu_deploy
./model_compile2.sh
```
### Run On Target
```
cd gpu_deploy
bash deploy2.sh
```
![Run On Target 2](https://user-images.githubusercontent.com/3085564/41492257-c2b19f52-70b2-11e8-89ec-818b45ed9185.png)

# How to run CPU benchmark
## CPU version High SPEED
```
cd fast_facenet
git checkout f7c89abbf5dbc1f7b64c4b069bfcb025c2fac452
./run.sh solderzzc/rocketchat:mxnet
cd ~/test/deploy
apt-get install -y python-opencv python-sklearn python-skimage
pip install -r requirements.txt
python test_feature.py --model ../model-y1-test2/model,0
```

## CPU version High Accurate
```
cd fast_facenet
git checkout f7c89abbf5dbc1f7b64c4b069bfcb025c2fac452
./run.sh solderzzc/rocketchat:mxnet
cd ~/test/deploy
apt-get install -y python-opencv python-sklearn python-skimage
pip install -r requirements.txt
python test_feature.py --model ../model-r50-am-lfw/model,0
```

# CPU Version (ARM ASM)
## Run fast facenet on Embedded board
`docker build .`
