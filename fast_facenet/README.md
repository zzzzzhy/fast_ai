
# CPU Version (ARM ASM)
## Run fast facenet on Embedded board
`docker build .`
# GPU Version (ARM Mali)
## Compile Fast Network into Runtime
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


## Compile High Acc Network into Runtime
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

## Performance On GPU (ARM Mali)

| Model        | Time Cost Per Frame | Memory Usage | CPU Usage |
|:-----------:|:-------------------:| :-----------:|:---------:|
| High Speed Model | 0.054s | 150 MB | 70% |
| High Acc Model | 0.580s | 750 MB | 7% |
