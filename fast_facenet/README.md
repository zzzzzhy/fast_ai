
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
