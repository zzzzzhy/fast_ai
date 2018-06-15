
# CPU Version (ARM ASM)
## Run fast facenet on Embedded board

`docker build .`

# GPU Version (ARM Mali)
## Compile Network into Runtime

Compiled model will be placed in foder [gpu_deploy/export](gpu_deploy/export)
```
cd gpu_deploy
./model_compile.sh
```
