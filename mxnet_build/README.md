# Build latest MXNet from source code for CPU
## Run cross build on powerful x86 server
```
./cross_build.sh
```

## Run final build on Embedded board
```
./final_build.sh
```

# Build 0.11.0 MXNet from source code for GPU (ARM Mali)

## Run on Target
```
cd native_build
docker build .
```
