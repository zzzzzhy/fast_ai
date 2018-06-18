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

# Prebuilt Docker List

| Docker Tag | MXNet Version | With GPU | Verify |
|:----------:|:-------------:|:--------:|:--------:|
|solderzzc/rocketchat:mxnet| 1.2.0 | ❌ | [face feature/detection](https://github.com/solderzzc/fast_ai/tree/master/fast_facenet#how-to-run-cpu-benchmark) |
|solderzzc/rocketchat:mxnet_mali|0.11.0|✅|[face detection](https://github.com/solderzzc/fast_ai/blob/master/fast_facenet/README.md#gpu-version-with-profile)|
|solderzzc/rocketchat:mxnet_mali_no_profile|0.11.0|✅|[face detection](https://github.com/solderzzc/fast_ai/blob/master/fast_facenet/README.md#gpu-version-without-profile)|
