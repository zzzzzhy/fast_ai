

# Build And run Docker Image

```
docker build -t solderzzc/rockchat:tvm .
./run.sh solderzzc/rockchat:tvm
```

# Set Test Environment in Docker container
```
echo performance > /sys/class/misc/mali0/device/devfreq/ff9a0000.gpu/governor
cd /root/benchmark
```
This can make the environment more stable.

**Note**: You need more than 2.5GB of memory to run the following test.
Otherwise, you must skip the test of vgg16 by replacing `--model all` with `--model resnet18` or `--model mobilenet`
in the commond.

## Run Test for TVM/NNVM
```
python -m tvm.exec.rpc_server --host 0.0.0.0 --port=9090 &
```

``` bash
python mali_imagenet_bench.py --target-host 'llvm -target=aarch64-linux-gnu -mattr=+neon' --host 127.0.0.1 --port 9090 --model all
```

## Run Test for MXNet + Openblas
This test is executed locally on your device. So you need install the mxnet with openblas on your device first.

``` bash
python mxnet_test.py --model all
```

## Run Test for Arm Compute Library
```
/build/ComputeLibrary/build/acl_test all
cat result-acl.txt
```
results are recored in `result-acl.txt`

**Note** Some testcases (e.g. resnet) are missing because Arm Compute Library currently (v17.12) does not 
support skip connection in its graph runtime. Also some testcases are too slow so that be skipped.

# Performance Test Result on DEEPEYE Box

## MXNet + Openblas

```
backend: MXNet+OpenBLAS	model: resnet18	dtype: float32	cost:0.4227
backend: MXNet+OpenBLAS	model: mobilenet	dtype: float32	cost:0.2499
backend: MXNet+OpenBLAS	model: vgg16	dtype: float32	cost:5.2321
```

## ACL

```
backend: ARMComputeLib-mali	model: vgg16	conv_method: gemm	dtype: float32	cost: 1.81407
backend: ARMComputeLib-mali	model: vgg16	conv_method: gemm	dtype: float16	cost: 1.07664
backend: ARMComputeLib-mali	model: vgg16	conv_method: direct	dtype: float32	cost: 5.41556
backend: ARMComputeLib-mali	model: vgg16	conv_method: direct	dtype: float16	cost: 1.67393
backend: ARMComputeLib-mali	model: mobilenet	conv_method: gemm	dtype: float32	cost: 0.205742
backend: ARMComputeLib-mali	model: mobilenet	conv_method: direct	dtype: float32	cost: 0.207053
```

## TVM/NNVM

```
backend: TVM-mali	model: vgg16	dtype: float32	cost:1.0510
backend: TVM-mali	model: vgg16	dtype: float16	cost:0.6011
backend: TVM-mali	model: resnet18	dtype: float32	cost:0.1999
backend: TVM-mali	model: resnet18	dtype: float16	cost:0.1213
backend: TVM-mali	model: mobilenet	dtype: float32	cost:0.0903
backend: TVM-mali	model: mobilenet	dtype: float16	cost:0.0585
```

# Use Remote RPC to run model on board, build on host(x86)

## On Board to run rpc server
```
./run.sh solderzzc/rocketchat:tvm_06142018 /root/rpc_server.sh
```

