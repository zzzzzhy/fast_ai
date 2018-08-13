## Benchmark on X86

### Command
`docker build .` Inspect log to see the benchmark of Models.

### Facenet
2.83 BFLOPs (GFLOPs)

### Mobilenetv2 SSDLite
1.50 BFLOPs (GFLOPs)

### PoseNet(Mobilenet)
3.82 BFLOPs (GFLOPs)

### LapSRN
496.71 BFLOPs (GFLOPs)

## Reference

https://github.com/onnx/tensorflow-onnx/blob/master/tests/run_pretrained_models.yaml


# On ARM RK3399

```
cd fast_tensorflow
./run.sh
```

```
apt-get install opencl-headers
git clone https://github.com/krrishnarraj/clpeak
cd clpeak
mkdir build
cd build
cmake ..
make -j6
```


```
./clpeak 

Platform: ARM Platform
  Device: Mali-T860
    Driver version  : 1.2 (Linux ARM64)
    Compute units   : 4
    Clock frequency : 800 MHz

    Global memory bandwidth (GBPS)
      float   : 3.59
      float2  : 5.77
      float4  : 6.63
      float8  : 5.36
      float16 : 4.79

    Single-precision compute (GFLOPS)
      float   : 25.03
      float2  : 45.18
      float4  : 45.10
      float8  : 41.62
      float16 : 46.38

    half-precision compute (GFLOPS)
      half   : 23.14
      half2  : 49.99
      half4  : 98.90
      half8  : 93.37
      half16 : 93.07

    Double-precision compute (GFLOPS)
      double   : 12.39
      double2  : 3.28
      double4  : 20.89
      double8  : 20.61
      double16 : 20.38

    Integer compute (GIOPS)
      int   : 20.17
      int2  : 49.50
      int4  : 46.82
      int8  : 48.55
      int16 : 41.43

    Transfer bandwidth (GBPS)
      enqueueWriteBuffer         : 4.62
      enqueueReadBuffer          : 2.60
      enqueueMapBuffer(for read) : 509.80
        memcpy from mapped ptr   : 2.54
      enqueueUnmap(after write)  : 4021.50
        memcpy to mapped ptr     : 2.61

    Kernel launch latency : 149.21 us
```
