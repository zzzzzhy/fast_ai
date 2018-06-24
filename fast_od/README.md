
# Yolo v2 Tiny to TVM

## Build TVM model on X86

```
./build_model.sh
```

## Run on ARM

```
./deploy.sh
```

## Performance

### Image Load, Preprocessing

| Load Method | Code In Commit | Duration |
|:-----------:|:--------------:|:--------:|
| Python Code | [By Python](https://github.com/solderzzc/fast_ai/blob/020ffef678d266b5ed07ed9bdad5f2864fade1a2/fast_od/deploy_od.py#L104) | 8s|
| C++ Load    | [C++ Load](https://github.com/solderzzc/fast_ai/blob/8096c9ca2b4a8efc0eb93f11955c9f81684c8e29/fast_od/deploy_od.py#L51) | 0.3 |
| Python Convert | [Python Convert](https://github.com/solderzzc/fast_ai/blob/8096c9ca2b4a8efc0eb93f11955c9f81684c8e29/fast_od/deploy_od.py#L59) | 0.75s|


Yolov2 Tiny 0.3s, image load 8s (Python code)
