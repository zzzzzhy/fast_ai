## How to freeze graph

```
python freeze_graph.py ./20170512-110547 freezed.pb
python remove_phase_train.py
```

## How to convery tensorflow freezed graph model into ONNX

```
python test.py
```

## How to test on ARMNN
```
cp freezed_clean.pb ../GPU_ARM/app/app/model/freezed.pb
cd ../GPU_ARM
docker build -f Dockerfile.x86 .
```

## How to show graph in tensorboard

```
python -m tensorflow.python.tools.import_pb_to_tensorboard --model_dir ./freezed.pb --log_dir ./log/
tensorboard --logdir=./log
```
