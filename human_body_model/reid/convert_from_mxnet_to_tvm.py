# some standard imports
import mxnet as mx
import nnvm
import tvm
import numpy as np

def get_feature(self, aligned):
  #face_img is bgr image
  #print(nimg.shape)
  input_blob = np.expand_dims(aligned, axis=0)
  data = mx.nd.array(input_blob)
  db = mx.io.DataBatch(data=(data,))

def convert(net, params, data_shape, dtype, target_host):
    # compile
    opt_level = 2 if dtype == 'float32' else 1
    with nnvm.compiler.build_config(opt_level=opt_level):
        graph, lib, params = nnvm.compiler.build(
            net, tvm.target.mali(), shape={"data": data_shape}, params=params,
            dtype=dtype, target_host=target_host)

    lib.export_library('./reid.tar')

    with open("./reid", "w") as fo:
        fo.write(graph.json())
    with open("./reid.params", "wb") as fo:
        fo.write(nnvm.compiler.save_param_dict(params))

net, params, auxs = mx.model.load_checkpoint('resnet50_v1', 0)
nnvm_net, nnvm_params = nnvm.frontend.from_mxnet(net, params, auxs)

convert(nnvm_net, nnvm_params, (1, 3, 384, 128) ,'float32','llvm -target=arm64-linux-android -mattr=+neon')
