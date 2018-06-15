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

    lib.export_library('./net1.tar')

    with open("./net1", "w") as fo:
        fo.write(graph.json())
    with open("./net1.params", "wb") as fo:
        fo.write(nnvm.compiler.save_param_dict(params))

net, params, auxs = mx.model.load_checkpoint('./model-y1-test2/model', 0)
nnvm_net, nnvm_params = nnvm.frontend.from_mxnet(net, params, auxs)

convert(nnvm_net, nnvm_params, (1, 3, 112, 112) ,'float32','llvm --system-lib -target=aarch64-linux-gnu -mattr=+neon')
