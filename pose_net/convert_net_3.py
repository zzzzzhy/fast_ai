# some standard imports
import os
import nnvm
import tvm
import numpy as np
import onnx

from nnvm import frontend

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
onnx_file = 'model.onnx'

onnx_model = onnx.load(onnx_file)
onnx_sym, params = nnvm.frontend.from_onnx(onnx_model)

convert(onnx_sym, params, (1, 3, 112, 112) ,'float32','llvm --system-lib -target=aarch64-linux-gnu -mattr=+neon')
