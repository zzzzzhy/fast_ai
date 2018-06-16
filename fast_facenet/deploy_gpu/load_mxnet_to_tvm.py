
#from mxnet.model import load_checkpoint as load_mxnet_checkpoint

import time
import numpy as np
import tvm.target
import nnvm.compiler

#import mxnet as mx

def build_module(net, params, dtype):
    # compile
    data_shape = (1, 3, 112, 112)
    opt_level = 2 if dtype == 'float32' else 1
    with nnvm.compiler.build_config(opt_level=opt_level):
        graph, lib, params = nnvm.compiler.build(
            net, tvm.target.mali(), shape={"data": data_shape}, params=params,
            dtype=dtype, target_host='llvm -target=aarch64-linux-gnu -mattr=+neon')
    return graph, lib, params

def get_model(model_str, layer):
  _vec = model_str.split(',')
  assert len(_vec)==2
  prefix = _vec[0]
  epoch = int(_vec[1])
  print('loading',prefix, epoch)
  sym, arg_params, aux_params = load_mxnet_checkpoint(prefix, epoch)
  all_layers = sym.get_internals()
  sym = all_layers[layer+'_output']

  print('code to compile network')
  build_module(sym, params, 'float32')
  print('code to compile network, end')

  #model.set_params(arg_params, aux_params)
  return model

get_model('../model-y1-test2/model,0','fc1')
