import time
import argparse
import numpy as np
import tvm
import nnvm.compiler
import nnvm.testing
from tvm.contrib import util, rpc
from tvm.contrib import graph_runtime as runtime

def build_module(net, params, dtype):
    # compile
    opt_level = 2 if dtype == 'float32' else 1
    with nnvm.compiler.build_config(opt_level=opt_level):
        graph, lib, params = nnvm.compiler.build(
            net, tvm.target.mali(), shape={"data": data_shape}, params=params,
            dtype=dtype, target_host='llvm -target=aarch64-linux-gnu -mattr=+neon')
    return graph, lib, params 
