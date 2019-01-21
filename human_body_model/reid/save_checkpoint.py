from __future__ import print_function, division

import mxnet as mx
import numpy as np
from mxnet import gluon, image, nd
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms

from networks import resnet18, resnet34, resnet50


def load_network(network, ctx):
    network.load_parameters('params/resnet50.params', ctx=ctx, allow_missing=True, ignore_extra=True)
    return network


context = mx.cpu()

# Load Collected data Trained model
model = resnet50(ctx=context, pretrained=False)
model = load_network(model, context)
#model.hybridize() 
#model.export('Gluon_FashionMNIST')
#print(model)

def block2symbol(block):
    data = mx.sym.Variable('data')
    sym = block(data)
    args = {}
    auxs = {}
    for k, v in block.collect_params().items():
        args[k] = mx.nd.array(v.data().asnumpy())
    return sym, args, auxs


mx_sym, args, auxs = block2symbol(model)
# usually we would save/load it as checkpoint
mx.model.save_checkpoint('resnet50_v1', 0, mx_sym, args, auxs)
# there are 'resnet50_v1-0000.params' and 'resnet50_v1-symbol.json' on disk

