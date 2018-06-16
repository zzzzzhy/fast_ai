#from nnvm.frontend import from_mxnet

from mxnet.gluon.model_zoo.vision import get_model
block = get_model('resnet18_v1', pretrained=True)

import nnvm
