"""
Compile MXNet Models
====================
**Author**: `Joshua Z. Zhang <https://zhreshold.github.io/>`_

This article is an introductory tutorial to deploy mxnet models with NNVM.

For us to begin with, mxnet module is required to be installed.

A quick solution is
```
pip install mxnet --user
```
or please refer to offical installation guide.
https://mxnet.incubator.apache.org/versions/master/install/index.html
"""
# some standard imports
import mxnet as mx
import numpy as np

######################################################################
# Download Resnet18 model from Gluon Model Zoo
# ---------------------------------------------
# In this section, we download a pretrained imagenet model and classify an image.
from mxnet.gluon.model_zoo.vision import get_model
from mxnet.gluon.utils import download
from PIL import Image
from pickle import dump, load
#from matplotlib import pyplot as plt
block = get_model('resnet18_v1', pretrained=True)
del get_model
img_name = 'cat.jpg'
synset_url = ''.join(['https://gist.githubusercontent.com/zhreshold/',
                      '4d0b62f3d01426887599d4f7ede23ee5/raw/',
                      '596b27d23537e5a1b5751d2b0481ef172f58b539/',
                      'imagenet1000_clsid_to_human.txt'])
synset_name = 'synset.txt'
download('https://github.com/dmlc/mxnet.js/blob/master/data/cat.png?raw=true', img_name)
download(synset_url, synset_name)
del download

with open(synset_name) as f:
    synset = eval(f.read())
image = Image.open(img_name).resize((224, 224))

def transform_image(image):
    image = np.array(image) - np.array([123., 117., 104.])
    image /= np.array([58.395, 57.12, 57.375])
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :]
    return image

x = transform_image(image)
print('x', x.shape)

######################################################################
# Use MXNet symbol with pretrained weights
# ----------------------------------------
# MXNet often use `arg_prams` and `aux_params` to store network parameters
# separately, here we show how to use these weights with existing API
def block2symbol(block):
    data = mx.sym.Variable('data')
    sym = block(data)
    args = {}
    auxs = {}
    for k, v in block.collect_params().items():
        args[k] = mx.nd.array(v.data().asnumpy())
    return sym, args, auxs
mx_sym, args, auxs = block2symbol(block)
# usually we would save/load it as checkpoint
mx.model.save_checkpoint('resnet18_v1', 0, mx_sym, args, auxs)
# there are 'resnet18_v1-0000.params' and 'resnet18_v1-symbol.json' on disk

######################################################################
# for a normal mxnet model, we start from here
mx_sym, args, auxs = mx.model.load_checkpoint('resnet18_v1', 0)

to_pack = [ mx_sym, args, auxs ]

with open('data.pkl', 'wb') as f:
    dump(mx_sym, f ,2)

with open('data.pkl', 'rb') as f:
    mx_sym = load(f)
del x
del transform_image
del synset_url
del synset_name
del synset
del np
del mx_sym
del mx
del load
del img_name
del image
del f
del dump
del block2symbol
del block
del auxs
del args

del Image
del to_pack
del __package__
del __name__
del __file__
del __doc__
del __builtins__

print(dir())

from nnvm.frontend import from_mxnet

#nnvm_sym, nnvm_params = nnvm.frontend.from_mxnet(mx_sym, args, auxs)
