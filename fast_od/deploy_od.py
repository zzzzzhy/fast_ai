import nnvm
import nnvm.testing.darknet
#import matplotlib.pyplot as plt
import numpy as np
import tvm
import os
import time

from ctypes import *

from tvm.contrib.download import download
from nnvm.testing.darknet import __darknetffi__

model_name = 'yolov2-tiny-voc'
test_image = 'dog.jpg'

ctx = tvm.cl(0)

######################################################################
# Prepare cfg and weights file
# ----------------------------
# Pretrained model available https://pjreddie.com/darknet/imagenet/
# Download cfg and weights file first time.

cfg_name = model_name + '.cfg'
weights_name = model_name + '.weights'
cfg_url = 'https://github.com/siju-samuel/darknet/blob/master/cfg/' + \
                    cfg_name + '?raw=true'
weights_url = 'http://pjreddie.com/media/files/' + weights_name + '?raw=true'

download(cfg_url, cfg_name)
download(weights_url, weights_name)

######################################################################
# Download and Load darknet library
#if the file doesnt exist, then exit normally.
darknet_lib = 'libdarknet.so'
if os.path.isfile('./' + darknet_lib) is False:
        exit(0)

darknet_lib = __darknetffi__.dlopen('./' + darknet_lib)

load_image = darknet_lib.load_image_color

step_start = time.time()
#data = nnvm.testing.darknet.load_image(test_image, 416, 416)
image = load_image(test_image, 416, 416)
step_done = time.time()
print(image.data)
a = np.frombuffer(image.data)
print('load image cost {}'.format((step_done - step_start)))

cfg = "./" + str(cfg_name)
weights = "./" + str(weights_name)
net = darknet_lib.load_network(cfg.encode('utf-8'), weights.encode('utf-8'), 0)

dtype = 'float32'
batch_size = 1

from tvm.contrib import rpc, util, graph_runtime

# tvm module for compiled functions.
lib = tvm.module.load('/root/od.tar')
# json graph
graph = open("/root/od").read()
# parameters in binary
params = bytearray(open("/root/od.params", "rb").read())

######################################################################
# Load a test image
# --------------------------------------------------------------------
print("Loading the test image...")

#data = nnvm.testing.darknet.load_image(test_image, 416, 416)

step_start = time.time()
#data = nnvm.testing.darknet.load_image(test_image, 416, 416)
image = load_image(test_image, 416, 416)
step_done = time.time()
print(image.data)
a = np.frombuffer(image.data)
print('load image cost {}'.format((step_done - step_start)))

######################################################################
# Execute on TVM Runtime
# ----------------------
# The process is no different from other examples.
from tvm.contrib import graph_runtime

m = graph_runtime.create(graph, lib, ctx)

m.load_params(params)
# set inputs
m.set_input('data', tvm.nd.array(image.data))
#m.set_input(**params)
# execute
print("Running the test image...")
start = time.time()
m.run()
done = time.time()
print('first run {}'.format((done - start)/1))

step_start = time.time()
#data = nnvm.testing.darknet.load_image(test_image, 416, 416)
data = load_image(test_image, 416, 416)
step_done = time.time()
print('load image cost {}'.format((step_done - step_start)))

print("Running the test image...")
start = time.time()
for i in range(100):
  step_start = time.time()
  m.set_input('data', tvm.nd.array(data.astype(dtype)))
  m.run()
  step_done = time.time()
  print('step {} cost {}'.format(i,(step_done - step_start)))
done = time.time()
print('everage cost {}'.format((done - start)/100))
# get outputs
out_shape = (net.outputs,)
tvm_out = m.get_output(0, tvm.nd.empty(out_shape, dtype)).asnumpy()

#do the detection and bring up the bounding boxes
thresh = 0.24
hier_thresh = 0.5
img = nnvm.testing.darknet.load_image_color(test_image)
_, im_h, im_w = img.shape
probs= []
boxes = []
region_layer = net.layers[net.n - 1]
boxes, probs = nnvm.testing.yolo2_detection.get_region_boxes(region_layer, im_w, im_h, 416, 416,
                       thresh, probs, boxes, 1, tvm_out)

boxes, probs = nnvm.testing.yolo2_detection.do_nms_sort(boxes, probs,
                       region_layer.w*region_layer.h*region_layer.n, region_layer.classes, 0.3)

coco_name = 'od.names'

with open(coco_name) as f:
    content = f.readlines()

names = [x.strip() for x in content]

def draw_detections(im, num, thresh, boxes, probs, names, classes):
    "Draw the markings around the detected region"
    for i in range(num):
        labelstr = []
        category = -1
        for j in range(classes):
            if probs[i][j] > thresh:
                if category == -1:
                    category = j
                labelstr.append(names[j])
                print("{}:{}".format(names[j],probs[i][j]))
draw_detections(img, region_layer.w*region_layer.h*region_layer.n,
    thresh, boxes, probs, names, region_layer.classes)
            #_draw_label(im, top + width, left, label, rgb)
