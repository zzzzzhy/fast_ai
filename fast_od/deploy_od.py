import nnvm
#import matplotlib.pyplot as plt
import numpy as np
import tvm
import os
import time
import convert
import nnvm.testing.darknet

from tvm.contrib import rpc, util, graph_runtime
from tvm.contrib.download import download
from nnvm.testing.darknet import __darknetffi__
from cffi import FFI
from ctypes import *

model_name = 'od'
test_image = 'dog.jpg'
dtype = 'float32'
batch_size = 1
cfg_name = model_name + '.cfg'
darknet_lib = 'libruntime.so'

ctx = tvm.cl(0)
ffi = FFI()

if os.path.isfile('./' + darknet_lib) is False:
  exit(0)

darknet_lib = __darknetffi__.dlopen('./' + darknet_lib)

cfg = "./" + str(cfg_name)
net = darknet_lib.load_network(cfg, ffi.NULL, 0)

region_layer = net.layers[net.n - 1]
print('region layer classes {}'.format(region_layer.classes))
coco_name = 'od.names'

with open(coco_name) as f:
    content = f.readlines()

names = [x.strip() for x in content]

print('names {}'.format(names))

def get_data(net, img_path, LIB):
    start = time.time()
    orig_image = LIB.load_image_color(img_path.encode('utf-8'), 0, 0)
    img_w = orig_image.w
    img_h = orig_image.h
    img = LIB.letterbox_image(orig_image, net.w, net.h)
    LIB.free_image(orig_image)
    done = time.time()
    print('1: Image Load run {}'.format((done - start)))

    dtype = 'float32'
    data = np.empty([img.c, img.h, img.w], dtype)
    start = time.time()
    convert.float32_convert(data,img.data)
    done = time.time()
    print('2: data convert in C {}'.format((done - start)))
    i = 0
    LIB.free_image(img)
    return img_w,img_h, data


# tvm module for compiled functions.
lib = tvm.module.load('/root/od.tar')
# json graph
graph = open("/root/od").read()
# parameters in binary
params = bytearray(open("/root/od.params", "rb").read())

m = graph_runtime.create(graph, lib, ctx)

step_start = time.time()
im_w, im_h, data = get_data(net,test_image,darknet_lib)
step_done = time.time()
print('Lib load image cost {}'.format((step_done - step_start)))

m.load_params(params)
m.set_input('data', tvm.nd.array(data.astype(dtype)))
print("Running the test image...")
start = time.time()
m.run()
done = time.time()
print('first run {}'.format((done - start)/1))

out_shape = (net.outputs,)
tvm_out = m.get_output(0, tvm.nd.empty(out_shape, dtype)).asnumpy()
print('tvm_out shape is {}'.format(tvm_out.shape))
convert.calc_result(im_w, im_h, tvm_out)

print("Running the test image...")
start = time.time()
for i in range(1000):
  step_start = time.time()
  _, _, data = get_data(net,test_image,darknet_lib)
  step_done = time.time()
  print('Lib load image cost {}'.format((step_done - step_start)))

  step_start = time.time()
  m.set_input('data', tvm.nd.array(data.astype(dtype)))
  m.run()
  tvm_out = m.get_output(0, tvm.nd.empty(out_shape, dtype)).asnumpy()
  convert.calc_result(im_w, im_h, tvm_out)
  step_done = time.time()
  print('step {} cost {}'.format(i,(step_done - step_start)))
done = time.time()
print('everage cost {}'.format((done - start)/1000))
# get outputs
out_shape = (net.outputs,)
tvm_out = m.get_output(0, tvm.nd.empty(out_shape, dtype)).asnumpy()

print('tvm_out shape is {}'.format(tvm_out.shape))

#do the detection and bring up the bounding boxes
thresh = 0.24
hier_thresh = 0.5
probs= []
boxes = []
boxes, probs = nnvm.testing.yolo2_detection.get_region_boxes(region_layer, im_w, im_h, net.w, net.h,
                       thresh, probs, boxes, 1, tvm_out)

boxes, probs = nnvm.testing.yolo2_detection.do_nms_sort(boxes, probs,
                       region_layer.w*region_layer.h*region_layer.n, region_layer.classes, 0.3)

def draw_detections(num, thresh, boxes, probs, names, classes):
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
draw_detections(region_layer.w*region_layer.h*region_layer.n,
    thresh, boxes, probs, names, region_layer.classes)
