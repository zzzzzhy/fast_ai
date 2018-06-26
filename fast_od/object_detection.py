import nnvm
import numpy as np
import tvm
import os
import time
import convert
import nnvm.testing.darknet

from tvm.contrib import rpc, util, graph_runtime
from nnvm.testing.darknet import __darknetffi__
from cffi import FFI
from ctypes import *

model_name = 'od'
test_image = 'dog.jpg'
dtype = 'float32'
batch_size = 1
cfg_name = model_name + '.cfg'
darknet_lib = 'libruntime.so'
#do the detection and bring up the bounding boxes
thresh = 0.24
hier_thresh = 0.5

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

lib = tvm.module.load('/root/od.tar')
graph = open("/root/od").read()
params = bytearray(open("/root/od.params", "rb").read())
m = graph_runtime.create(graph, lib, ctx)
m.load_params(params)
# get outputs
out_shape = (net.outputs,)

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

def detect(test_image):
    global net
    global darknet_lib
    global out_shape
    global region_layer
    global thresh
    global names

    im_w, im_h, data = get_data(net,test_image,darknet_lib)
    m.set_input('data', tvm.nd.array(data.astype(dtype)))
    m.run()
    tvm_out = m.get_output(0, tvm.nd.empty(out_shape, dtype)).asnumpy()
    result = convert.calc_result(im_w, im_h, tvm_out)

    return result

    probs= []
    boxes = []
    boxes, probs = nnvm.testing.yolo2_detection.get_region_boxes(region_layer, im_w, im_h, net.w, net.h,
                           thresh, probs, boxes, 1, tvm_out)

    boxes, probs = nnvm.testing.yolo2_detection.do_nms_sort(boxes, probs,
                           region_layer.w*region_layer.h*region_layer.n, region_layer.classes, 0.3)

    draw_detections(region_layer.w*region_layer.h*region_layer.n,
        thresh, boxes, probs, names, region_layer.classes)
