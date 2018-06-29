import cv2
import tvm
import numpy as np
import time
from scipy import misc
import os

from tvm.contrib import rpc, util, graph_runtime


def transform_image(image):
    image = np.array(image) - np.array([123., 117., 104.])
    image /= np.array([58.395, 57.12, 57.375])
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :]
    return image

# tvm module for compiled functions.
loaded_lib = tvm.module.load('/root/net1.tar')
# json graph
loaded_json = open("/root/net1").read()
# parameters in binary
loaded_params = bytearray(open("/root/net1.params", "rb").read())

ctx = tvm.cl(0)

mod = graph_runtime.create(loaded_json, loaded_lib, ctx)
mod.load_params(loaded_params)

#a = np.random.uniform(size=(1,3,112,112)).astype('float32')
img = misc.imread(os.path.expanduser("/root/img/11.png"))
resize_img = misc.imresize(img, [112, 112], interp='nearest')
a = transform_image(resize_img).astype('float32')

print("first run, need calc graph")

start = time.time()
mod.run(data=a)
done = time.time()

print('cost {}'.format(done-start))

img = misc.imread(os.path.expanduser("/root/img/roger1.jpg"))
resize_img = misc.imresize(img, [112, 112], interp='nearest')
a = transform_image(resize_img).astype('float32')
mod.run(data=a)
out1 = mod.get_output(0, tvm.nd.empty((128,))).asnumpy()
#print(out1)

img = misc.imread(os.path.expanduser("/root/img/roger2.jpg"))
resize_img = misc.imresize(img, [112, 112], interp='nearest')
a = transform_image(resize_img).astype('float32')
mod.run(data=a)
out2 = mod.get_output(0, tvm.nd.empty((128,))).asnumpy()
#print(out2)


img = misc.imread(os.path.expanduser("/root/img/libby2.jpg"))
resize_img = misc.imresize(img, [112, 112], interp='nearest')
a = transform_image(resize_img).astype('float32')
mod.run(data=a)
libby1 = mod.get_output(0, tvm.nd.empty((128,))).asnumpy()
#print(libby2)


img = misc.imread(os.path.expanduser("/root/img/libby3.jpg"))
resize_img = misc.imresize(img, [112, 112], interp='nearest')
a = transform_image(resize_img).astype('float32')
mod.run(data=a)
libby2 = mod.get_output(0, tvm.nd.empty((128,))).asnumpy()
#print(libby2)


#print(type(out1))
dist = np.sum(np.square(out1-out2))
print("roger1 and roger2: ",dist)

dist1 = np.sum(np.square(out1-libby1))
print("roger1 and libby1: ",dist1)

dist2 = np.sum(np.square(out2-libby2))
print("roger2 and libby2: ",dist2)

dist3 = np.sum(np.square(libby1-libby2))
print("libby1 and libby2: ",dist3)

#for i in range(0,100):
  

#set_input, get_output, run = gmodule["set_input"], gmodule["get_output"], gmodule["run"]
#gmodule["load_params"](loaded_params)

#set_input("x", tvm.nd.array(x_np))
#run()
#exit(0)

#ctx = tvm.gpu(0)
#gmodule = fcreate(loaded_json, loaded_lib, ctx.device_type, ctx.device_id)
#set_input("x", tvm.nd.array(x_np))
#run()
#out = tvm.nd.empty(shape)
#get_output(0, out)
#print(out.asnumpy())
