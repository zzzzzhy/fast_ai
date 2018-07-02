import cv2
import tvm
import numpy as np
import time
from scipy import misc
import os
import sklearn.preprocessing

from tvm.contrib import rpc, util, graph_runtime


def transform_image(image):
    #image = np.array(image) - np.array([123., 117., 104.])
    #image /= np.array([58.395, 57.12, 57.375])
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :]
    return image

def align_face(face):
    aligned_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    aligned_face = cv2.medianBlur(aligned_face,5)
    aligned_face = cv2.GaussianBlur(aligned_face,(5,5),0)
    return aligned_face

# tvm module for compiled functions.
loaded_lib = tvm.module.load('/root/net1.tar')
# json graph
loaded_json = open("/root/net1").read()
# parameters in binary
loaded_params = bytearray(open("/root/net1.params", "rb").read())

ctx = tvm.cl(0)

mod = graph_runtime.create(loaded_json, loaded_lib, ctx)
mod.load_params(loaded_params)

img = misc.imread(os.path.expanduser("/root/img/12.png"))
resize_img = misc.imresize(img, [112, 112], interp='bilinear')
a = transform_image(resize_img).astype('float32')
print("first run, need calc graph")

start = time.time()
mod.run(data=a)
done = time.time()

print('cost {}'.format(done-start))

img = np.load('/root/img/roger2.npy')
resize_img = misc.imresize(img, [112, 112], interp='bilinear')
a = transform_image(resize_img).astype('float32')

mod.run(data=a)
out1 = mod.get_output(0, tvm.nd.empty((128,))).asnumpy()

img = np.load('/root/img/roger3.npy')
resize_img = misc.imresize(img, [112, 112], interp='bilinear')
a = transform_image(resize_img).astype('float32')
mod.run(data=a)
out2 = mod.get_output(0, tvm.nd.empty((128,))).asnumpy()
#print(out2)

img = np.load('/root/img/libby1.npy')
resize_img = misc.imresize(img, [112, 112], interp='bilinear')
a = transform_image(resize_img).astype('float32')
mod.run(data=a)
libby1 = mod.get_output(0, tvm.nd.empty((128,))).asnumpy()
#print(libby2)

img = np.load('/root/img/libby2.npy')
resize_img = misc.imresize(img, [112, 112], interp='bilinear')
a = transform_image(resize_img).astype('float32')
mod.run(data=a)
libby2 = mod.get_output(0, tvm.nd.empty((128,))).asnumpy()
#print(libby2)

out1 = sklearn.preprocessing.normalize(out1).flatten()
out2 = sklearn.preprocessing.normalize(out2).flatten()
libby1 = sklearn.preprocessing.normalize(libby1).flatten()
libby2 = sklearn.preprocessing.normalize(libby2).flatten()

dist = np.sum(np.square(out1-out2))
print("roger1 and roger2: ",dist)

dist1 = np.sum(np.square(out1-libby1))
print("roger1 and libby1: ",dist1)

dist2 = np.sum(np.square(out2-libby2))
print("roger2 and libby2: ",dist2)

dist3 = np.sum(np.square(libby1-libby2))
print("libby1 and libby2: ",dist3)
